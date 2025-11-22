import random
import time
from typing import Any, Callable, Dict, List
from tqdm import tqdm
from factor_search.audit_log import get_audit_logger

from .config import BacktestConfig, ControllerConfig, MetricThresholds, SearchTask
from .db import FactorRepository
from .personas import random_persona
from .searcher_agent import SearcherAgent
from .schemas import FactorCandidate, SearcherReport
from .utils import dedup_by_expression, rank_by_ic, select_seed_pool
from .validator import ValidationResult, Validator

QualityCheckFn = Callable[[Dict[str, Any], Dict[str, Any]], bool]
EvalFn = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]


class Controller:
    """
    Multi-agent controller that orchestrates searchers and validation.

    Typical usage:

        repo = FactorRepository(uri=...)
        seeds = repo.get_seeds(limit=60, include_search=False)
        controller = Controller(repo=repo, seeds=seeds, quality_check_fn=..., evaluate_fn=...)
        summary = controller.run(task, ctrl_cfg, backtest_cfg, thresholds)
    """

    def __init__(
        self,
        *,
        repo: FactorRepository,
        seeds: List[Dict[str, Any]],
        quality_check_fn: QualityCheckFn,
        evaluate_fn: EvalFn,
    ) -> None:
        self.repo = repo
        self.quality_check_fn = quality_check_fn
        self.validator = Validator(evaluate_fn)

        # Ensure each seed has a metrics dict
        for s in seeds:
            s.setdefault("metrics", {})
        self.pool: List[Dict[str, Any]] = list(seeds)

    # ------------------------------------------------------------------ #
    # Searcher management
    # ------------------------------------------------------------------ #

    def _spawn_searchers(self, cfg: ControllerConfig) -> List[SearcherAgent]:
        n = cfg.num_searchers
        n_mutation = int(round(n * cfg.mutation_share))
        n_mutation = max(0, min(n, n_mutation))
        n_crossover = n - n_mutation

        modes = ["mutation"] * n_mutation + ["crossover"] * n_crossover
        random.shuffle(modes)

        searchers: List[SearcherAgent] = []
        for i in range(n):
            persona = random_persona()
            mode = modes[i]
            agent = SearcherAgent(
                mode=mode,
                persona_name=persona.name,
                persona_description=persona.description,
                model=cfg.llm_model,
                temperature=cfg.temperature,
                max_retries=cfg.max_retries_per_searcher,
                enable_reason=True,
                quality_check_fn=self.quality_check_fn,
            )
            searchers.append(agent)
        return searchers

    def _maybe_refresh_personas(
        self, searchers: List[SearcherAgent], refresh_prob: float
    ) -> None:
        from .personas import random_persona as _rp

        for s in searchers:
            if random.random() < refresh_prob:
                persona = _rp()
                s.persona_name = persona.name
                s.persona_description = persona.description

    @staticmethod
    def _dedup_candidates(candidates: List[FactorCandidate]) -> List[FactorCandidate]:
        """
        Deduplicate FactorCandidate objects by normalized expression.
        """
        import re

        seen = set()
        out: List[FactorCandidate] = []
        for c in candidates:
            key = re.sub(r"\s+", "", c.expression)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    # ------------------------------------------------------------------ #
    # Main run loop
    # ------------------------------------------------------------------ #

    def run(
        self,
        task: SearchTask,
        ctrl_cfg: ControllerConfig,
        backtest_cfg: BacktestConfig,
        thresholds: MetricThresholds,
    ) -> Dict[str, Any]:
        """
        Run multi-round factor search and validation.
        """
        searchers = self._spawn_searchers(ctrl_cfg)
        round_summaries: List[Dict[str, Any]] = []
        all_searcher_reports: List[SearcherReport] = []
        accepted_overall: List[Dict[str, Any]] = []
        rejected_overall: List[Dict[str, Any]] = []

        for round_id in tqdm(range(1, ctrl_cfg.rounds + 1), desc="EA Search Rounds"):
            t_round_start = time.time()

            if round_id > 1:
                self._maybe_refresh_personas(searchers, ctrl_cfg.persona_refresh_prob)

            # Select seeds for this round
            seeds = select_seed_pool(self.pool, top_k=ctrl_cfg.seeds_top_k)

            # Determine per-searcher quotas
            total = ctrl_cfg.factors_per_round
            base_quota = total // len(searchers)
            remainder = total - base_quota * len(searchers)
            per_searcher_quota = [
                base_quota + (1 if i < remainder else 0) for i in range(len(searchers))
            ]

            # Run all searchers
            round_candidates: List[FactorCandidate] = []
            round_reports: List[SearcherReport] = []

            for agent, quota in zip(searchers, per_searcher_quota):
                if quota <= 0:
                    continue

                context = {
                    "backtest": {
                        "market": backtest_cfg.market,
                        "universe": backtest_cfg.universe,
                        "benchmark": backtest_cfg.benchmark,
                        "start_date": backtest_cfg.start_date,
                        "end_date": backtest_cfg.end_date,
                    },
                    "task": task.user_request,
                    "mode": agent.mode,
                }

                cands, report = agent.search(
                    user_request=task.user_request,
                    seeds=seeds,
                    required_components=task.required_components,
                    avoided_operators=task.avoided_operators,
                    market=task.target_market,
                    universe=task.universe,
                    style=task.style,
                    horizon=task.horizon,
                    n_factors=quota,
                    round_id=round_id,
                    context=context,
                )
                round_candidates.extend(cands)
                round_reports.append(report)

            all_searcher_reports.extend(round_reports)

            # Deduplicate and validate
            unique_candidates = self._dedup_candidates(round_candidates)

            validation: ValidationResult = self.validator.validate(
                candidates=unique_candidates, thresholds=thresholds
            )

            accepted_overall.extend(validation.accepted)
            rejected_overall.extend(validation.rejected)

            # Persist newly accepted search factors to MongoDB
            self.repo.store_search_results(validation.accepted)

            # Update in-memory pool with accepted factors
            self.pool.extend(validation.accepted)
            self.pool = dedup_by_expression(self.pool)
            self.pool = rank_by_ic(self.pool)
            if ctrl_cfg.seed_pool_size > 0:
                k = max(1, min(ctrl_cfg.seed_pool_size, len(self.pool)))
                self.pool = self.pool[:k]

            t_round = time.time() - t_round_start
            best_ic = self.pool[0]["metrics"].get("ic", 0.0) if self.pool else 0.0

            round_summary = {
                "round": round_id,
                "num_candidates": len(unique_candidates),
                "accepted": len(validation.accepted),
                "rejected": len(validation.rejected),
                "best_ic": best_ic,
                "elapsed_sec": t_round,
            }
            round_summaries.append(round_summary)

            print(
                f"[Controller] Round {round_id}/{ctrl_cfg.rounds}: "
                f"candidates={len(unique_candidates)}, accepted={len(validation.accepted)}, "
                f"rejected={len(validation.rejected)}, best_ic={best_ic:.6f}, elapsed={t_round:.1f}s"
            )

        return {
            "final_pool": self.pool,
            "accepted_factors": accepted_overall,
            "rejected_factors": rejected_overall,
            "round_summaries": round_summaries,
            "searcher_reports": [r.__dict__ for r in all_searcher_reports],
        }
