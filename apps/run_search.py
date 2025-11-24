"""
Main entry point: run multi-agent factor search starting from the MongoDB database.

Pipeline:
1. Load origin factors from MongoDB as seeds.
2. Run baseline evaluation via batch_evaluate_factors_via_api.
3. Launch multi-agent search + validator.
4. Persist accepted search factors back into MongoDB.
"""
# apps/run_search.py
import argparse
import os
import time
from datetime import datetime
from typing import List, Dict, Any

from factor_search.config import (
    SearchTask,
    ControllerConfig,
    BacktestConfig,
    MetricThresholds,
)
from factor_search.db.mongo import FactorRepository
from factor_search.controller import Controller
from factor_search.utils import dedup_by_expression, rank_by_ic
from factor_search.quality import default_quality_check  # your quality check
from api.factor_eval_client import batch_evaluate_factors_via_api  # external API


def baseline_eval_and_update(repo: FactorRepository, seeds: List[Dict[str, Any]],
                             market: str, start_date: str, end_date: str, label: str) -> List[Dict[str, Any]]:
    """
    Evaluate baseline metrics for seeds and update local copies.
    Note: This only updates the in-memory seeds list used by the controller;
          if you want to persist metrics to Mongo before the run, add a repo method to do so.
    """
    eval_input = [{"name": s["name"], "expression": s["expression"]} for s in seeds]
    results = batch_evaluate_factors_via_api(
        eval_input,
        market=market,
        start_date=start_date,
        end_date=end_date,
        label=label,
    )
    for s, r in zip(seeds, results):
        s["metrics"] = r.get("metrics", {})
    return seeds


def parse_args():
    p = argparse.ArgumentParser("Run multi-agent factor search")
    p.add_argument("--mongo-uri", default=os.environ.get("MONGODB_URI", "mongodb://localhost:27017"))
    p.add_argument("--db-name", default="factor_search")
    p.add_argument("--collection", default="factors")

    p.add_argument("--run-name", default=datetime.utcnow().strftime("run_%Y%m%d_%H%M%S"))
    p.add_argument("--runs-dir", default="./runs/ea_search")

    # Search task
    p.add_argument("--task-text", default="Discover robust cross-sectional alphas for large-cap equities.")
    p.add_argument("--required", nargs="*", default=["Rank", "Std"])
    p.add_argument("--avoid", nargs="*", default=[])
    p.add_argument("--market", default="csi300")
    p.add_argument("--universe", default="CSI300")
    p.add_argument("--style", default="momentum")
    p.add_argument("--horizon", default="10-20d")

    # Backtest window / eval label
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2024-01-01")
    p.add_argument("--benchmark", default="CSI300")
    p.add_argument("--label", default="close_return")

    # Controller config
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--num-searchers", type=int, default=4)
    p.add_argument("--factors-per-round", type=int, default=24)
    p.add_argument("--mutation-share", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--seeds-top-k", type=int, default=12)
    p.add_argument("--seed-pool-size", type=int, default=60)
    p.add_argument("--persona-refresh-prob", type=float, default=0.2)

    # Thresholds
    p.add_argument("--ic-min", type=float, default=0.01)
    p.add_argument("--rank-ic-min", type=float, default=0.01)
    p.add_argument("--icir-min", type=float, default=0.05)
    # p.add_argument("--winrate-min", type=float, default=0.0)
    # p.add_argument("--stability-min", type=float, default=0.0)

    return p.parse_args()


def main():
    args = parse_args()

    save_dir = os.path.join(args.runs_dir, args.run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Logs will be written under: {save_dir}")

    repo = FactorRepository(uri=args.mongo_uri, db_name=args.db_name, collection_name=args.collection)

    # Load origin seeds
    seeds = repo.get_seeds(limit=200, include_search=False)
    if not seeds:
        raise SystemExit("No origin factors found. Please load with apps/init_factors_from_json.py first.")

    # Baseline eval of seeds (in-memory)
    seeds = baseline_eval_and_update(
        repo, seeds, market=args.market, start_date=args.start, end_date=args.end, label=args.label
    )

    # Sort seeds by IC
    seeds = rank_by_ic(seeds)

    # Compose configs
    task = SearchTask(
        user_request=args.task_text,
        target_market=args.market,
        universe=args.universe,
        style=args.style,
        horizon=args.horizon,
        required_components=args.required,
        avoided_operators=args.avoid,
    )

    ctrl_cfg = ControllerConfig(
        num_searchers=args.num_searchers,
        mutation_share=args.mutation_share,
        llm_model=args.model,
        temperature=args.temperature,
        max_retries_per_searcher=args.max_retries,
        factors_per_round=args.factors_per_round,
        rounds=args.rounds,
        seeds_top_k=args.seeds_top_k,
        seed_pool_size=args.seed_pool_size,
        persona_refresh_prob=args.persona_refresh_prob,
    )

    backtest_cfg = BacktestConfig(
        market=args.market,
        universe=args.universe,
        benchmark=args.benchmark,
        start_date=args.start,
        end_date=args.end,
        # label=args.label,
    )

    thresholds = MetricThresholds(
        ic_min=args.ic_min,
        rank_ic_min=args.rank_ic_min,
        icir_min=args.icir_min,
        # winrate_min=args.winrate_min,
        # stability_min=args.stability_min,
    )

    controller = Controller(
        repo=repo,
        seeds=seeds,
        quality_check_fn=default_quality_check,     # calls your /check or custom logic
        evaluate_fn=lambda facs: batch_evaluate_factors_via_api(
            facs, market=args.market, start_date=args.start, end_date=args.end, label=args.label
        ),
    )

    summary = controller.run(
        task=task,
        ctrl_cfg=ctrl_cfg,
        backtest_cfg=backtest_cfg,
        thresholds=thresholds,
        save_dir=save_dir,     # <<<<<< write ./raw and ./record here
    )

    print("Search complete.")
    print("Accepted factors:", len(summary["accepted_factors"]))
    print("Saved logs under:", save_dir)


if __name__ == "__main__":
    main()
