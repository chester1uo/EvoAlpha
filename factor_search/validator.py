from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .config import MetricThresholds
from .schemas import FactorCandidate

EvalFn = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]


@dataclass
class ValidationResult:
    accepted: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]


class Validator:
    """
    Wraps the external performance evaluation/backtest API and applies metric thresholds.

    The evaluation function should accept:
        [{"name": ..., "expression": ...}, ...]
    and return:
        [{"metrics": {"ic": ..., "rank_ic": ..., "icir": ..., "winrate": ..., "stability": ...}}, ...]
    """

    def __init__(self, evaluate_fn: EvalFn):
        self.evaluate_fn = evaluate_fn

    def validate(
        self,
        *,
        candidates: List[FactorCandidate],
        thresholds: MetricThresholds,
    ) -> ValidationResult:
        if not candidates:
            return ValidationResult(accepted=[], rejected=[])

        eval_input: List[Dict[str, Any]] = [
            {"name": c.name, "expression": c.expression} for c in candidates
        ]
        results = self.evaluate_fn(eval_input)

        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        # import pdb; pdb.set_trace()
        for cand, res in zip(candidates, results):
            metrics = res.get("metrics", {})
            ic = metrics.get("ic", -9.0)
            rank_ic = metrics.get("rank_ic", -9.0)
            icir = metrics.get("icir", -9.0)
            winrate = metrics.get("winrate", -9.0)
            stability = metrics.get("stability", 0.0)

            ok = True
            if abs(ic) < thresholds.ic_min:
                ok = False
            if abs(rank_ic) < thresholds.rank_ic_min:
                ok = False
            if abs(icir) < thresholds.icir_min:
                ok = False
            # if winrate < thresholds.winrate_min:
            #     ok = False
            # if stability < thresholds.stability_min:
            #     ok = False

            enriched = {
                "name": cand.name,
                "expression": cand.expression,
                "type": cand.doc_type,       # "origin" or "search"
                "meta": cand.meta,           # classification info (origin/mutation/crossover)
                "reason": cand.reason,
                "tags": cand.tags,
                "provenance": cand.provenance,
                "metrics": metrics,
            }

            if ok:
                accepted.append(enriched)
            else:
                rejected.append(enriched)

        return ValidationResult(accepted=accepted, rejected=rejected)
