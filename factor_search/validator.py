from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .config import MetricThresholds
from .schemas import FactorCandidate

EvalFn = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]


@dataclass
class ValidationResult:
    accepted: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]
    # New: JSON-friendly stream for logging per candidate with accepted flag.
    # Each item has: {name, expression, type, meta, reason, tags, provenance, metrics, accepted}
    per_candidate: List[Dict[str, Any]]


class Validator:
    """
    Wraps the external performance evaluation/backtest API and applies metric thresholds.

    The evaluation function should accept:
        [{"name": ..., "expression": ...}, ...]
    and return (order-preserving):
        [{"metrics": {"ic": ..., "rank_ic": ..., "icir": ..., "winrate": ..., "stability": ...}}, ...]
    """

    def __init__(self, evaluate_fn: EvalFn):
        self.evaluate_fn = evaluate_fn

    def _zero_metrics(self) -> Dict[str, float]:
        # Default failure metrics
        return {
            "ic": 0.0,
            "rank_ic": 0.0,
            "ir": 0.0,
            "icir": 0.0,
            "rank_icir": 0.0,
            "turnover": 1.0,
            "n_dates": 0,
        }

    def validate(
        self,
        *,
        candidates: List[FactorCandidate],
        thresholds: MetricThresholds,
    ) -> ValidationResult:
        if not candidates:
            return ValidationResult(accepted=[], rejected=[], per_candidate=[])

        # Prepare input for eval API
        eval_input: List[Dict[str, Any]] = [
            {"name": c.name, "expression": c.expression} for c in candidates
        ]

        # Call eval function robustly
        try:
            results = self.evaluate_fn(eval_input) or []
        except Exception:
            # If the API fails, treat all as zero metrics so the pipeline still completes
            results = [{"metrics": self._zero_metrics()} for _ in eval_input]

        # Defensive: ensure results length matches candidates
        if len(results) != len(candidates):
            # Pad or trim to match length (we keep order)
            fixed: List[Dict[str, Any]] = []
            for i in range(len(candidates)):
                if i < len(results) and isinstance(results[i], dict):
                    r = results[i]
                else:
                    r = {"metrics": self._zero_metrics()}
                # Ensure 'metrics' exists
                r_metrics = r.get("metrics") or self._zero_metrics()
                r["metrics"] = r_metrics
                fixed.append(r)
            results = fixed

        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        per_candidate: List[Dict[str, Any]] = []

        for cand, res in zip(candidates, results):
            metrics = res.get("metrics") or self._zero_metrics()

            ic = float(metrics.get("ic", 0.0))
            rank_ic = float(metrics.get("rank_ic", 0.0))
            icir = float(metrics.get("icir", 0.0))
            winrate = float(metrics.get("winrate", 0.0))
            stability = float(metrics.get("stability", 0.0))

            # Your current absolute-threshold logic
            ok = True
            if abs(ic) < thresholds.ic_min:
                ok = False
            if abs(rank_ic) < thresholds.rank_ic_min:
                ok = False
            if abs(icir) < thresholds.icir_min:
                ok = False
            # If you want to enforce these again, uncomment:
            # if winrate < thresholds.winrate_min:
            #     ok = False
            # if stability < thresholds.stability_min:
            #     ok = False

            # JSON-friendly enriched record
            enriched = {
                "name": cand.name,
                "expression": cand.expression,
                "type": cand.doc_type,       # "origin" or "search"
                "meta": getattr(cand, "meta", {}),  # origin/mutation/crossover and details
                "reason": cand.reason,
                "tags": cand.tags,
                "provenance": cand.provenance,      # includes agent_id, round, attempt
                "metrics": metrics,
            }

            # Stream for per-file record logging
            per_candidate.append({**enriched, "accepted": ok})

            if ok:
                accepted.append(enriched)
            else:
                rejected.append(enriched)

        return ValidationResult(accepted=accepted, rejected=rejected, per_candidate=per_candidate)
