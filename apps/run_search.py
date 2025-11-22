"""
Main entry point: run multi-agent factor search starting from the MongoDB database.

Pipeline:
1. Load origin factors from MongoDB as seeds.
2. Run baseline evaluation via batch_evaluate_factors_via_api.
3. Launch multi-agent search + validator.
4. Persist accepted search factors back into MongoDB.
"""

import argparse
import os
from typing import Any, Dict, List

from factor_search.config import (
    BacktestConfig,
    ControllerConfig,
    MetricThresholds,
    SearchTask,
)
from factor_search.controller import Controller
from factor_search.db import FactorRepository
from factor_search.quality import default_quality_check

# Provided by your existing codebase
from api.factor_eval_client import batch_evaluate_factors_via_api


def baseline_evaluate_seeds(seeds: List[Dict[str, Any]]) -> None:
    """
    Evaluate IC / RankIC / ICIR / etc. for seed factors and attach metrics in-place.
    """
    if not seeds:
        return
    eval_input = [{"name": s["name"], "expression": s["expression"]} for s in seeds]
    results = batch_evaluate_factors_via_api(eval_input)
    for s, res in zip(seeds, results):
        s["metrics"] = res.get("metrics", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-agent factor search.")
    parser.add_argument(
        "--mongo-uri",
        default=os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
        help="MongoDB URI (default: env MONGODB_URI or mongodb://localhost:27017)",
    )
    parser.add_argument(
        "--db-name",
        default="factor_search",
        help="MongoDB database name (default: factor_search)",
    )
    parser.add_argument(
        "--collection",
        default="factors",
        help="MongoDB collection name (default: factors)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of search rounds (default: 3)",
    )
    parser.add_argument(
        "--factors-per-round",
        type=int,
        default=10,
        help="Total number of factors to generate per round (default: 30)",
    )
    parser.add_argument(
        "--task-text",
        type=str,
        default=(
            "Search for nonlinear volatility-aware factors for the NASDAQ-100. "
            "Focus on medium-term signals; avoid simple price ratios; "
            "prefer conditional logic and multi-window structures."
        ),
        help="User search request / task description.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1) Connect to MongoDB and load seed factors
    # ------------------------------------------------------------------
    repo = FactorRepository(
        uri=args.mongo_uri, db_name=args.db_name, collection_name=args.collection
    )

    ctrl_cfg = ControllerConfig(
        rounds=args.rounds,
        factors_per_round=args.factors_per_round,
        num_searchers=5,
        mutation_share=0.5,
        crossover_share=0.5,
        seeds_top_k=12,
        persona_refresh_prob=0.3,
        llm_model="gpt-4.1-mini",
        temperature=1.2,
        max_retries_per_searcher=5,
        seed_pool_size=40,
    )

    seeds = repo.get_seeds(limit=ctrl_cfg.seed_pool_size, include_search=False)
    if not seeds:
        print(
            "No seed factors found in MongoDB. "
            "Please run apps/init_factors_from_json.py first to load origin factors."
        )
        return

    print(f"Loaded {len(seeds)} seed factors from MongoDB for initial pool.")

    # ------------------------------------------------------------------
    # 2) Baseline evaluation of seeds
    # ------------------------------------------------------------------
    baseline_evaluate_seeds(seeds)
    repo.update_metrics_bulk(seeds)
    print("Baseline metrics evaluated for seed factors and stored in MongoDB.")

    # ------------------------------------------------------------------
    # 3) Build configs
    # ------------------------------------------------------------------
    task = SearchTask(
        user_request=args.task_text,
        target_market="US",
        universe="NASDAQ-100",
        style="volatility-sensitive + nonlinear transformation",
        horizon="10-40 days",
    )

    backtest_cfg = BacktestConfig(
        market="US",
        universe="NASDAQ-100",
        benchmark="NDX",
        start_date="2021-01-01",
        end_date="2022-12-31",
        neutralization="industry",
        freq="D",
    )

    thresholds = MetricThresholds(
        ic_min=0.01,
        rank_ic_min=0.01,
        icir_min=0.02,
        winrate_min=0.52,
        stability_min=0.0,
    )

    # ------------------------------------------------------------------
    # 4) Run multi-agent controller
    # ------------------------------------------------------------------
    controller = Controller(
        repo=repo,
        seeds=seeds,
        quality_check_fn=default_quality_check,
        evaluate_fn=batch_evaluate_factors_via_api,
    )

    summary = controller.run(task, ctrl_cfg, backtest_cfg, thresholds)

    print("------------------------------------------------------------")
    print("Search completed.")
    print(f"Final pool size:        {len(summary['final_pool'])}")
    print(f"Total accepted factors: {len(summary['accepted_factors'])}")
    print(f"Total rejected factors: {len(summary['rejected_factors'])}")
    if summary["round_summaries"]:
        print("Round summaries:")
        for rs in summary["round_summaries"]:
            print(
                f"  Round {rs['round']}: "
                f"candidates={rs['num_candidates']}, "
                f"accepted={rs['accepted']}, "
                f"rejected={rs['rejected']}, "
                f"best_ic={rs['best_ic']:.6f}, "
                f"elapsed={rs['elapsed_sec']:.1f}s"
            )


if __name__ == "__main__":
    main()
