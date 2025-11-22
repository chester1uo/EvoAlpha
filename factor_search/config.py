from dataclasses import dataclass, field
from typing import List


@dataclass
class SearchTask:
    """
    High-level user intent and soft/hard constraints for factor search.
    """
    user_request: str
    target_market: str = "US"
    universe: str = "NASDAQ-100"
    style: str = "volatility-sensitive + nonlinear transforms"
    horizon: str = "10-40 days"
    required_components: List[str] = field(
        default_factory=lambda: ["Std", "ATR", "Log", "Slope", "Rank"]
    )
    avoided_operators: List[str] = field(
        default_factory=lambda: ["EMA", "SMA", "MA", "simple_ratio"]
    )


@dataclass
class ControllerConfig:
    """
    Algorithmic knobs for the multi-agent controller.
    """
    num_searchers: int = 6
    factors_per_round: int = 30
    rounds: int = 5
    mutation_share: float = 0.3         # fraction of searchers doing mutation
    crossover_share: float = 0.7        # fraction doing crossover
    seeds_top_k: int = 12               # how many top seeds to expose to LLM
    persona_refresh_prob: float = 0.25  # per-agent chance to change persona each round
    llm_model: str = "gpt-4.1-mini"
    temperature: float = 1.1
    max_retries_per_searcher: int = 5
    seed_pool_size: int = 60            # keep this many best factors in pool over rounds


@dataclass
class BacktestConfig:
    """
    Parameters forwarded to the external evaluation/backtest API.
    You can adapt this to your own evaluation service.
    """
    market: str
    universe: str
    benchmark: str
    start_date: str   # "YYYY-MM-DD"
    end_date: str     # "YYYY-MM-DD"
    neutralization: str = "industry"
    freq: str = "D"


@dataclass
class MetricThresholds:
    """
    Hard performance thresholds for accepting factors.
    """
    ic_min: float = 0.02
    rank_ic_min: float = 0.02
    icir_min: float = 0.2
    winrate_min: float = 0.52
    stability_min: float = 0.0
