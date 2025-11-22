"""
Multi-agent factor search package.

Exposes:
- SearchTask, ControllerConfig, BacktestConfig, MetricThresholds
- Controller: multi-agent orchestrator
- Validator: wraps eval/backtest API
- SearcherAgent: single mutation/crossover agent
"""

from .config import SearchTask, ControllerConfig, BacktestConfig, MetricThresholds
from .controller import Controller
from .validator import Validator
from .searcher_agent import SearcherAgent
from .schemas import FactorCandidate, SearcherReport

__all__ = [
    "SearchTask",
    "ControllerConfig",
    "BacktestConfig",
    "MetricThresholds",
    "Controller",
    "Validator",
    "SearcherAgent",
    "FactorCandidate",
    "SearcherReport",
]
