"""Simple SMA crossover backtest package."""

from .backtest import BacktestConfig, BacktestResult, run_sma_backtest
from .experiments import ResearchConfig, ResearchResult, run_research

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "ResearchConfig",
    "ResearchResult",
    "run_research",
    "run_sma_backtest",
]
