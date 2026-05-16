from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_backtest.experiments import (
    ResearchConfig,
    create_fixture_prices,
    parameter_grid,
    run_research,
    run_train_test,
)
from quant_backtest.reports import save_research_outputs
from quant_backtest.strategies import SmaCrossoverStrategy, SmaParameters, build_fallback_weights


def tiny_config(tmp_path: Path) -> ResearchConfig:
    return ResearchConfig(
        start="2020-01-01",
        end="2021-12-31",
        initial_capital=10_000.0,
        base_ticker="AAPL",
        universe=["AAPL", "MSFT", "SPY", "QQQ"],
        cost_bps=[0.0, 5.0, 10.0, 20.0],
        short_windows=[5, 10],
        long_windows=[20, 40],
        train_start="2020-01-01",
        train_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2021-12-31",
        walk_forward_train_years=1,
        walk_forward_test_years=1,
        walk_forward_step_years=1,
        output_dir=str(tmp_path),
    )


def test_sma_spread_threshold_and_momentum_filter() -> None:
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    price = pd.Series(range(100, 190), index=dates, dtype=float)
    params = SmaParameters(short_window=5, long_window=20, spread_threshold=0.01, momentum_window=20)

    signals = SmaCrossoverStrategy(params).generate(price)

    assert signals.target_position.iloc[:20].sum() == 0.0
    assert signals.target_position.iloc[-1] == 1.0
    assert signals.spread.iloc[-1] > 0.01


def test_partial_exposure_uses_half_weight_for_weak_trend() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    price = pd.Series([100.0] * 40 + [101.0] * 40, index=dates)
    params = SmaParameters(short_window=5, long_window=20, spread_threshold=0.05, partial_exposure=True)

    signals = SmaCrossoverStrategy(params).generate(price)

    assert 0.5 in set(signals.target_position.dropna())
    assert signals.target_position.max() <= 1.0


def test_fallback_weights_allocate_remaining_capital() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    target = pd.Series([0.0, 0.5, 1.0], index=dates)

    weights = build_fallback_weights("AAPL", "SPY", target)

    assert list(weights["AAPL"]) == [0.0, 0.5, 1.0]
    assert list(weights["SPY"]) == [1.0, 0.5, 0.0]
    assert all(weights.sum(axis=1) == 1.0)


def test_parameter_grid_filters_invalid_pairs(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)

    assert parameter_grid(config) == [(5, 20), (5, 40), (10, 20), (10, 40)]


def test_train_test_split_has_no_overlap(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    prices = create_fixture_prices(config)

    table = run_train_test(prices, config)

    assert set(table["label"]) == {"train", "test"}
    assert table.loc[table["label"] == "train", "cagr"].notna().all()
    assert table.loc[table["label"] == "test", "cagr"].notna().all()


def test_research_smoke_with_fixture_data_and_reports(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)

    result = run_research(config, fixture_data=True)
    save_research_outputs(result, tmp_path)

    assert (tmp_path / "cost_sensitivity.csv").exists()
    assert (tmp_path / "parameter_sweep.csv").exists()
    assert (tmp_path / "research_report.xlsx").exists()
    assert not result.model_leaderboard.empty


def test_leaderboard_passes_selection_constraints(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)

    result = run_research(config, fixture_data=True)
    passing = result.model_leaderboard[result.model_leaderboard["passes_selection"]]

    assert (passing["cagr"] > 0).all()
    assert (passing["max_drawdown"] >= passing["benchmark_max_drawdown"] - 0.05).all()
    assert (passing["turnover"] < 8.0).all()
