from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .backtest import calculate_closed_trade_returns, calculate_win_rate
from .costs import BpsCost
from .data import default_end_date, download_adjusted_close
from .engine import EngineConfig, run_weight_backtest
from .metrics import (
    annualized_turnover,
    metrics_table,
    summarize_performance,
)
from .strategies import (
    SmaCrossoverStrategy,
    SmaParameters,
    build_fallback_weights,
    build_single_asset_weights,
)


DEFAULT_VARIANTS = (
    "long_cash",
    "fallback_spy",
    "fallback_qqq",
    "partial_exposure",
    "spread_threshold",
    "momentum_3m",
    "momentum_6m",
)


@dataclass(frozen=True)
class ResearchConfig:
    start: str
    end: str | None
    initial_capital: float
    base_ticker: str
    universe: list[str]
    cost_bps: list[float]
    short_windows: list[int]
    long_windows: list[int]
    train_start: str
    train_end: str
    test_start: str
    test_end: str | None
    walk_forward_train_years: int
    walk_forward_test_years: int
    walk_forward_step_years: int
    spread_threshold: float = 0.01
    output_dir: str = "outputs"


@dataclass(frozen=True)
class ResearchResult:
    prices: pd.DataFrame
    baseline_curve: pd.DataFrame
    base_backtest: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    parameter_sweep: pd.DataFrame
    train_test_results: pd.DataFrame
    walk_forward_results: pd.DataFrame
    multi_asset_results: pd.DataFrame
    model_leaderboard: pd.DataFrame


def load_research_config(path: Path) -> ResearchConfig:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    period = raw["period"]
    train_test = raw["train_test"]
    walk_forward = raw["walk_forward"]
    grid = raw["sma_grid"]
    return ResearchConfig(
        start=str(period["start"]),
        end=None if period.get("end") in (None, "latest") else str(period["end"]),
        initial_capital=float(raw.get("initial_capital", 10_000.0)),
        base_ticker=str(raw["base_ticker"]).upper(),
        universe=[str(ticker).upper() for ticker in raw["universe"]],
        cost_bps=[float(value) for value in raw["cost_bps"]],
        short_windows=[int(value) for value in grid["short"]],
        long_windows=[int(value) for value in grid["long"]],
        train_start=str(train_test["train_start"]),
        train_end=str(train_test["train_end"]),
        test_start=str(train_test["test_start"]),
        test_end=None if train_test.get("test_end") in (None, "latest") else str(train_test["test_end"]),
        walk_forward_train_years=int(walk_forward["train_years"]),
        walk_forward_test_years=int(walk_forward["test_years"]),
        walk_forward_step_years=int(walk_forward["step_years"]),
        spread_threshold=float(raw.get("spread_threshold", 0.01)),
        output_dir=str(raw.get("output_dir", "outputs")),
    )


def run_research(config: ResearchConfig, fixture_data: bool = False) -> ResearchResult:
    prices = create_fixture_prices(config) if fixture_data else _download_prices(config)
    prices = prices.loc[config.start : config.end or prices.index.max()]

    base_params = SmaParameters(short_window=20, long_window=100)
    baseline = evaluate_strategy(
        prices=prices,
        ticker=config.base_ticker,
        params=base_params,
        variant="long_cash",
        cost_bps=10.0,
        initial_capital=config.initial_capital,
        label="baseline",
    )
    parameter_sweep = run_parameter_sweep(prices, config)
    train_prices = prices.loc[config.train_start : config.train_end]
    selected_params = select_best_parameters(run_parameter_sweep(train_prices, config, period_name="train"), config)

    cost_sensitivity = run_cost_sensitivity(prices, config, selected_params)
    train_test_results = run_train_test(prices, config)
    walk_forward_results = run_walk_forward(prices, config)
    multi_asset_results = run_multi_asset(prices, config, selected_params)
    leaderboard = run_model_leaderboard(prices, config, selected_params)

    return ResearchResult(
        prices=prices,
        baseline_curve=baseline["curve"],
        base_backtest=baseline["metrics"],
        cost_sensitivity=cost_sensitivity,
        parameter_sweep=parameter_sweep,
        train_test_results=train_test_results,
        walk_forward_results=walk_forward_results,
        multi_asset_results=multi_asset_results,
        model_leaderboard=leaderboard,
    )


def run_parameter_sweep(prices: pd.DataFrame, config: ResearchConfig, period_name: str = "full") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for short_window, long_window in parameter_grid(config):
        params = SmaParameters(short_window=short_window, long_window=long_window)
        result = evaluate_strategy(
            prices=prices,
            ticker=config.base_ticker,
            params=params,
            variant="long_cash",
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label=period_name,
        )
        rows.append(result["row"])
    table = pd.DataFrame(rows)
    table = add_parameter_stability(table)
    return table.sort_values("sharpe", ascending=False).reset_index(drop=True)


def run_cost_sensitivity(prices: pd.DataFrame, config: ResearchConfig, selected_params: SmaParameters) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scenarios = [
        ("base_20_100", SmaParameters(20, 100)),
        ("selected", selected_params),
    ]
    for scenario_name, params in scenarios:
        for cost_bps in config.cost_bps:
            result = evaluate_strategy(
                prices=prices,
                ticker=config.base_ticker,
                params=params,
                variant="long_cash",
                cost_bps=cost_bps,
                initial_capital=config.initial_capital,
                label=scenario_name,
            )
            rows.append(result["row"])
    table = pd.DataFrame(rows)
    return table.sort_values(["label", "cost_bps"]).reset_index(drop=True)


def run_train_test(prices: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    train_prices = prices.loc[config.train_start : config.train_end]
    test_prices = prices.loc[config.test_start : config.test_end or prices.index.max()]
    train_sweep = run_parameter_sweep(train_prices, config, period_name="train")
    selected = select_best_parameters(train_sweep, config)
    rows = []
    for period_name, period_prices in [("train", train_prices), ("test", test_prices)]:
        result = evaluate_strategy(
            prices=period_prices,
            ticker=config.base_ticker,
            params=selected,
            variant="long_cash",
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label=period_name,
        )
        rows.append(result["row"] | {"selected_on": "train"})
    return pd.DataFrame(rows)


def run_walk_forward(prices: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start = pd.Timestamp(config.start)
    end = pd.Timestamp(config.end or prices.index.max())
    train_offset = pd.DateOffset(years=config.walk_forward_train_years)
    test_offset = pd.DateOffset(years=config.walk_forward_test_years)
    step_offset = pd.DateOffset(years=config.walk_forward_step_years)

    train_start = start
    window_id = 1
    while True:
        train_end = train_start + train_offset - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + test_offset - pd.DateOffset(days=1)
        if test_start > end:
            break
        if test_end > end:
            test_end = end

        train_prices = prices.loc[train_start:train_end]
        test_prices = prices.loc[test_start:test_end]
        if len(train_prices) > 250 and len(test_prices) > 20:
            selected = select_best_parameters(run_parameter_sweep(train_prices, config, period_name="wf_train"), config)
            result = evaluate_strategy(
                prices=test_prices,
                ticker=config.base_ticker,
                params=selected,
                variant="long_cash",
                cost_bps=10.0,
                initial_capital=config.initial_capital,
                label="walk_forward_test",
            )
            rows.append(
                result["row"]
                | {
                    "window_id": window_id,
                    "train_start": train_start.date().isoformat(),
                    "train_end": train_end.date().isoformat(),
                    "test_start": test_start.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                }
            )
        train_start = train_start + step_offset
        window_id += 1
    return pd.DataFrame(rows)


def run_multi_asset(prices: pd.DataFrame, config: ResearchConfig, params: SmaParameters) -> pd.DataFrame:
    rows = []
    for ticker in config.universe:
        result = evaluate_strategy(
            prices=prices,
            ticker=ticker,
            params=params,
            variant="long_cash",
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label="multi_asset",
        )
        rows.append(result["row"])
    portfolio = evaluate_equal_weight_signal_portfolio(prices, config.universe, params, config)
    rows.append(portfolio)
    table = pd.DataFrame(rows)
    table["beats_benchmark_cagr"] = table["cagr"] > table["benchmark_cagr"]
    table["beats_benchmark_sharpe"] = table["sharpe"] > table["benchmark_sharpe"]
    return table.sort_values("sharpe", ascending=False).reset_index(drop=True)


def run_model_leaderboard(prices: pd.DataFrame, config: ResearchConfig, selected_params: SmaParameters) -> pd.DataFrame:
    rows = []
    evaluation_prices = prices.loc[config.test_start : config.test_end or prices.index.max()]
    variants = [
        ("long_cash", selected_params),
        ("fallback_spy", selected_params),
        ("fallback_qqq", selected_params),
        ("partial_exposure", SmaParameters(selected_params.short_window, selected_params.long_window, partial_exposure=True)),
        (
            "spread_threshold",
            SmaParameters(
                selected_params.short_window,
                selected_params.long_window,
                spread_threshold=config.spread_threshold,
            ),
        ),
        ("momentum_3m", SmaParameters(selected_params.short_window, selected_params.long_window, momentum_window=63)),
        ("momentum_6m", SmaParameters(selected_params.short_window, selected_params.long_window, momentum_window=126)),
    ]
    for variant, params in variants:
        result = evaluate_strategy(
            prices=evaluation_prices,
            ticker=config.base_ticker,
            params=params,
            variant=variant,
            cost_bps=10.0,
            initial_capital=config.initial_capital,
            label="leaderboard_test",
        )
        rows.append(result["row"])
    table = pd.DataFrame(rows)
    table["passes_selection"] = (
        (table["cagr"] > 0)
        & (table["max_drawdown"] >= table["benchmark_max_drawdown"] - 0.05)
        & (table["turnover"] < 8.0)
    )
    return table.sort_values(["passes_selection", "sharpe"], ascending=[False, False]).reset_index(drop=True)


def evaluate_strategy(
    prices: pd.DataFrame,
    ticker: str,
    params: SmaParameters,
    variant: str,
    cost_bps: float,
    initial_capital: float,
    label: str,
) -> dict[str, Any]:
    ticker = ticker.upper()
    if ticker not in prices.columns:
        raise ValueError(f"Missing ticker in price data: {ticker}")

    needed = [ticker]
    if variant == "fallback_spy":
        needed.append("SPY")
    if variant == "fallback_qqq":
        needed.append("QQQ")
    needed = list(dict.fromkeys(needed))
    available = [column for column in needed if column in prices.columns]

    price = prices[ticker].dropna()
    signals = SmaCrossoverStrategy(params).generate(price)
    if variant == "fallback_spy" and "SPY" in prices.columns:
        weights = build_fallback_weights(ticker, "SPY", signals.target_position)
    elif variant == "fallback_qqq" and "QQQ" in prices.columns:
        weights = build_fallback_weights(ticker, "QQQ", signals.target_position)
    else:
        weights = build_single_asset_weights(ticker, signals.target_position)

    returns = prices[available].pct_change().fillna(0.0)
    engine_result = run_weight_backtest(
        returns=returns,
        target_weights=weights,
        config=EngineConfig(initial_capital=initial_capital, cost_model=BpsCost(cost_bps)),
    )
    curve = _combine_curve(price, signals, engine_result.curve, engine_result.executed_weights, ticker, prices)
    row = summarize_curve(
        curve=curve,
        executed_weights=engine_result.executed_weights,
        ticker=ticker,
        label=label,
        variant=variant,
        params=params,
        cost_bps=cost_bps,
    )
    return {"row": row, "curve": curve, "metrics": pd.DataFrame([row])}


def evaluate_equal_weight_signal_portfolio(
    prices: pd.DataFrame,
    universe: list[str],
    params: SmaParameters,
    config: ResearchConfig,
) -> dict[str, Any]:
    weights = []
    valid_tickers = [ticker for ticker in universe if ticker in prices.columns]
    for ticker in valid_tickers:
        signals = SmaCrossoverStrategy(params).generate(prices[ticker].dropna())
        weights.append(signals.target_position.rename(ticker))
    target_weights = pd.concat(weights, axis=1).reindex(prices.index).fillna(0.0)
    if valid_tickers:
        target_weights = target_weights / len(valid_tickers)
    returns = prices[valid_tickers].pct_change().fillna(0.0)
    result = run_weight_backtest(
        returns,
        target_weights,
        EngineConfig(initial_capital=config.initial_capital, cost_model=BpsCost(10.0)),
    )

    basket_return = returns.mean(axis=1)
    benchmark_equity = config.initial_capital * (1.0 + basket_return).cumprod()
    closed = calculate_closed_trade_returns(result.curve["strategy_return"], result.executed_weights.abs().sum(axis=1))
    win_rate = calculate_win_rate(closed)
    row = summarize_performance(
        name="equal_weight_signal_portfolio",
        equity=result.curve["strategy_equity"],
        returns=result.curve["strategy_return"],
        trades=int((result.curve["turnover"] > 0).sum()),
        win_rate=win_rate,
        exposure=float((result.executed_weights.abs().sum(axis=1) > 0).mean()),
        turnover=annualized_turnover(result.curve["turnover"]),
        closed_trade_returns=closed,
        gross_equity=result.curve["gross_strategy_equity"],
        benchmark_equity=benchmark_equity,
    )
    return row | {
        "ticker": "EQUAL_WEIGHT",
        "label": "multi_asset",
        "variant": "equal_weight_signal_portfolio",
        "short_window": params.short_window,
        "long_window": params.long_window,
        "cost_bps": 10.0,
        "benchmark_cagr": summarize_performance("benchmark", benchmark_equity, basket_return)["cagr"],
        "benchmark_sharpe": summarize_performance("benchmark", benchmark_equity, basket_return)["sharpe"],
        "benchmark_max_drawdown": summarize_performance("benchmark", benchmark_equity, basket_return)["max_drawdown"],
    }


def summarize_curve(
    curve: pd.DataFrame,
    executed_weights: pd.DataFrame,
    ticker: str,
    label: str,
    variant: str,
    params: SmaParameters,
    cost_bps: float,
) -> dict[str, Any]:
    closed = calculate_closed_trade_returns(curve["strategy_return"], executed_weights.abs().sum(axis=1))
    win_rate = calculate_win_rate(closed)
    benchmark_metrics = summarize_performance("benchmark", curve["buy_hold_equity"], curve["buy_hold_return"])
    row = summarize_performance(
        name=f"{ticker}_{variant}_{params.label()}_{cost_bps:g}bps",
        equity=curve["strategy_equity"],
        returns=curve["strategy_return"],
        trades=int((curve["turnover"] > 0).sum()),
        win_rate=win_rate,
        exposure=float((executed_weights.abs().sum(axis=1) > 0).mean()),
        turnover=annualized_turnover(curve["turnover"]),
        closed_trade_returns=closed,
        gross_equity=curve["gross_strategy_equity"],
        benchmark_equity=curve["buy_hold_equity"],
    )
    return row | {
        "ticker": ticker,
        "label": label,
        "variant": variant,
        "short_window": params.short_window,
        "long_window": params.long_window,
        "spread_threshold": params.spread_threshold,
        "momentum_window": np.nan if params.momentum_window is None else params.momentum_window,
        "partial_exposure": params.partial_exposure,
        "cost_bps": cost_bps,
        "benchmark_total_return": benchmark_metrics["total_return"],
        "benchmark_cagr": benchmark_metrics["cagr"],
        "benchmark_sharpe": benchmark_metrics["sharpe"],
        "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
    }


def select_best_parameters(table: pd.DataFrame, config: ResearchConfig) -> SmaParameters:
    if table.empty:
        return SmaParameters(20, 100)
    candidates = table[
        (table["cagr"] > 0)
        & (table["max_drawdown"] >= table["benchmark_max_drawdown"] - 0.05)
        & (table["turnover"] < 8.0)
        & (table["neighbor_sharpe"].fillna(table["sharpe"]) > 0)
    ]
    source = candidates if not candidates.empty else table
    best = source.sort_values("sharpe", ascending=False).iloc[0]
    return SmaParameters(short_window=int(best["short_window"]), long_window=int(best["long_window"]))


def add_parameter_stability(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    enriched = table.copy()
    neighbor_sharpes = []
    stable_counts = []
    for _, row in enriched.iterrows():
        neighbors = enriched[
            (enriched["short_window"].sub(row["short_window"]).abs() <= 25)
            & (enriched["long_window"].sub(row["long_window"]).abs() <= 100)
            & ~(
                (enriched["short_window"] == row["short_window"])
                & (enriched["long_window"] == row["long_window"])
            )
        ]
        neighbor_sharpes.append(float(neighbors["sharpe"].mean()) if not neighbors.empty else np.nan)
        stable_counts.append(int((neighbors["sharpe"] > 0).sum()) if not neighbors.empty else 0)
    enriched["neighbor_sharpe"] = neighbor_sharpes
    enriched["stable_neighbor_count"] = stable_counts
    enriched["hot_pixel_risk"] = enriched["sharpe"] - enriched["neighbor_sharpe"]
    return enriched


def parameter_grid(config: ResearchConfig) -> list[tuple[int, int]]:
    return [
        (short_window, long_window)
        for short_window in config.short_windows
        for long_window in config.long_windows
        if short_window < long_window
    ]


def create_fixture_prices(config: ResearchConfig) -> pd.DataFrame:
    dates = pd.date_range(config.start, config.end or "2026-05-15", freq="B")
    base = np.arange(len(dates))
    prices = {}
    for idx, ticker in enumerate(config.universe):
        drift = 0.00035 + idx * 0.000025
        seasonal = 0.015 * np.sin(base / (18 + idx))
        shock = 0.01 * np.sin(base / (7 + idx))
        returns = drift + seasonal / 252 + shock / 252
        prices[ticker] = 100 * (1.0 + pd.Series(returns, index=dates)).cumprod()
    return pd.DataFrame(prices, index=dates)


def _download_prices(config: ResearchConfig) -> pd.DataFrame:
    return download_adjusted_close(config.universe, start=config.start, end=config.end or default_end_date())


def _combine_curve(
    price: pd.Series,
    signals,
    engine_curve: pd.DataFrame,
    executed_weights: pd.DataFrame,
    ticker: str,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    curve = engine_curve.copy()
    curve["price"] = price.reindex(curve.index)
    curve["short_sma"] = signals.short_sma.reindex(curve.index)
    curve["long_sma"] = signals.long_sma.reindex(curve.index)
    curve["signal"] = signals.target_position.reindex(curve.index).fillna(0.0)
    curve["position"] = executed_weights.get(ticker, pd.Series(0.0, index=curve.index))
    curve["buy_hold_return"] = prices[ticker].pct_change().reindex(curve.index).fillna(0.0)
    curve["buy_hold_equity"] = engine_curve["strategy_equity"].iloc[0] * (1.0 + curve["buy_hold_return"]).cumprod()
    curve["buy_hold_drawdown"] = curve["buy_hold_equity"] / curve["buy_hold_equity"].cummax() - 1.0
    return curve
