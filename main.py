from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from quant_backtest.backtest import BacktestConfig, run_sma_backtest
from quant_backtest.data import default_end_date, download_ohlcv
from quant_backtest.plotting import save_backtest_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple SMA crossover backtest.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, default: AAPL.")
    parser.add_argument("--start", default="2015-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default=None, help="Exclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--short-window", type=int, default=20, help="Short SMA window.")
    parser.add_argument("--long-window", type=int, default=100, help="Long SMA window.")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Cost per position change in basis points.")
    parser.add_argument("--initial-capital", type=float, default=10_000.0, help="Initial capital.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV and PNG outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.strip().upper()
    end = args.end or default_end_date()
    output_dir = Path(args.output_dir)

    config = BacktestConfig(
        ticker=ticker,
        short_window=args.short_window,
        long_window=args.long_window,
        cost_bps=args.cost_bps,
        initial_capital=args.initial_capital,
    )

    prices = download_ohlcv(ticker=ticker, start=args.start, end=end)
    result = run_sma_backtest(prices=prices, config=config)

    output_dir.mkdir(parents=True, exist_ok=True)
    equity_path = output_dir / "equity_curve.csv"
    metrics_path = output_dir / "metrics.csv"
    plot_path = output_dir / f"{ticker.lower()}_sma_backtest.png"

    result.equity_curve.to_csv(equity_path, index_label="Date")
    result.metrics.to_csv(metrics_path)
    save_backtest_plot(
        equity_curve=result.equity_curve,
        ticker=ticker,
        short_window=args.short_window,
        long_window=args.long_window,
        output_path=plot_path,
    )

    print(_format_metrics_for_console(result.metrics))
    print()
    print(f"Wrote equity curve: {equity_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote plot: {plot_path}")


def _format_metrics_for_console(metrics):
    display = metrics.copy()
    percent_columns = ["total_return", "cagr", "ann_volatility", "max_drawdown", "win_rate"]
    for column in percent_columns:
        display[column] = display[column].map(lambda value: "" if value != value else f"{value * 100:.2f}%")

    display["sharpe"] = display["sharpe"].map(lambda value: "" if value != value else f"{value:.2f}")
    display["trades"] = display["trades"].map(lambda value: "" if value != value else f"{int(value)}")
    return display.to_string()


if __name__ == "__main__":
    main()
