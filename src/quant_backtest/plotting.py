from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def save_backtest_plot(
    equity_curve: pd.DataFrame,
    ticker: str,
    short_window: int,
    long_window: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.3, 1.0]},
    )

    _plot_price_panel(axes[0], equity_curve, ticker, short_window, long_window)
    _plot_equity_panel(axes[1], equity_curve)
    _plot_drawdown_panel(axes[2], equity_curve)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_price_panel(
    axis: plt.Axes,
    equity_curve: pd.DataFrame,
    ticker: str,
    short_window: int,
    long_window: int,
) -> None:
    axis.plot(equity_curve.index, equity_curve["price"], label=f"{ticker} adjusted close")
    axis.plot(equity_curve.index, equity_curve["short_sma"], label=f"SMA {short_window}")
    axis.plot(equity_curve.index, equity_curve["long_sma"], label=f"SMA {long_window}")

    position_change = equity_curve["position"].diff().fillna(equity_curve["position"])
    entries = equity_curve[position_change > 0]
    exits = equity_curve[position_change < 0]
    axis.scatter(entries.index, entries["price"], marker="^", color="green", label="entry", zorder=4)
    axis.scatter(exits.index, exits["price"], marker="v", color="red", label="exit", zorder=4)

    axis.set_title(f"{ticker} SMA crossover backtest")
    axis.set_ylabel("Price")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")


def _plot_equity_panel(axis: plt.Axes, equity_curve: pd.DataFrame) -> None:
    axis.plot(equity_curve.index, equity_curve["strategy_equity"], label="strategy")
    axis.plot(equity_curve.index, equity_curve["buy_hold_equity"], label="buy and hold")
    axis.set_ylabel("Equity")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")


def _plot_drawdown_panel(axis: plt.Axes, equity_curve: pd.DataFrame) -> None:
    axis.fill_between(
        equity_curve.index,
        equity_curve["strategy_drawdown"],
        0,
        alpha=0.35,
        label="strategy drawdown",
    )
    axis.plot(
        equity_curve.index,
        equity_curve["buy_hold_drawdown"],
        linewidth=1,
        label="buy and hold drawdown",
    )
    axis.set_ylabel("Drawdown")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
