from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
EQUITY_CSV = OUTPUT_DIR / "equity_curve.csv"
METRICS_CSV = OUTPUT_DIR / "metrics.csv"
PNG_OUT = OUTPUT_DIR / "aapl_model_forecast_visual.png"
EXCEL_READY_CSV = OUTPUT_DIR / "excel_model_data.csv"


def main() -> None:
    curve = pd.read_csv(EQUITY_CSV, parse_dates=["Date"])
    metrics = pd.read_csv(METRICS_CSV)

    curve = _add_forecast_columns(curve)
    curve.to_csv(EXCEL_READY_CSV, index=False)
    _save_forecast_plot(curve, metrics)

    print(f"Wrote visual PNG: {PNG_OUT}")
    print(f"Wrote Excel-ready CSV: {EXCEL_READY_CSV}")


def _add_forecast_columns(curve: pd.DataFrame) -> pd.DataFrame:
    curve = curve.copy()
    curve["model_forecast"] = curve["signal"].map({1.0: "LONG", 0.0: "CASH"}).fillna("CASH")
    curve["entry_price"] = curve["price"].where(curve["position"].diff().fillna(curve["position"]) > 0)
    curve["exit_price"] = curve["price"].where(curve["position"].diff().fillna(curve["position"]) < 0)
    curve["long_forecast_price"] = curve["price"].where(curve["signal"] == 1.0)
    curve["cash_forecast_price"] = curve["price"].where(curve["signal"] == 0.0)
    return curve


def _save_forecast_plot(curve: pd.DataFrame, metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 0.7, 1.4]},
    )

    price_axis, forecast_axis, equity_axis = axes
    price_axis.plot(curve["Date"], curve["price"], color="#1F2937", linewidth=1.6, label="AAPL adjusted close")
    price_axis.plot(curve["Date"], curve["short_sma"], color="#2563EB", linewidth=1.1, label="SMA 20")
    price_axis.plot(curve["Date"], curve["long_sma"], color="#B45309", linewidth=1.1, label="SMA 100")
    price_axis.scatter(curve["Date"], curve["entry_price"], marker="^", color="#15803D", s=42, label="model entry")
    price_axis.scatter(curve["Date"], curve["exit_price"], marker="v", color="#B91C1C", s=42, label="model exit")
    _shade_forecast_regions(price_axis, curve)
    price_axis.set_title("AAPL price history with SMA model forecasts")
    price_axis.set_ylabel("Adjusted price")
    price_axis.grid(alpha=0.22)
    price_axis.legend(loc="upper left", ncols=3)

    forecast_axis.fill_between(
        curve["Date"],
        curve["signal"],
        0,
        step="pre",
        color="#16A34A",
        alpha=0.35,
        label="model forecast: LONG",
    )
    forecast_axis.plot(curve["Date"], curve["signal"], color="#166534", linewidth=0.9)
    forecast_axis.set_yticks([0, 1], labels=["CASH", "LONG"])
    forecast_axis.set_ylim(-0.1, 1.1)
    forecast_axis.set_ylabel("Forecast")
    forecast_axis.grid(alpha=0.22)
    forecast_axis.legend(loc="upper left")

    equity_axis.plot(curve["Date"], curve["strategy_equity"], color="#2563EB", label="strategy equity")
    equity_axis.plot(curve["Date"], curve["buy_hold_equity"], color="#6B7280", label="buy and hold")
    equity_axis.set_ylabel("Equity, $")
    equity_axis.grid(alpha=0.22)
    equity_axis.legend(loc="upper left")

    _add_metric_box(price_axis, metrics)
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=170)
    plt.close(fig)


def _shade_forecast_regions(axis: plt.Axes, curve: pd.DataFrame) -> None:
    signal = curve["signal"].fillna(0).astype(int)
    start_idx = 0

    for idx in range(1, len(curve)):
        if signal.iloc[idx] != signal.iloc[start_idx]:
            _shade_region(axis, curve, start_idx, idx - 1, signal.iloc[start_idx])
            start_idx = idx

    _shade_region(axis, curve, start_idx, len(curve) - 1, signal.iloc[start_idx])


def _shade_region(axis: plt.Axes, curve: pd.DataFrame, start_idx: int, end_idx: int, value: int) -> None:
    if end_idx <= start_idx:
        return

    color = "#DCFCE7" if value == 1 else "#F3F4F6"
    axis.axvspan(curve["Date"].iloc[start_idx], curve["Date"].iloc[end_idx], color=color, alpha=0.45, linewidth=0)


def _add_metric_box(axis: plt.Axes, metrics: pd.DataFrame) -> None:
    metric_map = metrics.set_index("name")
    strategy = metric_map.loc["strategy"]
    buy_hold = metric_map.loc["buy_hold"]
    text = (
        "Backtest summary\n"
        f"Strategy CAGR: {strategy['cagr']:.2%}\n"
        f"Buy & hold CAGR: {buy_hold['cagr']:.2%}\n"
        f"Strategy Sharpe: {strategy['sharpe']:.2f}\n"
        f"Trades: {int(strategy['trades'])}"
    )
    axis.text(
        0.985,
        0.03,
        text,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.92},
    )


if __name__ == "__main__":
    main()
