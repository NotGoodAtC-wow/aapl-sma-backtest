from __future__ import annotations

import math

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365.25


def total_return(equity: pd.Series) -> float:
    clean = equity.dropna()
    if clean.empty:
        return math.nan
    return float(clean.iloc[-1] / clean.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    clean = equity.dropna()
    if len(clean) < 2:
        return math.nan

    years = (clean.index[-1] - clean.index[0]).days / CALENDAR_DAYS_PER_YEAR
    if years <= 0:
        return math.nan

    ending_ratio = clean.iloc[-1] / clean.iloc[0]
    if ending_ratio <= 0:
        return math.nan
    return float(ending_ratio ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return math.nan
    return float(clean.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    clean = returns.dropna()
    if clean.empty:
        return math.nan

    volatility = annualized_volatility(clean)
    if volatility == 0 or math.isnan(volatility):
        return math.nan

    annualized_return = clean.mean() * TRADING_DAYS_PER_YEAR
    return float((annualized_return - risk_free_rate) / volatility)


def drawdown_series(equity: pd.Series) -> pd.Series:
    clean = equity.astype(float)
    running_max = clean.cummax()
    return clean / running_max - 1.0


def max_drawdown(equity: pd.Series) -> float:
    drawdowns = drawdown_series(equity).dropna()
    if drawdowns.empty:
        return math.nan
    return float(drawdowns.min())


def summarize_performance(
    name: str,
    equity: pd.Series,
    returns: pd.Series,
    trades: int | None = None,
    win_rate: float | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int | str]:
    return {
        "name": name,
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "ann_volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free_rate=risk_free_rate),
        "max_drawdown": max_drawdown(equity),
        "trades": math.nan if trades is None else int(trades),
        "win_rate": math.nan if win_rate is None else float(win_rate),
    }


def metrics_table(rows: list[dict[str, float | int | str]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    return table.set_index("name")
