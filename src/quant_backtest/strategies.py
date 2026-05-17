from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SmaParameters:
    short_window: int
    long_window: int
    spread_threshold: float = 0.0
    momentum_window: int | None = None
    partial_exposure: bool = False

    def label(self) -> str:
        parts = [f"sma_{self.short_window}_{self.long_window}"]
        if self.spread_threshold:
            parts.append(f"thr_{self.spread_threshold:.3f}")
        if self.momentum_window:
            parts.append(f"mom_{self.momentum_window}")
        if self.partial_exposure:
            parts.append("partial")
        return "_".join(parts)


@dataclass(frozen=True)
class SmaSignalFrame:
    target_position: pd.Series
    short_sma: pd.Series
    long_sma: pd.Series
    spread: pd.Series
    momentum: pd.Series | None


@dataclass(frozen=True)
class TrendAllocationParameters:
    short_window: int
    long_window: int
    entry_threshold: float = 0.0
    exit_threshold: float = 0.0
    min_hold_days: int = 0
    cooldown_days: int = 0

    def label(self) -> str:
        return (
            f"trend_{self.short_window}_{self.long_window}"
            f"_entry_{self.entry_threshold:.3f}"
            f"_exit_{self.exit_threshold:.3f}"
            f"_hold_{self.min_hold_days}"
            f"_cool_{self.cooldown_days}"
        )


class SmaCrossoverStrategy:
    name = "sma_crossover"

    def __init__(self, params: SmaParameters) -> None:
        _validate_params(params)
        self.params = params

    def generate(self, price: pd.Series) -> SmaSignalFrame:
        clean = price.dropna().astype(float)
        short_sma = clean.rolling(self.params.short_window, min_periods=self.params.short_window).mean()
        long_sma = clean.rolling(self.params.long_window, min_periods=self.params.long_window).mean()
        spread = short_sma / long_sma - 1.0

        strong_trend = spread > self.params.spread_threshold
        valid = long_sma.notna()
        if self.params.momentum_window:
            momentum = clean.pct_change(self.params.momentum_window)
            strong_trend = strong_trend & (momentum > 0)
            valid = valid & momentum.notna()
        else:
            momentum = None

        if self.params.partial_exposure:
            weak_trend = clean > long_sma
            target_position = pd.Series(0.0, index=clean.index, name="target_position")
            target_position = target_position.where(~(weak_trend & valid), 0.5)
            target_position = target_position.where(~(strong_trend & valid), 1.0)
        else:
            target_position = (strong_trend & valid).astype(float)
            target_position.name = "target_position"

        target_position = target_position.where(valid, 0.0)
        return SmaSignalFrame(
            target_position=target_position,
            short_sma=short_sma,
            long_sma=long_sma,
            spread=spread,
            momentum=momentum,
        )


class TrendAllocationStrategy:
    name = "trend_allocation"

    def __init__(self, params: TrendAllocationParameters) -> None:
        _validate_trend_params(params)
        self.params = params

    def generate(self, price: pd.Series) -> SmaSignalFrame:
        clean = price.dropna().astype(float)
        short_sma = clean.rolling(self.params.short_window, min_periods=self.params.short_window).mean()
        long_sma = clean.rolling(self.params.long_window, min_periods=self.params.long_window).mean()
        spread = short_sma / long_sma - 1.0
        valid = long_sma.notna()

        position = 0.0
        hold_days = 0
        cooldown_days = 0
        values: list[float] = []

        for date in clean.index:
            if not bool(valid.loc[date]):
                position = 0.0
                hold_days = 0
                cooldown_days = 0
                values.append(0.0)
                continue

            current_spread = float(spread.loc[date])
            if position == 0.0:
                if cooldown_days > 0:
                    cooldown_days -= 1
                elif current_spread > self.params.entry_threshold:
                    position = 1.0
                    hold_days = 1
            else:
                if hold_days < self.params.min_hold_days:
                    hold_days += 1
                elif current_spread < self.params.exit_threshold:
                    position = 0.0
                    hold_days = 0
                    cooldown_days = self.params.cooldown_days
                else:
                    hold_days += 1

            values.append(position)

        target_position = pd.Series(values, index=clean.index, name="target_position")
        return SmaSignalFrame(
            target_position=target_position,
            short_sma=short_sma,
            long_sma=long_sma,
            spread=spread,
            momentum=None,
        )


def build_single_asset_weights(ticker: str, target_position: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({ticker: target_position.astype(float)})


def build_fallback_weights(
    target_ticker: str,
    fallback_ticker: str,
    target_position: pd.Series,
) -> pd.DataFrame:
    position = target_position.astype(float).clip(lower=0.0, upper=1.0)
    if target_ticker == fallback_ticker:
        return pd.DataFrame({target_ticker: position})
    return pd.DataFrame(
        {
            target_ticker: position,
            fallback_ticker: 1.0 - position,
        }
    )


def build_regime_fallback_weights(
    target_ticker: str,
    fallback_ticker: str,
    target_position: pd.Series,
    fallback_regime: pd.Series,
) -> pd.DataFrame:
    target = target_position.astype(float).clip(lower=0.0, upper=1.0)
    regime = fallback_regime.reindex(target.index).fillna(False).astype(bool)
    fallback = (1.0 - target).where(regime, 0.0)
    if target_ticker == fallback_ticker:
        return pd.DataFrame({target_ticker: target})
    return pd.DataFrame({target_ticker: target, fallback_ticker: fallback})


def build_hybrid_regime_weights(
    target_ticker: str,
    fallback_ticker: str,
    signals: SmaSignalFrame,
    params: TrendAllocationParameters,
    fallback_regime: pd.Series,
) -> pd.DataFrame:
    target = signals.target_position.astype(float).clip(lower=0.0, upper=1.0)
    regime = fallback_regime.reindex(target.index).fillna(False).astype(bool)
    weak_trend = (
        (target == 0.0)
        & signals.long_sma.reindex(target.index).notna()
        & (signals.spread.reindex(target.index) > params.exit_threshold)
        & (signals.spread.reindex(target.index) <= params.entry_threshold)
        & regime
    )
    target_weight = target.where(~weak_trend, 0.5)
    fallback_weight = pd.Series(0.0, index=target.index, name=fallback_ticker)
    fallback_weight = fallback_weight.where(~((target == 0.0) & regime), 1.0)
    fallback_weight = fallback_weight.where(~weak_trend, 0.5)
    if target_ticker == fallback_ticker:
        return pd.DataFrame({target_ticker: target_weight})
    return pd.DataFrame({target_ticker: target_weight, fallback_ticker: fallback_weight})


def build_sma_regime(price: pd.Series, short_window: int = 50, long_window: int = 200) -> pd.Series:
    clean = price.dropna().astype(float)
    short_sma = clean.rolling(short_window, min_periods=short_window).mean()
    long_sma = clean.rolling(long_window, min_periods=long_window).mean()
    return (short_sma > long_sma).rename("regime")


def _validate_params(params: SmaParameters) -> None:
    if params.short_window <= 0 or params.long_window <= 0:
        raise ValueError("SMA windows must be positive.")
    if params.short_window >= params.long_window:
        raise ValueError("short_window must be smaller than long_window.")
    if params.spread_threshold < 0:
        raise ValueError("spread_threshold must be non-negative.")
    if params.momentum_window is not None and params.momentum_window <= 0:
        raise ValueError("momentum_window must be positive when provided.")


def _validate_trend_params(params: TrendAllocationParameters) -> None:
    if params.short_window <= 0 or params.long_window <= 0:
        raise ValueError("SMA windows must be positive.")
    if params.short_window >= params.long_window:
        raise ValueError("short_window must be smaller than long_window.")
    if params.entry_threshold < params.exit_threshold:
        raise ValueError("entry_threshold must be greater than or equal to exit_threshold.")
    if params.min_hold_days < 0 or params.cooldown_days < 0:
        raise ValueError("min_hold_days and cooldown_days must be non-negative.")
