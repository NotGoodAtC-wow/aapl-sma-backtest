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


def _validate_params(params: SmaParameters) -> None:
    if params.short_window <= 0 or params.long_window <= 0:
        raise ValueError("SMA windows must be positive.")
    if params.short_window >= params.long_window:
        raise ValueError("short_window must be smaller than long_window.")
    if params.spread_threshold < 0:
        raise ValueError("spread_threshold must be non-negative.")
    if params.momentum_window is not None and params.momentum_window <= 0:
        raise ValueError("momentum_window must be positive when provided.")
