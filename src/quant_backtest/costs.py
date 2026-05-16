from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class CostModel:
    """Interface for portfolio transaction-cost models."""

    def calculate(self, turnover: pd.Series) -> pd.Series:
        raise NotImplementedError


@dataclass(frozen=True)
class BpsCost(CostModel):
    bps: float = 10.0

    def calculate(self, turnover: pd.Series) -> pd.Series:
        if self.bps < 0:
            raise ValueError("bps must be non-negative.")
        return turnover.astype(float) * (self.bps / 10_000.0)


@dataclass(frozen=True)
class CostScenario:
    name: str
    model: CostModel


def bps_scenarios(values: list[float]) -> list[CostScenario]:
    return [CostScenario(name=f"{value:g}bps", model=BpsCost(value)) for value in values]
