"""Data models for the BIRT scoring module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class GaussianPrior:
    """Gaussian prior for agent capability."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")


@dataclass
class PosteriorEstimate:
    """Result of Bayesian IRT scoring for a single agent."""

    theta: float
    sigma: float
    confidence: float
    n_probes: int
    testability_tier: Literal["FULLY_PROBED", "PARTIALLY_PROBED", "UNTESTABLE"]
    prior_influence: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
