"""Prior construction — combine metadata signals into a Gaussian prior.

Signals and weights:
  Arena ELO:              0.35 (if available)
  Retrieval similarity:   0.22 (always)
  Tool-task coverage:     0.18 (always)
  Community rating:       0.13 (if available)
  Documentation quality:  0.12 (if available)

Prior sigma = 0.3 (tight, 3+ signals) or 0.5 (diffuse, fewer signals).
"""

from __future__ import annotations

import logging
from typing import Any

import yaml
from pathlib import Path

from src.models.scoring import GaussianPrior

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml"


def _load_scoring_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load scoring weights and sigma values from config."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)["scoring"]


def _normalise_elo(elo: float, elo_min: float = 800, elo_max: float = 2000) -> float:
    """Normalise an ELO rating to [0, 1]."""
    return max(0.0, min(1.0, (elo - elo_min) / (elo_max - elo_min)))


def _normalise_community_rating(rating: float, max_rating: float = 5.0) -> float:
    """Normalise a community rating (e.g. 0-5 stars) to [0, 1]."""
    return max(0.0, min(1.0, rating / max_rating))


def construct_prior(
    retrieval_score: float,
    coverage_score: float,
    arena_elo: float | None = None,
    community_rating: float | None = None,
    documentation_quality: float | None = None,
    config_path: str | Path | None = None,
) -> GaussianPrior:
    """Construct a Gaussian prior for agent capability from metadata signals.

    Args:
        retrieval_score: Similarity score from Stream B retrieval (0-1).
        coverage_score: Tool-task coverage from Stage 2 (0-1).
        arena_elo: Agent Arena ELO rating, if available.
        community_rating: Community rating (e.g. 0-5 stars), if available.
        documentation_quality: Documentation quality score (0-1), if available.
        config_path: Path to config YAML. Defaults to config/default.yaml.

    Returns:
        GaussianPrior with weighted mean and appropriate sigma.
    """
    cfg = _load_scoring_config(config_path)
    weights = cfg["weights"]

    signals: list[tuple[float, float]] = []  # (value, weight) pairs
    signal_count = 0

    # Always-available signals
    signals.append((retrieval_score, weights["retrieval_similarity"]))
    signal_count += 1

    signals.append((coverage_score, weights["coverage"]))
    signal_count += 1

    # Optional signals
    if arena_elo is not None:
        normalised_elo = _normalise_elo(arena_elo)
        signals.append((normalised_elo, weights["arena_elo"]))
        signal_count += 1

    if community_rating is not None:
        normalised_rating = _normalise_community_rating(community_rating)
        signals.append((normalised_rating, weights["community_rating"]))
        signal_count += 1

    if documentation_quality is not None:
        clamped_dq = max(0.0, min(1.0, documentation_quality))
        signals.append((clamped_dq, weights["documentation_quality"]))
        signal_count += 1

    # Compute weighted mean, renormalising weights to sum to 1
    total_weight = sum(w for _, w in signals)
    if total_weight > 0:
        mu = sum(v * w for v, w in signals) / total_weight
    else:
        mu = 0.5

    # Clamp to [0.05, 0.95] to avoid degenerate priors
    mu = max(0.05, min(0.95, mu))

    # Sigma depends on how many signals we have
    sigma = cfg["prior_sigma_tight"] if signal_count >= 3 else cfg["prior_sigma_diffuse"]

    prior = GaussianPrior(mu=mu, sigma=sigma)

    logger.info(
        "Constructed prior: mu=%.3f, sigma=%.3f (%d signals)",
        mu, sigma, signal_count,
    )

    return prior
