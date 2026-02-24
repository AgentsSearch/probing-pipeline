"""Bayesian IRT update — 2PL model for agent capability estimation.

Model:
  P(correct | theta, d, a) = sigmoid(a * (theta - d))

  posterior_precision = (1/prior_sigma^2) + a^2 * p * (1-p)
  posterior_mu = (prior_mu/prior_sigma^2 + a*(x-p)) / posterior_precision
  posterior_sigma = 1 / sqrt(posterior_precision)

Where:
  theta = agent capability (estimated)
  d     = probe difficulty
  a     = probe discrimination
  x     = observed score (0-1)
  p     = predicted probability of success
"""

from __future__ import annotations

import logging
import math

from src.models.probe import Probe
from src.models.scoring import GaussianPrior, PosteriorEstimate

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def bayesian_update(
    prior: GaussianPrior,
    probe: Probe,
    observed_score: float,
) -> GaussianPrior:
    """Perform a single Bayesian IRT update given a probe result.

    Args:
        prior: Current Gaussian belief about agent capability.
        probe: The probe that was executed (provides difficulty, discrimination).
        observed_score: The score achieved (0 for fail, 1 for pass, fractional for partial).

    Returns:
        Updated GaussianPrior reflecting the new evidence.
    """
    d = probe.estimated_difficulty
    a = probe.discrimination
    theta = prior.mu

    # Predicted probability of success
    p = _sigmoid(a * (theta - d))
    # Clamp to avoid numerical issues
    p = max(1e-6, min(1 - 1e-6, p))

    # Bayesian update (Laplace approximation to 2PL posterior)
    prior_precision = 1.0 / (prior.sigma ** 2)
    info_gain = (a ** 2) * p * (1 - p)
    posterior_precision = prior_precision + info_gain

    posterior_mu = (prior.mu * prior_precision + a * (observed_score - p)) / posterior_precision
    posterior_sigma = 1.0 / math.sqrt(posterior_precision)

    # Clamp mu to [0, 1]
    posterior_mu = max(0.0, min(1.0, posterior_mu))

    logger.debug(
        "BIRT update: prior=(%.3f, %.3f) + probe(d=%.2f, a=%.2f, x=%.2f) "
        "-> posterior=(%.3f, %.3f)",
        prior.mu, prior.sigma, d, a, observed_score,
        posterior_mu, posterior_sigma,
    )

    return GaussianPrior(mu=posterior_mu, sigma=posterior_sigma)


def score_agent(
    prior: GaussianPrior,
    probes: list[Probe],
    observed_scores: list[float],
) -> PosteriorEstimate:
    """Score an agent by sequentially updating the prior with probe results.

    Args:
        prior: Initial Gaussian prior from metadata signals.
        probes: List of executed probes.
        observed_scores: Corresponding scores for each probe (same length as probes).

    Returns:
        PosteriorEstimate with final capability score and confidence.

    Raises:
        ValueError: If probes and scores have different lengths.
    """
    if len(probes) != len(observed_scores):
        raise ValueError(
            f"Mismatched lengths: {len(probes)} probes vs {len(observed_scores)} scores"
        )

    initial_sigma = prior.sigma
    current = prior

    for probe, score in zip(probes, observed_scores):
        current = bayesian_update(current, probe, score)

    n_probes = len(probes)

    # Determine testability tier
    if n_probes == 0:
        tier = "UNTESTABLE"
    elif n_probes >= 2:
        tier = "FULLY_PROBED"
    else:
        tier = "PARTIALLY_PROBED"

    # Prior influence: how much the posterior is still driven by the prior
    # Measured as ratio of posterior variance explained by prior
    prior_influence = (current.sigma ** 2) / (initial_sigma ** 2) if initial_sigma > 0 else 0.0

    confidence = 1.0 - current.sigma

    estimate = PosteriorEstimate(
        theta=current.mu,
        sigma=current.sigma,
        confidence=max(0.0, confidence),
        n_probes=n_probes,
        testability_tier=tier,
        prior_influence=prior_influence,
    )

    logger.info(
        "Agent scored: theta=%.3f, sigma=%.3f, confidence=%.3f, tier=%s, n_probes=%d",
        estimate.theta, estimate.sigma, estimate.confidence,
        estimate.testability_tier, n_probes,
    )

    return estimate
