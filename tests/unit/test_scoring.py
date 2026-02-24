"""Unit tests for the BIRT scoring module."""

import pytest

from src.models.probe import Probe, RubricDimension
from src.models.scoring import GaussianPrior
from src.models.integration import ProbeExecutionResult, ActionStep
from src.scoring.prior import construct_prior, _normalise_elo, _normalise_community_rating
from src.scoring.birt import bayesian_update, score_agent, _sigmoid
from src.scoring.confidence import assess_confidence


def _make_rubric() -> list[RubricDimension]:
    return [
        RubricDimension(name="correctness", weight=0.6, criteria="c", pass_threshold="p"),
        RubricDimension(name="completeness", weight=0.4, criteria="c", pass_threshold="p"),
    ]


def _make_probe(**overrides) -> Probe:
    defaults = dict(
        probe_id="P1", targets_subtask="S1", tool="test_tool",
        arguments={"x": 1}, estimated_difficulty=0.5,
        discrimination=1.0, rubric=_make_rubric(),
        timeout_seconds=15, priority="PRIMARY",
    )
    defaults.update(overrides)
    return Probe(**defaults)


# --- Prior ---


class TestPrior:
    def test_all_signals_tight_sigma(self):
        prior = construct_prior(
            retrieval_score=0.7, coverage_score=0.8,
            arena_elo=1500, community_rating=4.0,
        )
        assert prior.sigma == 0.3
        assert 0.0 < prior.mu < 1.0

    def test_two_signals_diffuse_sigma(self):
        prior = construct_prior(retrieval_score=0.6, coverage_score=0.5)
        assert prior.sigma == 0.5

    def test_three_signals_tight_sigma(self):
        prior = construct_prior(
            retrieval_score=0.7, coverage_score=0.8, arena_elo=1200,
        )
        assert prior.sigma == 0.3

    def test_mu_clamped(self):
        # Very low scores
        prior = construct_prior(retrieval_score=0.0, coverage_score=0.0)
        assert prior.mu >= 0.05
        # Very high scores
        prior = construct_prior(
            retrieval_score=1.0, coverage_score=1.0,
            arena_elo=2000, community_rating=5.0,
        )
        assert prior.mu <= 0.95

    def test_normalise_elo(self):
        assert _normalise_elo(800) == 0.0
        assert _normalise_elo(2000) == 1.0
        assert 0.0 < _normalise_elo(1400) < 1.0

    def test_normalise_community_rating(self):
        assert _normalise_community_rating(0.0) == 0.0
        assert _normalise_community_rating(5.0) == 1.0
        assert _normalise_community_rating(2.5) == 0.5


# --- BIRT ---


class TestBIRT:
    def test_sigmoid_bounds(self):
        assert 0.0 < _sigmoid(-100) < 0.01
        assert _sigmoid(100) >= 0.99
        assert abs(_sigmoid(0) - 0.5) < 1e-6

    def test_pass_increases_mu(self):
        prior = GaussianPrior(mu=0.5, sigma=0.3)
        probe = _make_probe(estimated_difficulty=0.5, discrimination=1.0)
        posterior = bayesian_update(prior, probe, observed_score=1.0)
        assert posterior.mu > prior.mu

    def test_fail_decreases_mu(self):
        prior = GaussianPrior(mu=0.5, sigma=0.3)
        probe = _make_probe(estimated_difficulty=0.5, discrimination=1.0)
        posterior = bayesian_update(prior, probe, observed_score=0.0)
        assert posterior.mu < prior.mu

    def test_update_reduces_uncertainty(self):
        prior = GaussianPrior(mu=0.5, sigma=0.3)
        probe = _make_probe()
        posterior = bayesian_update(prior, probe, observed_score=0.8)
        assert posterior.sigma < prior.sigma

    def test_high_discrimination_stronger_update(self):
        prior = GaussianPrior(mu=0.5, sigma=0.3)
        low_disc = _make_probe(discrimination=0.5)
        high_disc = _make_probe(discrimination=2.0)
        post_low = bayesian_update(prior, low_disc, observed_score=1.0)
        post_high = bayesian_update(prior, high_disc, observed_score=1.0)
        # Higher discrimination should move mu more
        assert abs(post_high.mu - prior.mu) > abs(post_low.mu - prior.mu)

    def test_mu_stays_in_bounds(self):
        prior = GaussianPrior(mu=0.99, sigma=0.1)
        probe = _make_probe(discrimination=2.0)
        posterior = bayesian_update(prior, probe, observed_score=1.0)
        assert 0.0 <= posterior.mu <= 1.0

    def test_score_agent_fully_probed(self):
        prior = GaussianPrior(mu=0.5, sigma=0.3)
        probes = [_make_probe(probe_id="P1"), _make_probe(probe_id="P2")]
        scores = [1.0, 0.8]
        estimate = score_agent(prior, probes, scores)
        assert estimate.testability_tier == "FULLY_PROBED"
        assert estimate.n_probes == 2
        assert estimate.theta > prior.mu

    def test_score_agent_partially_probed(self):
        prior = GaussianPrior(mu=0.5, sigma=0.3)
        estimate = score_agent(prior, [_make_probe()], [1.0])
        assert estimate.testability_tier == "PARTIALLY_PROBED"

    def test_score_agent_untestable(self):
        prior = GaussianPrior(mu=0.5, sigma=0.3)
        estimate = score_agent(prior, [], [])
        assert estimate.testability_tier == "UNTESTABLE"
        assert estimate.theta == prior.mu

    def test_mismatched_lengths_raises(self):
        prior = GaussianPrior(mu=0.5, sigma=0.3)
        with pytest.raises(ValueError, match="Mismatched lengths"):
            score_agent(prior, [_make_probe()], [1.0, 0.5])


# --- Confidence ---


class TestConfidence:
    def _make_result(self, **overrides) -> ProbeExecutionResult:
        defaults = dict(
            agent_id="agent1", probe_id="P1",
            output={"data": "hello"},
            trajectory=[ActionStep(action="call_tool", tool_name="test_tool", result="ok")],
            latency_ms=100, success=True, error_info=None,
        )
        defaults.update(overrides)
        return ProbeExecutionResult(**defaults)

    def test_successful_execution_high_confidence(self):
        result = self._make_result()
        conf = assess_confidence(result)
        assert conf.sufficient is True
        assert conf.score >= 0.5

    def test_timeout_low_confidence(self):
        result = self._make_result(
            output=None, trajectory=[], success=False,
            error_info="timeout exceeded",
        )
        conf = assess_confidence(result)
        assert conf.sufficient is False
        assert conf.reason is not None

    def test_empty_output_reduces_confidence(self):
        result = self._make_result(output=None)
        conf = assess_confidence(result)
        assert conf.checks["nontrivial_content"] is False

    def test_schema_error_detected(self):
        result = self._make_result(
            trajectory=[
                ActionStep(action="call_tool", tool_name="t", error="Invalid schema for parameter x")
            ],
        )
        conf = assess_confidence(result)
        assert conf.checks["schema_valid"] is False

    def test_all_checks_pass(self):
        result = self._make_result()
        conf = assess_confidence(result)
        assert all(conf.checks.values())
