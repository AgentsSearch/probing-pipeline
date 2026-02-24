"""End-to-end pipeline orchestrator — runs all stages for a set of candidate agents.

Flow:
  1. Task Analysis:    query -> TaskDAG
  2. Tool Alignment:   TaskDAG + agents -> AlignmentMap per agent
  3. Probe Generation: AlignmentMap -> ProbePlan per agent
  4. Probe Validation: ProbePlan -> validated ProbePlan
  5. (Execution:       handled externally by Stream C)
  6. Scoring:          prior + probe results -> PosteriorEstimate -> RankedAgent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.llm.client import LLMClient
from src.models.alignment import AlignmentMap
from src.models.integration import (
    CandidateAgent,
    ProbeExecutionResult,
    RankedAgent,
    RetrievalResult,
)
from src.models.probe import ProbePlan
from src.models.scoring import PosteriorEstimate
from src.models.task import TaskDAG
from src.scoring.birt import score_agent
from src.scoring.confidence import assess_confidence
from src.scoring.prior import construct_prior
from src.stages.probe_generation import generate_probe_plan
from src.stages.probe_validation import validate_plan
from src.stages.task_analysis import analyse_task
from src.stages.tool_alignment import align_tools_for_agent
from src.templates.library import TemplateLibrary
from src.tool_index.retriever import ToolRetriever

logger = logging.getLogger(__name__)


# Structured error codes
TASK_ANALYSIS_FAILED = "TASK_ANALYSIS_FAILED"
ALIGNMENT_FAILED = "ALIGNMENT_FAILED"
PROBE_GENERATION_FAILED = "PROBE_GENERATION_FAILED"
PROBE_VALIDATION_FAILED = "PROBE_VALIDATION_FAILED"
EXECUTION_FAILED = "EXECUTION_FAILED"
SCORING_FAILED = "SCORING_FAILED"


@dataclass
class AgentPipelineResult:
    """Intermediate result for a single agent through the pipeline."""

    agent: CandidateAgent
    alignment: AlignmentMap | None = None
    probe_plan: ProbePlan | None = None
    validated_plan: ProbePlan | None = None
    execution_results: list[ProbeExecutionResult] = field(default_factory=list)
    estimate: PosteriorEstimate | None = None
    error_code: str | None = None
    error_detail: str | None = None


@dataclass
class PipelineConfig:
    """Configuration for the pipeline run."""

    probe_budget: int = 2
    total_timeout: int = 30
    min_interaction_confidence: float = 0.5
    difficulty_tolerance: float = 0.3
    min_rubric_dimensions: int = 2


class ProbePipeline:
    """Orchestrates the full probing pipeline.

    Args:
        llm: Initialised LLMClient.
        retriever: Initialised ToolRetriever with loaded FAISS index.
        template_library: Optional TemplateLibrary for probe template cache.
        config: Pipeline configuration.
    """

    def __init__(
        self,
        llm: LLMClient,
        retriever: ToolRetriever,
        template_library: TemplateLibrary | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.template_library = template_library
        self.config = config or PipelineConfig()

    def run_stages_1_to_4(
        self, retrieval_result: RetrievalResult
    ) -> tuple[TaskDAG | None, list[AgentPipelineResult]]:
        """Run Stages 1-4: analysis, alignment, generation, validation.

        Args:
            retrieval_result: Query and candidate agents from Stream B.

        Returns:
            Tuple of (TaskDAG, list of per-agent pipeline results).
            TaskDAG is None if Stage 1 fails.
        """
        query = retrieval_result.query

        # Stage 1: Task Analysis (shared across all agents)
        try:
            dag = analyse_task(query, self.llm)
        except Exception as e:
            logger.error("Stage 1 failed: %s", e)
            results = [
                AgentPipelineResult(
                    agent=agent,
                    error_code=TASK_ANALYSIS_FAILED,
                    error_detail=str(e),
                )
                for agent in retrieval_result.candidates
            ]
            return None, results

        # Stages 2-4: per agent
        agent_results: list[AgentPipelineResult] = []
        for agent in retrieval_result.candidates:
            result = self._process_agent(dag, agent)
            agent_results.append(result)

        return dag, agent_results

    def _process_agent(
        self, dag: TaskDAG, agent: CandidateAgent
    ) -> AgentPipelineResult:
        """Run Stages 2-4 for a single agent."""
        result = AgentPipelineResult(agent=agent)

        # Stage 2: Tool-Task Alignment
        try:
            result.alignment = align_tools_for_agent(
                dag, agent.agent_id, self.retriever, self.llm
            )
        except Exception as e:
            logger.error("Stage 2 failed for %s: %s", agent.agent_id, e)
            result.error_code = ALIGNMENT_FAILED
            result.error_detail = str(e)
            return result

        # Stage 3: Probe Generation
        try:
            result.probe_plan = generate_probe_plan(
                query=dag.query,
                dag=dag,
                alignment=result.alignment,
                llm=self.llm,
                template_library=self.template_library,
                budget=self.config.probe_budget,
                total_timeout=self.config.total_timeout,
            )
        except Exception as e:
            logger.error("Stage 3 failed for %s: %s", agent.agent_id, e)
            result.error_code = PROBE_GENERATION_FAILED
            result.error_detail = str(e)
            return result

        # Stage 4: Probe Validation
        try:
            validated, _validation_results = validate_plan(
                result.probe_plan,
                difficulty_tolerance=self.config.difficulty_tolerance,
                min_rubric_dimensions=self.config.min_rubric_dimensions,
            )
            result.validated_plan = validated
        except Exception as e:
            logger.error("Stage 4 failed for %s: %s", agent.agent_id, e)
            result.error_code = PROBE_VALIDATION_FAILED
            result.error_detail = str(e)
            return result

        return result

    def score_agent_results(
        self,
        agent_result: AgentPipelineResult,
        execution_results: list[ProbeExecutionResult],
    ) -> RankedAgent:
        """Score an agent after probe execution (called after Stream C returns).

        Args:
            agent_result: The pipeline result from Stages 1-4.
            execution_results: Results from Stream C probe execution.

        Returns:
            RankedAgent with final score and confidence.
        """
        agent = agent_result.agent
        agent_result.execution_results = execution_results

        # Construct prior from metadata
        coverage = (
            agent_result.alignment.coverage_score
            if agent_result.alignment
            else 0.0
        )
        prior = construct_prior(
            retrieval_score=agent.retrieval_score,
            coverage_score=coverage,
            arena_elo=agent.arena_elo,
            community_rating=agent.community_rating,
        )

        # Filter execution results by interaction confidence
        probes = agent_result.validated_plan.probes if agent_result.validated_plan else []
        probe_map = {p.probe_id: p for p in probes}

        scored_probes = []
        observed_scores = []
        summaries = []

        for exec_result in execution_results:
            confidence = assess_confidence(exec_result)
            if not confidence.sufficient:
                logger.info(
                    "Discarding result for %s/%s: %s",
                    agent.agent_id, exec_result.probe_id, confidence.reason,
                )
                summaries.append(f"{exec_result.probe_id}: discarded (low confidence)")
                continue

            probe = probe_map.get(exec_result.probe_id)
            if probe is None:
                continue

            # For MVP: binary scoring based on success flag
            # TODO: replace with LLM judge for rubric-based partial credit
            score = 1.0 if exec_result.success else 0.0
            scored_probes.append(probe)
            observed_scores.append(score)
            status = "PASS" if score >= 0.5 else "FAIL"
            summaries.append(f"{probe.probe_id}: {status}")

        # Score the agent
        try:
            estimate = score_agent(prior, scored_probes, observed_scores)
            agent_result.estimate = estimate
        except Exception as e:
            logger.error("Scoring failed for %s: %s", agent.agent_id, e)
            agent_result.error_code = SCORING_FAILED
            agent_result.error_detail = str(e)
            estimate = PosteriorEstimate(
                theta=prior.mu,
                sigma=prior.sigma,
                confidence=0.0,
                n_probes=0,
                testability_tier="UNTESTABLE",
                prior_influence=1.0,
            )

        return RankedAgent(
            agent_id=agent.agent_id,
            theta=estimate.theta,
            sigma=estimate.sigma,
            confidence=estimate.confidence,
            testability_tier=estimate.testability_tier,
            probe_summary="; ".join(summaries) if summaries else "No probes executed",
            prior_influence=estimate.prior_influence,
        )
