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
import time
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
from src.tool_index.indexer import ToolRecord
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
    timings: dict[str, float] = field(default_factory=dict)  # stage -> seconds


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

    def run_stage_1(self, query: str) -> tuple[TaskDAG | None, float]:
        """Run Stage 1: Task Analysis.

        Args:
            query: The user query string.

        Returns:
            Tuple of (TaskDAG or None on failure, elapsed seconds).
        """
        t0 = time.monotonic()
        try:
            dag = analyse_task(query, self.llm)
        except Exception as e:
            logger.error("Stage 1 failed: %s", e)
            return None, time.monotonic() - t0
        return dag, time.monotonic() - t0

    def run_stages_2_to_4_for_agent(
        self, dag: TaskDAG, agent: CandidateAgent
    ) -> AgentPipelineResult:
        """Run Stages 2-4 for a single agent with per-stage timing.

        Args:
            dag: TaskDAG from Stage 1.
            agent: The candidate agent to evaluate.

        Returns:
            AgentPipelineResult with timings populated.
        """
        return self._process_agent(dag, agent)

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

        # Filter unavailable agents early
        available = [a for a in retrieval_result.candidates if a.is_available]
        skipped = [a for a in retrieval_result.candidates if not a.is_available]
        if skipped:
            logger.info(
                "Skipping %d unavailable agents: %s",
                len(skipped),
                ", ".join(a.agent_id for a in skipped),
            )

        # Stage 1: Task Analysis (shared across all agents)
        dag, _ = self.run_stage_1(query)
        if dag is None:
            results = [
                AgentPipelineResult(
                    agent=agent,
                    error_code=TASK_ANALYSIS_FAILED,
                    error_detail="Task analysis failed",
                )
                for agent in retrieval_result.candidates
            ]
            return None, results

        # Stages 2-4: per agent (only available agents)
        agent_results: list[AgentPipelineResult] = []
        for agent in skipped:
            agent_results.append(AgentPipelineResult(
                agent=agent,
                error_code=ALIGNMENT_FAILED,
                error_detail="Agent unavailable",
            ))
        for agent in available:
            result = self._process_agent(dag, agent)
            agent_results.append(result)

        return dag, agent_results

    def _process_agent(
        self, dag: TaskDAG, agent: CandidateAgent
    ) -> AgentPipelineResult:
        """Run Stages 2-4 for a single agent with per-stage timing."""
        result = AgentPipelineResult(agent=agent)

        # Build ephemeral FAISS index for inline tools (discarded after this method)
        extra_index = None
        if agent.tools:
            tool_records = [
                ToolRecord(
                    tool_name=t.name,
                    server_id=agent.agent_id,
                    description=t.description,
                    parameter_schema=t.input_schema,
                )
                for t in agent.tools
            ]
            extra_index = self.retriever.build_ephemeral_index(tool_records)

        if extra_index is not None:
            logger.info(
                "Agent %s: ephemeral index with %d tools",
                agent.agent_id, len(extra_index[1]),
            )
        else:
            logger.warning(
                "Agent %s: no inline tools — ephemeral index NOT built (agent.tools=%s)",
                agent.agent_id, "empty" if agent.tools is not None else "None",
            )

        # Build agent context for Stage 2
        agent_description = agent.description
        agent_capabilities = (
            agent.llm_extracted.capabilities
            if agent.llm_extracted
            else None
        )

        # Build limitations for Stage 3
        limitations = (
            agent.llm_extracted.limitations
            if agent.llm_extracted
            else None
        )

        # Stage 2: Tool-Task Alignment
        t0 = time.monotonic()
        try:
            result.alignment = align_tools_for_agent(
                dag, agent.agent_id, self.retriever, self.llm,
                agent_description=agent_description,
                agent_capabilities=agent_capabilities,
                extra_index=extra_index,
            )
        except Exception as e:
            logger.error("Stage 2 failed for %s: %s", agent.agent_id, e)
            result.error_code = ALIGNMENT_FAILED
            result.error_detail = str(e)
            result.timings["stage_2_alignment"] = time.monotonic() - t0
            return result
        result.timings["stage_2_alignment"] = time.monotonic() - t0

        # Stage 3: Probe Generation
        t0 = time.monotonic()
        try:
            result.probe_plan = generate_probe_plan(
                query=dag.query,
                dag=dag,
                alignment=result.alignment,
                llm=self.llm,
                template_library=self.template_library,
                budget=self.config.probe_budget,
                total_timeout=self.config.total_timeout,
                limitations=limitations,
            )
        except Exception as e:
            logger.error("Stage 3 failed for %s: %s", agent.agent_id, e)
            result.error_code = PROBE_GENERATION_FAILED
            result.error_detail = str(e)
            result.timings["stage_3_generation"] = time.monotonic() - t0
            return result
        result.timings["stage_3_generation"] = time.monotonic() - t0

        # Stage 4: Probe Validation
        t0 = time.monotonic()
        try:
            # Build tool schema map from alignment data for schema-aware validation
            tool_schemas = {
                a.tool_name: a.tool_parameter_schema
                for a in result.alignment.alignments
                if a.tool_parameter_schema
            } if result.alignment else {}

            validated, _validation_results = validate_plan(
                result.probe_plan,
                tool_schemas=tool_schemas,
                difficulty_tolerance=self.config.difficulty_tolerance,
                min_rubric_dimensions=self.config.min_rubric_dimensions,
            )
            result.validated_plan = validated
        except Exception as e:
            logger.error("Stage 4 failed for %s: %s", agent.agent_id, e)
            result.error_code = PROBE_VALIDATION_FAILED
            result.error_detail = str(e)
            result.timings["stage_4_validation"] = time.monotonic() - t0
            return result
        result.timings["stage_4_validation"] = time.monotonic() - t0

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
            retrieval_score=agent.score,
            coverage_score=coverage,
            arena_elo=agent.arena_elo,
            community_rating=agent.community_rating,
            documentation_quality=agent.documentation_quality,
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
                testability_tier=agent.testability_tier or "UNTESTABLE",
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
