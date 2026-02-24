# Agent Search Engine — Probing Pipeline (Stream D)

The probing component of the Agent Search Engine, a research project for COMP0031 at UCL Computer Science. Given a user query and candidate MCP server agents, this pipeline generates targeted lightweight probes, executes them, and produces scored rankings using Bayesian Item Response Theory (BIRT).

## Architecture

```
USER QUERY
    │
    ▼
[Stage 1] Task Analysis         query → TaskDAG (1 LLM call)
    │
    ▼  (per candidate agent)
[Stage 2] Tool-Task Alignment   FAISS retrieval + LLM reranking
    │
    ▼
[Stage 3] Probe Plan Generation template cache → LLM fallback
    │
    ▼
[Stage 4] Probe Validation      rule-based schema/rubric checks
    │
    ▼
[Execution] MCP tool calls      Stream C sandbox (external)
    │
    ▼
[Scoring] BIRT                  Bayesian 2PL IRT with informative priors
    │
    ▼
RANKED AGENT LIST with confidence intervals
```

**Key properties:**
- 1–3 probes per agent, under 30s total execution
- Discriminative critical-path strategy maximises information gain per probe
- Bayesian scoring provides calibrated confidence intervals even with few probes
- Graceful degradation: agents are classified as FULLY_PROBED, PARTIALLY_PROBED, or UNTESTABLE

## Quick Start

### Prerequisites

- Python 3.11+
- An API key for one of the supported LLM providers

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd probing-pipeline

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

### Environment Setup

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API key
# Cerebras free tier: https://cloud.cerebras.ai/ (30 RPM, 1M tokens/day)
export CEREBRAS_API_KEY=your_key_here
```

### Run the Pipeline

```bash
# Default: uses Cerebras with sample MCP servers
python scripts/run_pipeline.py --query "Find the current weather in London"

# With OpenAI
python scripts/run_pipeline.py \
  --query "Search GitHub for Python ML repos and summarise the top 5" \
  --api-key-env OPENAI_API_KEY \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini

# Custom probe budget
python scripts/run_pipeline.py \
  --query "Translate this README from English to Spanish" \
  --budget 3
```

### Run Tests

```bash
# All tests (82 tests)
pytest

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
pytest --cov=src --cov-report=term-missing
```

## Project Structure

```
probing-pipeline/
├── config/
│   ├── default.yaml                 # Pipeline configuration
│   └── prompts/                     # LLM prompt templates
│       ├── task_analysis.txt        # Stage 1 prompt
│       ├── tool_alignment.txt       # Stage 2 reranker prompt
│       └── probe_generation.txt     # Stage 3 probe generation prompt
├── scripts/
│   ├── run_pipeline.py              # End-to-end pipeline runner
│   ├── bootstrap_templates.py       # Generate probe template library
│   └── evaluate_strategies.py       # Compare probe selection strategies
├── src/
│   ├── models/                      # Data models
│   │   ├── task.py                  # SubtaskNode, TaskDAG
│   │   ├── alignment.py             # ToolAlignment, AlignmentMap
│   │   ├── probe.py                 # Probe, ProbePlan, ProbeTemplate
│   │   ├── scoring.py               # GaussianPrior, PosteriorEstimate
│   │   └── integration.py           # CandidateAgent, RankedAgent, etc.
│   ├── stages/                      # Pipeline stages
│   │   ├── task_analysis.py         # Stage 1: query → TaskDAG
│   │   ├── tool_alignment.py        # Stage 2: FAISS + LLM reranking
│   │   ├── probe_generation.py      # Stage 3: discriminative probe selection
│   │   └── probe_validation.py      # Stage 4: rule-based validation
│   ├── scoring/                     # BIRT scoring model
│   │   ├── prior.py                 # Weighted Gaussian prior from metadata
│   │   ├── birt.py                  # 2PL Bayesian IRT update
│   │   └── confidence.py            # Interaction confidence assessment
│   ├── tool_index/                  # FAISS tool index
│   │   ├── indexer.py               # Offline index builder
│   │   └── retriever.py             # Online retrieval with filtering
│   ├── templates/                   # Probe template library
│   │   ├── library.py               # Template storage and lookup
│   │   └── generator.py             # Back-instruct template generation
│   ├── llm/
│   │   └── client.py                # Model-agnostic LLM client
│   └── pipeline.py                  # End-to-end orchestrator
└── tests/
    ├── unit/                        # 79 unit tests
    ├── integration/                 # 3 integration tests
    └── fixtures/                    # Sample MCP servers and queries
```

## Configuration

All configuration lives in `config/default.yaml`:

| Section | Key | Default | Description |
|---|---|---|---|
| `llm.model` | — | `qwen-3-235b-a22b-instruct-2507` | Controller LLM model |
| `llm.base_url` | — | `https://api.cerebras.ai/v1` | LLM API endpoint |
| `llm.temperature` | — | `0` | Deterministic output |
| `probe.budget_per_agent` | — | `2` | Max probes per agent |
| `probe.total_timeout_seconds` | — | `30` | Max wall-clock per agent |
| `scoring.prior_sigma_tight` | — | `0.3` | Prior σ with 3+ metadata signals |
| `scoring.prior_sigma_diffuse` | — | `0.5` | Prior σ with fewer signals |
| `tool_index.embedding_model` | — | `all-MiniLM-L6-v2` | Embedding model for FAISS |
| `tool_index.retrieval_k` | — | `20` | Candidates per subtask retrieval |
| `validation.difficulty_tolerance` | — | `0.3` | Max probe/tool difficulty gap |

## Supported LLM Providers

The pipeline uses an OpenAI-compatible API interface. Any provider with a compatible endpoint works:

| Provider | Model | Endpoint | Notes |
|---|---|---|---|
| **Cerebras** (default) | Qwen3-235B-A22B-Instruct | `https://api.cerebras.ai/v1` | Free tier: 30 RPM, 1M tok/day |
| OpenAI | GPT-4o-mini | `https://api.openai.com/v1` | Paid |
| Together AI | Kimi K2 | `https://api.together.xyz/v1` | Paid |

Switch providers by changing `llm.base_url` and `llm.model` in config, or via CLI flags.

## Scoring Model

The pipeline uses a **Bayesian 2-Parameter Logistic IRT** model:

```
P(correct | θ, d, a) = σ(a · (θ − d))
```

- **θ** (theta): agent capability (what we estimate)
- **d**: probe difficulty (from Stage 3)
- **a**: probe discrimination (how sharply the probe separates agents)

The prior is constructed from metadata signals:

| Signal | Weight | Source |
|---|---|---|
| Arena ELO | 0.40 | Agent Arena (if available) |
| Retrieval similarity | 0.25 | Stream B score |
| Tool-task coverage | 0.20 | Stage 2 coverage |
| Community rating | 0.15 | TAAFT (if available) |

Posterior updates use Laplace approximation, providing calibrated uncertainty estimates (σ) even with 1–3 probes.

## Integration

### Input (from Stream B)

The pipeline receives a `RetrievalResult` containing the query and top-k candidate agents with metadata.

### Output (to Stream C)

For each agent, the pipeline produces a `ProbeExecutionRequest` with validated probes to be executed in the sandbox.

### Output (to Stream E)

Final output is a list of `RankedAgent` objects:

```python
RankedAgent(
    agent_id="weather-server",
    theta=0.73,          # capability score
    sigma=0.21,          # uncertainty
    confidence=0.79,     # 1 - sigma
    testability_tier="FULLY_PROBED",
    probe_summary="P1: PASS; P2: PASS",
    prior_influence=0.18 # how much prior vs evidence drives the score
)
```

## Scripts

### `run_pipeline.py`

End-to-end pipeline runner. Builds a FAISS index from MCP server definitions, runs all 4 stages, simulates execution, and prints ranked results.

### `bootstrap_templates.py`

Generates a probe template library from MCP server schemas using the back-instruct method:

```bash
python scripts/bootstrap_templates.py \
  --servers tests/fixtures/sample_mcp_servers.json \
  --output data/templates.json \
  --api-key $CEREBRAS_API_KEY
```

### `evaluate_strategies.py`

Compares probe selection strategies with synthetic data:

```bash
python scripts/evaluate_strategies.py --budget 2
```

## Team

**Course:** COMP0031, UCL Computer Science
**Supervisors:** Prof. Emine Yilmaz (primary), Bin (co-supervisor)

| Stream | Responsibility |
|---|---|
| A | Data collection and curation |
| B | Agent indexing and retrieval |
| C | Sandbox execution environment |
| **D** | **Probing and evaluation (this repo)** |
| E | API and UI integration |
