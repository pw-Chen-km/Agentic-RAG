# Agentic RAG

A single-agent, multi-substrate retrieval system for HotpotQA. The repository
contains one production architecture: Semantic Memory, episode-local typed
references, centralized State Management, and a stateless Controller.

```mermaid
flowchart LR
    Q["Question"] --> CB["Context Builder"]
    SM["State Management"] --> CB
    SK["Selected Skill"] --> CB
    CB --> P["LLM Policy"]
    P --> C["Stateless Controller"]
    C --> RV["Reference Resolver + Validator"]
    RV --> ENV["Multi-substrate Retrieval Environment"]
    ENV --> O["Observation"]
    O --> SM
    C --> F["Final Answer"]
```

## Design

The Policy sees the question, neutral action protocol, selected Skill, last
assessment, complete visible Semantic Memory, semantic action history, and one
compact remaining-budget line. It chooses exactly one `SEARCH`, `EXPAND`,
`READ`, or `FINISH` action.

State Management is the only state owner. It accumulates novel entities,
sentences, chunks, observations, action history, usage, and budgets. The
Controller stores nothing; it resolves the current frozen `E#/S#/C#` map,
validates the decision, executes retrieval, and returns the observation to
State Management.

`SEARCH` never needs a reference. `EXPAND` and `READ` use currently visible
typed refs. `FINISH` answers directly and cites only a complete visible `S#` or
an already-read `C#`. Invalid attempts consume a Policy call but do not consume
a retrieval step.

See [architecture.md](docs/architecture.md),
[action-contract.md](docs/action-contract.md), and
[evaluation.md](docs/evaluation.md). For a clean-machine setup, pinned HotpotQA
data, local Qwen SkillOpt, and the A-RAG baseline procedure, use the
[reproduction and A-RAG baseline guide](docs/reproduction-and-arag-baseline.md).

## Install

Python 3.12 is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

For SkillOpt:

```powershell
pip install -e ".[dev,skillopt]"
```

Secrets belong in an untracked `.env` or the process environment. Never put an
API key in YAML, a trajectory, or a commit.

## Build a HotpotQA substrate

Query-scoped HotpotQA input:

```powershell
agentic-rag build data\hotpotqa.json artifacts\hotpotqa `
  --corpus-id hotpotqa-dev `
  --source-format hotpotqa_scoped
agentic-rag validate artifacts\hotpotqa
```

Pinned reduced benchmark input (`chunks.json` plus `questions.json`):

```powershell
agentic-rag build data\hotpotqa artifacts\hotpotqa-benchmark `
  --corpus-id hotpotqa-benchmark `
  --source-format hotpotqa_benchmark_exact `
  --validate-benchmark-profile
```

## Run one episode

Ollama Qwen:

```powershell
agentic-rag run artifacts\hotpotqa-benchmark `
  "Which person was born earlier?" `
  --scope-id hotpotqa:benchmark_exact:dev `
  --skill-file skills\baseline.md `
  --config configs\hotpotqa_qwen.yaml `
  --output runs\qwen
```

OpenAI Luna uses `configs/hotpotqa_luna.yaml` and reads `OPENAI_API_KEY` from
the environment.

Three interchangeable Skill conditions share the exact same runtime contract:

- `skills/baseline.md`: neutral evidence-adaptive baseline.
- `skills/chunk_entity_expert.md`: chunk-first, entity drill-down procedure.
- `skills/guarded_procedure.md`: post-hoc experimental failure guard.

## Artifacts

Each episode writes the final record, full trajectory, frozen reference maps,
Policy conversation, Skill snapshot, and effective configuration. Stable
substrate IDs remain audit-only and never enter Policy context.

## Evaluation and SkillOpt

```powershell
python scripts\run_hotpotqa_eval.py --help
python scripts\judge_hotpotqa.py --help
agentic-rag skillopt-prepare --help
agentic-rag skillopt-train --help
```

The standard metrics are exact match, contain accuracy, Luna-as-judge,
invalid attempts, Policy calls/tokens, retrieved tokens, and action counts.

## Verification

```powershell
pytest
python -m compileall src\agentic_rag
```

The multi-version research snapshot remains available in branch
`codex/pre-singularity-version-archive` and tag
`pre-singularity-v3.2-20260806`; it is not part of this mainline runtime.
