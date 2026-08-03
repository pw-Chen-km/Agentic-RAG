# Agentic RAG

Typed multi-granularity substrate plus a query-time Agent Harness for English
HotpotQA contexts. The offline builder creates stable Document, Chunk, Sentence,
Entity, Mention, and EntityAlias records. The online controller executes one
structured `SEARCH`, `EXPAND`, `READ`, or `FINISH` decision per step.

## Quick start

```bash
uv sync --extra dev
uv run python -m spacy download en_core_web_sm

uv run agentic-rag build data/hotpotqa.json artifacts/hotpotqa \
  --split dev --corpus-id hotpotqa_dev

uv run agentic-rag validate artifacts/hotpotqa
uv run agentic-rag search artifacts/hotpotqa "Warsaw" \
  --scope-id QUESTION_ID --method lexical --target entity
```

### Linux and Windows

The package supports Python 3.12 on Linux and Windows. Use the same locked
environment on both platforms:

```bash
uv sync --frozen --extra dev
uv run pytest
```

PowerShell uses backticks for multi-line commands and `$env:` for environment
variables. For example:

```powershell
$env:OPENAI_API_KEY = "..."
uv run agentic-rag run artifacts/hotpotqa `
  "Where was Marie Curie born?" `
  --scope-id QUESTION_ID `
  --skill-file skills/initial.md `
  --config configs/agentic_hotpotqa_smoke.yaml `
  --output runs
```

Logical corpus and question IDs remain unchanged across operating systems.
When an ID contains a Windows-forbidden filename character such as `:`, only
its on-disk artifact directory is projected to a deterministic portable name;
`episode.json`, evaluation records, and Controller state retain the original
ID. GitHub Actions runs the default test suite on both Ubuntu and Windows.

## Query-time agent

Set `OPENAI_API_KEY`, then run one scope-isolated episode:

```bash
uv run agentic-rag run artifacts/hotpotqa \
  "Where was Marie Curie born?" \
  --scope-id QUESTION_ID \
  --skill-file skills/initial.md \
  --config configs/agentic_hotpotqa.yaml \
  --output runs
```

The formal config uses `gpt-5.6-terra`. The smoke config uses
`gpt-5.6-luna`:

```bash
uv run agentic-rag run artifacts/hotpotqa \
  "Where was Marie Curie born?" \
  --scope-id QUESTION_ID \
  --skill-file skills/initial.md \
  --config configs/agentic_hotpotqa_smoke.yaml \
  --output runs
```

The test suite never calls OpenAI by default. To opt into the live Luna
structured-output smoke test:

```bash
RUN_OPENAI_SMOKE=1 OPENAI_API_KEY=... \
  uv run pytest -m openai_smoke tests/test_openai_smoke.py
```

### Local Ollama

Policy decisions and final answer generation can both run against a local
Ollama server through its [native chat API](https://docs.ollama.com/api/chat)
and [JSON-schema structured outputs](https://docs.ollama.com/capabilities/structured-outputs).
Start Ollama yourself and make sure the configured model is already installed;
the Agent Harness never starts the daemon or pulls a model.
Use a local model rather than an Ollama `:cloud` model. The included
low-resource example uses
[`qwen2.5:3b`](https://ollama.com/library/qwen2.5):

```bash
ollama serve
ollama pull qwen2.5:3b
```

Then run:

```bash
uv run agentic-rag run artifacts/hotpotqa \
  "Where was Marie Curie born?" \
  --scope-id QUESTION_ID \
  --skill-file skills/initial.md \
  --config configs/agentic_hotpotqa_ollama.yaml \
  --output runs
```

The Ollama config independently selects the `policy` and `answer` providers, so
mixed OpenAI/Ollama configurations are also supported. `model` is required for
each Ollama provider. The local default host is
`http://localhost:11434`; Ollama Cloud hosts are rejected because this harness
requires structured policy output that Ollama Cloud does not support.

`num_ctx` defaults to `32768`. A 32k
[context window](https://docs.ollama.com/context-length) can materially increase
RAM/VRAM use, so lower it when running on constrained hardware while keeping
enough room for the fixed protocol, skill, retrieval history, and output. Other
deterministic runtime controls include `temperature: 0`, `think: false`,
`timeout_seconds`, `keep_alive`, and `max_retries`. No `OPENAI_API_KEY` is
needed when both providers use Ollama.

To exercise structured policy and answer output against an already installed
local model (the test never pulls one):

```bash
RUN_OLLAMA_SMOKE=1 OLLAMA_MODEL=qwen2.5:3b \
  uv run pytest -m ollama_smoke tests/test_ollama_smoke.py
```

The Policy-facing expansion space defaults to four explicit kinds:

- `ENTITY_MENTIONED_IN_SENTENCE`
- `SENTENCE_MENTIONS_ENTITY`
- `ENTITY_CO_OCCURS_ENTITY_SENTENCE`
- `CHUNK_ADJACENT_CHUNK`

The engine implements four additional expansion kinds for controlled
ablations. `SENTENCE_PART_OF_CHUNK` is intentionally absent: every complete
Sentence result includes `parent_chunk_id`, `document_id`, and `title`, so the
agent can READ the parent Chunk directly.

Chunk previews are navigation only. A complete Sentence can be submitted as
Sentence evidence, while a Chunk can be submitted only after `READ`.

### Episode-local node handles

The Policy LLM never operates on stable substrate node IDs. Every Episode owns
a typed, bidirectional handle registry:

```text
E1 <-> stable Entity ID
S1 <-> stable Sentence ID
C1 <-> stable Chunk ID
```

Policy observations and decisions use only these short handles. Each semantic
handle also states what is currently legal: `can_expand`, `can_read`,
`has_been_read`, and `can_use_as_evidence`. The Validator resolves handles back
to stable IDs before applying visibility, scope, evidence-eligibility, READ,
and duplicate-action rules. Substrate records and internal Controller state
continue to use stable IDs.

`episode.json` and `io_trace.json` preserve both representations for replay and
audit. `conversation.json`, Policy messages, and agent-visible observations are
handle-safe, so SkillOpt optimizes retrieval behavior instead of learning how
to copy database identifiers.

Each run writes an immutable directory containing `episode.json`,
SkillOpt-compatible `conversation.json`, the target prompts, a skill snapshot,
and the effective configuration. Gold supporting facts remain isolated in the
evaluation sidecar and are never loaded by the query-time stack.

## A-RAG benchmark profiles

The generic `arag_benchmark_exact` source format supports every evaluation set
published in `Ayanami0730/rag_test` at revision
`b9198a5a8702cc35c6df7542529357a9af95d928`:

| `--dataset` | Questions | Source Chunks | Answer mode | Reported metrics |
|---|---:|---:|---|---|
| `musique` | 1,000 | 1,354 | short | LLM-Acc, Contain-Acc |
| `hotpotqa` | 1,000 | 1,311 | short | LLM-Acc, Contain-Acc |
| `2wikimultihop` | 1,000 | 658 | short | LLM-Acc, Contain-Acc |
| `medical` | 2,062 | 225 | long | LLM-Acc |
| `novel` | 2,010 | 1,117 | long | LLM-Acc |

Each dataset gets one isolated global scope. Source Chunk boundaries and numeric
adjacency are preserved. The retrievable substrate never contains questions,
answers, or evidence fields; question metadata is written only to the evaluation
sidecar. Runtime question IDs are deterministic row-based IDs, while the raw ID
and row index remain available for evaluation joins. This also safely handles
the duplicated raw question ID in the Novel set.

The source can be either the dataset directory itself or the root containing
the five dataset subdirectories:

```bash
uv run agentic-rag build data/rag_test artifacts/musique_benchmark_exact \
  --corpus-id musique_benchmark_exact \
  --dataset musique \
  --source-format arag_benchmark_exact
```

Add `--validate-benchmark-profile` for a paper-dataset build to require the
reference question, Chunk, unique-ID, and task-type counts. Leave it off for
small fixtures and development subsets.

## Reduced HotpotQA benchmark parity

Use `hotpotqa_benchmark_exact` to build from the A-RAG/LinearRAG reduced
benchmark directory containing `chunks.json` and `questions.json`:

```bash
uv run agentic-rag build data/benchmark_exact/hotpotqa \
  artifacts/hotpotqa_benchmark_exact \
  --config configs/benchmark_exact_hotpotqa.yaml
```

This mode preserves all source Chunk boundaries and numeric adjacency. All
questions use the single scope `hotpotqa:benchmark_exact:dev`; question IDs do
not restrict retrieval to their original HotpotQA contexts. Questions and gold
answers are written only to `evaluation/benchmark_questions.parquet`, which the
query-time `Substrate` never loads.

## HotpotQA × SkillOpt workflow smoke

This workflow uses the pinned A-RAG reduced benchmark revision
`b9198a5a8702cc35c6df7542529357a9af95d928`. Download only the HotpotQA files,
then install the optional SkillOpt runtime:

```bash
uv run --with huggingface-hub python -c \
  "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Ayanami0730/rag_test', repo_type='dataset', revision='b9198a5a8702cc35c6df7542529357a9af95d928', allow_patterns=['hotpotqa/*'], local_dir='data/arag_hotpotqa')"

uv sync --extra dev --extra skillopt
uv run python -m spacy download en_core_web_sm
```

Prepare the deterministic 6/6/6 split. Each split contains four bridge and
two comparison questions, all using the global benchmark scope
`hotpotqa:benchmark_exact:dev`:

```bash
uv run agentic-rag skillopt-prepare \
  --dataset-dir data/arag_hotpotqa \
  --split-dir data/skillopt/hotpotqa_smoke
```

Build the substrate separately. `skillopt-prepare` never builds or mutates an
index:

```bash
uv run agentic-rag build data/arag_hotpotqa/hotpotqa \
  artifacts/hotpotqa_benchmark_exact \
  --config configs/benchmark_exact_hotpotqa.yaml
```

Set `OPENAI_API_KEY` in the process environment and start the native SkillOpt
v0.2.0 trainer:

```bash
uv run agentic-rag skillopt-train \
  artifacts/hotpotqa_benchmark_exact \
  --split-dir data/skillopt/hotpotqa_smoke \
  --agent-config configs/agentic_hotpotqa_skillopt_smoke.yaml \
  --skillopt-config configs/skillopt_hotpotqa_smoke.yaml \
  --skill-file skills/hotpotqa_skillopt_initial.md \
  --output runs/skillopt_hotpotqa_smoke
```

The command bridges `OPENAI_API_KEY` to SkillOpt's OpenAI-compatible
environment variables only in process memory. The key is never inserted into
the flattened trainer config or an artifact. Policy, Answer, Judge, and
Optimizer roles are configured for `gpt-5.6-luna`.

The 6/6/6 split means **18 unique question IDs**, not exactly 18 Episode
executions. Native `ReflACTTrainer` first scores the initial skill on the
validation set, validates candidate skills during training, and with
`eval_test: true` evaluates both initial and best skills on test. This is the
official validation-gated behavior; the workflow does not run a separate
one-off `agentic-rag run` before SkillOpt.

Every rollout task directory contains the six normal Harness artifacts plus
`io_trace.json`, `evaluation.json`, and `rollout_result.json`. The run root
contains SkillOpt's native steps and candidate skills, `best_skill.md`, the
pinned split manifest, native metrics, `initial_metrics.json`,
`final_metrics.json`, `token_summary.json`, and `workflow_summary.json`. The
last two combine exact Policy/Answer/Judge counters with SkillOpt's optimizer
token tracker. Gold answers are available only to post-Episode scoring and
train reflection; target Policy/Answer inputs never receive them.

This configuration is labeled `workflow_smoke`. It is not A-RAG paper parity:
it uses only 18 unique questions and a Luna judge.

## Profile-driven SkillOpt workflow

The same workflow supports all five A-RAG profiles. `skillopt-prepare` creates
deterministic 6/6/6 train, validation, and test files. Each six-item split first
covers every available task type, then assigns remaining positions by the
profile's reference distribution:

| Dataset | Per-split task allocation | SkillOpt metrics |
|---|---|---|
| MuSiQue | 3×2-hop, 2×3-hop, 1×4-hop | hard LLM-Acc, soft Contain-Acc |
| HotpotQA | 4×bridge, 2×comparison | hard LLM-Acc, soft Contain-Acc |
| 2WikiMultiHopQA | 2×compositional, 1×inference, 2×comparison, 1×bridge-comparison | hard LLM-Acc, soft Contain-Acc |
| Medical | 2×fact retrieval, 2×complex reasoning, 1×contextual summarize, 1×creative generation | hard/soft LLM-Acc |
| Novel | 2×fact retrieval, 2×complex reasoning, 1×contextual summarize, 1×creative generation | hard/soft LLM-Acc |

For example, prepare and train a MuSiQue smoke workflow:

```bash
uv run agentic-rag skillopt-prepare \
  --dataset-dir data/rag_test \
  --dataset musique \
  --split-dir data/skillopt/musique_smoke

uv run agentic-rag skillopt-train \
  artifacts/musique_benchmark_exact \
  --split-dir data/skillopt/musique_smoke \
  --agent-config configs/agentic_arag_skillopt_smoke.yaml \
  --skillopt-config configs/skillopt_arag_smoke.yaml \
  --skill-file skills/arag_skillopt_initial.md \
  --output runs/skillopt_musique_smoke
```

Use `--allow-subset` only for small local fixtures. Generic split manifests
record the profile, task allocation, answer mode, metric contract, source
hashes, raw question IDs, and source row indices. Training validates these
against the substrate before the first Target Agent call. Medical and Novel
automatically use the long-answer prompt; the other profiles retain concise
answer generation.

The paid live test is opt-in and reuses already prepared artifacts:

```bash
RUN_SKILLOPT_SMOKE=1 \
SKILLOPT_SMOKE_SUBSTRATE=artifacts/hotpotqa_benchmark_exact \
SKILLOPT_SMOKE_SPLIT_DIR=data/skillopt/hotpotqa_smoke \
OPENAI_API_KEY=... \
uv run pytest -m skillopt_smoke tests/test_skillopt_live_smoke.py
```
