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

Each run writes an immutable directory containing `episode.json`,
SkillOpt-compatible `conversation.json`, the target prompts, a skill snapshot,
and the effective configuration. Gold supporting facts remain isolated in the
evaluation sidecar and are never loaded by the query-time stack.
