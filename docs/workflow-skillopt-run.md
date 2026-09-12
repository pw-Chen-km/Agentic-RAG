# Workflow-Aware SkillOpt v1

This repository contains a complete, resumable three-stage training runner:
Retrieval, Meta workflow comparison, and Answer. It does not rebuild a
substrate or embedding index. The target agent sees only the question and the
normal runtime; gold answers are used only by the post-episode evaluator.

## Run on a new machine

```bash
uv sync --extra skillopt
uv run agentic-rag-workflow --config configs/workflow_skillopt.json --dry-run
uv run agentic-rag-workflow --config configs/workflow_skillopt.json
```

The configuration must name portable `train` and `validation` JSONL files,
the existing `substrate`, `agent_config`, a seed `skill`, and an `output`
directory. Each question row requires `id`, `question`, `answer`, and
`scope_id`; train and validation IDs/questions must not overlap. Ollama must
already expose the target, optimizer, and judge models at the configured host.

Each training batch is 40 questions; each reflection request has at most five
records. The last partial batch is retained. Results are written atomically
with request hashes, model digests, candidate archives, validation decisions,
replay purpose and a checkpoint. Re-running the same command resumes; a
changed input or model contract requires a new output directory.

`--demo` runs only a deterministic installation/resume check and must never be
used as research evidence. `--dry-run` makes no model call. The runner uses
one process per output directory; run independent datasets in independent
directories.
