# Evaluation

## Fixed experiment contract

Compare models or Skills only when the dataset profile, question IDs,
substrate lineage, embedding model, enabled expansions, initial budgets, and
evaluation contract are identical. The supplied Luna and Qwen configs both
begin with 10 retrieval steps, 12 Policy attempts, and 12,000 retrieved tokens.

V3.2 declares five profiles: 2WikiMultiHopQA, HotpotQA, Medical
(GraphRAG-Bench), MuSiQue, and Novel (GraphRAG-Bench). Each profile fixes the
dataset revision, scope, source counts, task-type normalization, answer mode,
and reported metrics. Never compare rows produced from different profiles as
though they were one homogeneous benchmark.

The three Skill files are an intended research variable. The guarded procedure
is post-hoc and must be labelled experimental in reports.

## Metrics

- Exact match and normalized contain accuracy for short-answer diagnostics.
- Luna-as-judge semantic correctness for every dataset.
- Answered episodes and budget exhaustion.
- Invalid attempts grouped by error code.
- Policy calls, input/output/reasoning/total tokens.
- Retrieved tokens.
- SEARCH, EXPAND, READ, and FINISH counts.
- Search-after-observation and search-after-no-progress rates.

Judge input contains the question, final prediction, and gold answer only. Gold
answers never enter the target Policy, Context Builder, Retriever, or episode
state.

2WikiMultiHopQA, HotpotQA, and MuSiQue report `llm_acc` and `contain_acc`.
Medical and Novel contain long-form summarization/generation tasks, so their
formal profile reports `llm_acc` only; contain and exact may remain in raw run
summaries as diagnostics but are not their benchmark score.

## Five-dataset execution

`scripts/run_benchmark_eval.py` runs one declared dataset and rejects a split
whose `source` or `scope_id` does not match that profile.
`scripts/run_benchmark_matrix.py` invokes that same runner for all five
datasets, preserving separate result directories and contracts. It does not
pool scores across incompatible answer modes.

Deterministic SkillOpt splits are created with:

```powershell
agentic-rag skillopt-prepare `
  --dataset-dir data\rag_test `
  --dataset medical `
  --split-dir data\skillopt\medical_smoke
```

The same command accepts every canonical dataset key. SkillOpt train,
validation, and test cardinalities are read from the signed split manifest,
not hard-coded by the Policy provider config.

## Artifact interpretation

Use the trajectory embedded in `episode.json` to analyze decisions, frozen ref
maps, validation, raw observations, state transitions, aggregate usage, and
termination. Use `conversation.json` for the compact SkillOpt projection; the
two target-prompt text files preserve the initial Policy input.

An invalid attempt is an interface/control failure, not a failed retrieval. A
wrong answer with valid evidence navigation is a reasoning or stopping failure.
A correct answer with weak citations should be reported separately from answer
accuracy rather than silently treated as the same metric.
