# Evaluation

## Fixed experiment contract

Compare models or Skills only when the HotpotQA question IDs, substrate,
embedding model, enabled expansions, initial budgets, and evaluation contract
are identical. The supplied Luna and Qwen configs both begin with 10 retrieval
steps, 12 Policy attempts, and 12,000 retrieved tokens.

The three Skill files are an intended research variable. The guarded procedure
is post-hoc and must be labelled experimental in reports.

## Metrics

- Exact match and normalized contain accuracy.
- Luna-as-judge semantic correctness.
- Answered episodes and budget exhaustion.
- Invalid attempts grouped by error code.
- Policy calls, input/output/reasoning/total tokens.
- Retrieved tokens.
- SEARCH, EXPAND, READ, and FINISH counts.
- Search-after-observation and search-after-no-progress rates.

Judge input contains the question, final prediction, and gold answer only. Gold
answers never enter the target Policy, Context Builder, Retriever, or episode
state.

## Artifact interpretation

Use the trajectory embedded in `episode.json` to analyze decisions, frozen ref
maps, validation, raw observations, state transitions, aggregate usage, and
termination. Use `conversation.json` for the compact SkillOpt projection; the
two target-prompt text files preserve the initial Policy input.

An invalid attempt is an interface/control failure, not a failed retrieval. A
wrong answer with valid evidence navigation is a reasoning or stopping failure.
A correct answer with weak citations should be reported separately from answer
accuracy rather than silently treated as the same metric.
