# Evidence Cards observation ablation

This experiment changes only how retrieval observations are shown to the
policy.  SEARCH, EXPAND, READ, FINISH, the Controller, the state-conditioned
schema, the retriever, and the validator remain unchanged.

## Modes

* `raw`: the original policy-visible observation.  This is the true baseline;
  repeated source text is not removed.
* `program`: a deterministic Evidence Registry deduplicates stable IDs and
  renders one card per source.  Each card contains only its source content and
  the operations that are already legal for it.
* `reader`: an additional structured-output Reader summarizes the new
  observation with source references and quotes.  The main Agent still owns
  assessment and action selection.
* `reader_assessed`: the reader mode plus an advisory resolved/still-missing
  assessment.  It is diagnostic only; the Controller does not force FINISH or
  continuation.

All modes use the same question, Initial Skill, substrate, embedding index,
model, seed, budget, and temperature.  Reader calls and tokens are reported
separately from Agent calls and tokens.

## Context contract for organized modes

```text
Original question
Previous assessment — model judgment, not verified fact
Previous action and outcome
Evidence cards (one card per stable source ID)
Global SEARCH instruction
Remaining budget
```

READ, EXPAND, and FINISH are attached to the relevant card.  There is no
second dynamic `Available actions` menu.  SEARCH remains part of the fixed
protocol and is constrained by the state-conditioned schema.

## Running a pilot

Prepare a 60-question HotpotQA split (20 direct, 20 bridge, 20 comparison),
then run:

```bash
PYTHONPATH=src python scripts/run_observation_ablation.py \
  --substrate /path/to/hotpotqa/substrate \
  --split /path/to/hotpotqa/observation60.jsonl \
  --skill skills/workflow_seed.md \
  --output-root /path/to/runs/observation_ablation_60 \
  --expected-count 60
```

Each mode is written to its own directory and can be resumed independently.
The benchmark runner saves `io_trace.json`, which includes the rendered card
view and Reader usage for each step.

To render the same saved episode under one mode without running the Target
Agent, use `scripts/replay_observation_context.py` with the corresponding
configuration.  This is the fixed-trajectory replay portion of the study.
