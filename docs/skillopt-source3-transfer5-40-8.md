# Independent source training and five-dataset transfer (40 / 8)

This design starts twelve independent runs from the same archived Initial Skill:
HotpotQA, Medical, and Novel, each using raw, organized,
organized_support_labels, and progress_abstracted reflection. It is not
leave-one-out training on a merged source corpus.

Each rollout batch contains at most 40 questions under one unchanged Skill.
Reflection partitions successes and failures into groups of at most eight,
then merges and ranks proposals before at most one validation-gated update.
One epoch retains the partial Medical (12) and Novel (2) final batches. The
maximum update counts per arm are 5, 11, and 11, respectively.

## Data and evaluation

Fresh grouped, stratified seed-42 splits retain historical exposure only as
audit metadata. They do not claim that every test question was never analyzed.

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| HotpotQA | 200 | 200 | 600 |
| Medical | 412 | 412 | 1,238 |
| Novel | 402 | 402 | 1,206 |
| MuSiQue | sealed unused 200 | sealed unused 200 | 600 |
| 2Wiki | sealed unused 198 | sealed unused 198 | 595 |

The nine previously identified invalid-coordinate 2Wiki rows remain excluded.
MuSiQue and 2Wiki are answer-only transfer targets; missing progress mappings
do not block inference. Search indexes and embeddings are not rebuilt.

All twelve successful training runs must be sealed before any final test,
including Initial. Every learned Skill is evaluated on every target dataset;
Initial is evaluated once per target. This gives 65 tasks and 55,107 answer
episodes, excluding training, validation, and Judge calls. Target Qwen runs
without thinking; Optimizer thinking remains enabled. The existing one-pass
Qwen binary Judge runs without thinking and the summary uses saved llm_acc,
not contain accuracy. The final report includes per-target paired bootstrap
intervals, costs, missing/empty answers, and equal-dataset macro averages.

## Process isolation, capacity, and resume

An experiment owns a process, output directory, checkpoint, and immutable lane
configuration. Target and reflection workers are each one. The outer scheduler
requires a measured capacity certificate even for one lane. It may use two or
four lanes only after the matching capacity stage passes; it never changes
Ollama services or unloads another model. Existing JJ port 11435 is preserved.

Capacity replays include a real dynamic-schema Target request and a real
eight-trajectory analyst request. Warmup is excluded from throughput timing;
all stages use the same total workload. GPU and container memory must retain
at least 10% headroom, with no CPU offload or service errors, and a higher
concurrency must improve throughput by at least 10%. Prompt tokenization is
checked before any large inference; truncation is not a capacity success.

The project-scoped checkpoint wrapper atomically commits completed update
state, history, current Skill, and best Skill. It preserves a zero validation
score on resume, which the native fallback expression otherwise loses. The
installed SkillOpt package is not edited. Live workers and completed receipts
can be recovered. Failed or ambiguous partial training is paused, not blindly
replayed; an explicit, hash-checked safe-boundary authorization is required.

## Entry points

- `prepare_multidataset_skillopt.py`: history_policy=fresh; transfer_only roles.
- `write_multidataset_skillopt_configs.py`: single_source_cross_dataset_v1.
- `run_skillopt_capacity.py`: offline by default, --execute for real probes.
- `run_multidataset_experiments.py`: validate/plan by default; measured capacity
  and explicit execution are required to start workers.
- `run_prepared_skillopt_test.py --aggregate`: only after all 65 tasks finish.

Do not resume old experiments under these new settings. Do not interpret an
unmeasured or blocked capacity report as permission to start training.
