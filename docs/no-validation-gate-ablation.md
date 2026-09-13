# No-validation-gate SkillOpt ablation

This run asks whether the optimizer can propose a useful Skill when the
validation gate is not allowed to reject the proposal. It is separate from
the production workflow and writes to a new output directory.

For every non-empty candidate proposal the runner:

1. evaluates the parent and candidate on the fixed validation split;
2. evaluates the parent and candidate on the fixed test split;
3. records both metric pairs in the batch receipt;
4. adopts the candidate when `use_validation_gate` is false.

The test split is never included in reflection, merge, replay, or Meta cases,
and its score never decides adoption. It is an evaluation-only result. This
keeps the experiment from using test answers to train the next Skill while
still showing whether an unblocked proposal transfers to held-out questions.

The optimizer uses Qwen 3.8 27B with `think: true`. The target Agent uses the
same model with thinking disabled. Rollout batches contain 40 questions and a
reflection request contains at most five cases. If a serialized reflection
request exceeds 120,000 characters, the runner halves that request until it
fits (usually one case); this is recorded through the stable reflect files.

Run on JJ with:

```bash
uv run agentic-rag-workflow --config configs/workflow_hotpotqa_jj27b_no_validation_gate.json
```

The output contains `retrieval/`, `meta/`, and `answer/` receipts. Each
`completed.json` records the candidate Skill, validation metrics, test metrics,
and the explicit reason `validation_gate_disabled_candidate_adopted` when a
candidate is adopted. The final Skill and final test metrics are in
`final_skill.md` and `summary.json`.
