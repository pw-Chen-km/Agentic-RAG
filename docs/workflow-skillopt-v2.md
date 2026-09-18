# Workflow-Aware SkillOpt v2

This version keeps the Agent runtime unchanged and changes only how the
training workflow edits a Skill.

## Source of truth

`skills/workflow_rules_v2.json` is the editable rule source.  The Agent-facing
file `skills/workflow_seed_v2.md` is generated from it.  The runtime guidance
and answer contract are fixed.  Retrieval and Meta may edit only `R` rules;
answer rules are present for traceability but are locked in this experiment.

## One training batch

The configured schedule is 40 train questions, followed by reflection groups
of at most 5 questions.  Reflection returns at most two `add`, `replace`, or
`delete` edits, or `no_change`.  The batch merge also keeps at most two edits,
checks duplicate rules and token limits, and then runs the existing validation
gate.  A rejected candidate can be replayed on the same train questions for
Meta analysis only; it never changes the active Skill.

Meta compares parent and candidate workflows from the same question and keeps
the parent/candidate Skill hashes, costs and outcomes.  It can emit the same
small rule-edit format.  It does not receive validation or test trajectories.

Reflection and Meta inputs are limited to 16,000 counted tokens.  Oversized
groups are split; an individual oversized case is recorded and skipped rather
than stopping the complete stage.  Every request and split is checkpointed.

## Run a fresh experiment

Use the independent v2 configuration:

```bash
PYTHONPATH=src python -m agentic_rag.skillopt.workflow.cli \
  --config configs/workflow_hotpotqa_jj27b_v2.json
```

Validate paths, hashes and configuration without model calls:

```bash
PYTHONPATH=src python -m agentic_rag.skillopt.workflow.cli \
  --config configs/workflow_hotpotqa_jj27b_v2.json --dry-run
```

The v2 output directory is separate from all legacy results.  Receipts include
the parent and candidate rule JSON, changed rule IDs, token counts, hashes,
validation decision and replay purpose, so a candidate can be inspected or
resumed without changing older runs.
