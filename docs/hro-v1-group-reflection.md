# HRO v1 grouped reflection

HRO v1 is an offline analysis layer over paired TRAIN trajectories. It does
not change the Agent, Policy, Retriever, State Management, evaluator, or test
split.

## Action-space contract

`src/arag_skillopt/action_space.py` is the single source of truth for the
current action menu. The LinearRAG-enabled menu is:

`semantic_search`, Direct-BM25 `entity_hop`, `linear_hop`, `read_chunk`, and
`stop_and_answer`.

The menu is recorded in the Phase 1 contract and appended to every optimizer
context together with a deterministic objective: maximize answer accuracy and
useful new evidence, minimize policy/tool calls and retrieved tokens, report
every enabled action separately, and keep runtime/Judge failures unknown.
When a config disables `linear_hop`, the contract records that fact rather than
pretending the action was available. Enabling it requires the corpus graph
index and uses the same snippet-first/read policy as the Agent runtime.

## Stage A

`MetaConfig(hro_v1=True)` groups eligible same-question pairs into G00/G01/G10/G11
based only on backend answer labels. Any missing, corrupt, runtime, or Judge
label is `UNSCORED` and is excluded from reflection. `comparison_pairs.json`
still contains one explicit row per TRAIN question, including exclusion
markers. Each Stage A context is deterministic, limited to five pairs, and
contains the changed rules, paired action/evidence/cost deltas, and group
statistics. It is diagnosis-only; no Skill edit or gate decision is allowed.

## Stage B

All group summaries and Stage A reflections are saved in
`stage_b_context.json` and passed to one bounded merge/proposer call. HRO v1
allows at most three logical edits per Meta update. Validation is the only gate
input; validation/test trajectories never enter the proposer context.

The example YAML keeps `hro_v1: false` so archived v6r3 runs cannot be
silently reinterpreted. New runs that intentionally adopt HRO v1 should set
it to `true` and use a new output directory.

