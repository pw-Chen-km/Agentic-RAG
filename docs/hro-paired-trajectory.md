# HRO Paired Trajectory Summary

HRO (Hindsight Reflection Optimization) is the offline Meta-analysis layer on
the `hro-meta-stage` branch. It does not change the Agent policy, action space,
retriever, controller, or state management.

For each same-question TRAIN parent/candidate pair, the runtime writes a
deterministic `comparison_pairs.json` receipt containing:

- normalized action sequences and the first divergence;
- unique newly observed evidence counts and their delta;
- policy-call and total-token deltas;
- `true`, `false`, or `null` answer correctness (`null` is unknown, not wrong);
- an informative-pair decision and the exclusion reason when applicable.

Only informative pairs are sent to the existing Meta diagnosis stage. A batch
with fewer than three informative pairs keeps the parent Skill and records
`skip_meta_update`; it never fabricates a lesson from identical trajectories.
The HRO merge configuration accepts at most one logical Skill edit per batch.
Validation/test trajectories are never used as Meta evidence.

The source of truth is GitHub. Local and JJ checkouts should use the same HRO
branch and be updated with a fast-forward pull; experiment data and model
weights are separate from this source-only repository.
