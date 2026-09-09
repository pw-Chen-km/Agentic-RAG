# Three training sources, five test datasets: 40 / 5

This run changes only the reflection group limit from 8 to 5. Use
`write_multidataset_skillopt_configs.py --design single_source_cross_dataset_v1
--reflection-minibatch-size 5` with a new output directory. The API and CLI
default remain 8 solely for reproducing the preceding 40 / 8 configuration.
The execution manifest and every task's configuration explicitly record 5.

## Update sequence

Each rollout batch contains at most 40 questions using one Skill version.
Reflection analyzes at most 5 trajectories per call, separately grouping
successful and unsuccessful episodes according to the existing SkillOpt logic.
All reflection groups finish before merging, ranking, and at most one update
opportunity for that rollout batch. Validation still decides whether to accept
the candidate. Final incomplete batches are retained.

Smaller groups reduce the trajectories in a single analysis request, not
necessarily the total experiment's tokens: more calls repeat the shared
instructions and may generate more analysis text. No trajectory truncation,
context reduction, prompt rewrite, or automatic further splitting is authorized.

## Unchanged experiment

- HotpotQA, Medical, and Novel independently train four representations each.
- The same freshly prepared splits, Initial Skill, Qwen digest, prompts, model
  settings, validation gate, substrate, and embeddings are retained.
- All 12 runs must finish and be sealed before the 65 final-test tasks start.
- Final tests still comprise 55,107 answers across five target datasets.
- Old configurations and results are preserved; no Git push is performed.

## Capacity approval

Capacity replay must contain exactly 5 distinct actual training trajectories
in its full analyst request, matching the manifest. Its full input and reserved
output must fit context 262,144. Backend tokenization is checked before sending
the long request; saved actual endpoint usage is required when its chat template
differs from the backend template. Nothing is silently truncated to pass.

Only measured safe lanes may start the scheduler. Start with the GPU1 lanes;
GPU0 remains excluded until the other user's work is confirmed finished. A
failed capacity check leaves formal training unstarted and records the reason.
