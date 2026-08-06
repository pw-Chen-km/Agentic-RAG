# HotpotQA Resample-20: Luna V2 Compact B/C

## Experimental contract

- Questions: `data/evaluations/hotpotqa_resample20_seed20260805/questions.jsonl`
- Substrate: `artifacts/hotpotqa_benchmark_exact`
- Policy model: `gpt-5.6-luna`
- V2 Compact config: `configs/agentic_hotpotqa_luna_v2_compact.yaml`
- Skill B: unchanged `skills/hotpotqa_v2_b_chunk_entity_expert.md`
- Skill C: unchanged `skills/hotpotqa_v2_c_posthoc_guarded.md`
- Budgets, enabled expansions, embedding substrate, and evaluation split are
  unchanged from the Luna V2/V3 resample-20 matrix.
- Answer quality uses the same answer-vs-GT Luna judge contract as the previous
  matrix. Supporting evidence is not visible to the judge.

V2 Compact retains V2's cumulative `selected_evidence_refs`, S#/C#/E# handles,
handle resolution, validation, and Policy output schema. Only the
Policy-visible context serialization is changed. The Controller still keeps
the complete internal `PolicyView` for deterministic resolution and
validation.

The compact projection removes repeated `can_expand`/`can_read` fields, the
verbose per-expansion tutorial, and full structured attempt records. It shows
semantic `known_nodes`, a minimal `expand_sources` legality map, selected
evidence, latest observation, semantic action history, assessment, and budget.

## Results

| Skill | Version | Luna judge | Contain | Exact | Invalid | Calls | Input tokens/call | Total tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B | V2 | 18/20 | 16/20 | 4/20 | 7 | 102 | 4,603.6 | 491,030 |
| B | V2 Compact | 17/20 | 16/20 | 8/20 | 7 | 122 | 4,557.5 | 583,606 |
| B | V3 | 18/20 | 16/20 | 9/20 | 1 | 98 | 3,472.1 | 356,430 |
| C | V2 | 17/20 | 17/20 | 8/20 | 0 | 86 | 3,663.9 | 333,616 |
| C | V2 Compact | 18/20 | 17/20 | 9/20 | 1 | 86 | 3,432.5 | 314,984 |
| C | V3 | 19/20 | 17/20 | 10/20 | 0 | 93 | 3,445.6 | 339,476 |

## Context-length distribution

| Skill/version | Mean | Median | P90 | Max | First call mean | Last call mean |
|---|---:|---:|---:|---:|---:|---:|
| V2-B | 4,603.6 | 4,132.0 | 6,939.0 | 9,614 | 2,333.3 | 6,216.9 |
| Compact-B | 4,557.5 | 3,875.5 | 6,684.5 | 11,032 | 2,138.3 | 6,037.1 |
| V3-B | 3,472.1 | 3,218.0 | 5,031.7 | 6,159 | 2,131.3 | 4,519.7 |
| V2-C | 3,663.9 | 3,383.5 | 5,074.0 | 9,468 | 2,478.3 | 4,733.4 |
| Compact-C | 3,432.5 | 3,105.0 | 5,425.0 | 7,179 | 2,283.3 | 4,111.9 |
| V3-C | 3,445.6 | 3,217.0 | 4,937.0 | 6,021 | 2,285.3 | 4,088.2 |

Compact reduced the first-call input by 8.4% for B and 7.9% for C. For C,
where the number of calls stayed at 86, this produced a 6.3% reduction in
average policy input and a 5.6% reduction in total tokens. For B, the Agent
made 20 more calls, so the 1.0% lower average policy input did not compensate
for the longer trajectory; total tokens increased by 18.9%.

## Paired answer changes

Only one Luna-judge verdict changed for each Skill relative to original V2:

- Skill B regressed on the Darkthrone/Motörhead question. Original V2 answered
  Motörhead; Compact exhausted its trajectory without retrieving the likened
  band.
- Skill C improved on Peter Daou's website slogan. Original V2 attached the
  SpaceCollective slogan to the wrong website; Compact retrieved and answered
  `Media for the 65.8 million.`

These are retrieval-policy changes, not reference-parsing corrections. V2-B
and Compact-B both had seven `selected_evidence_not_eligible` invalid attempts.
V2-C had no invalid attempt; Compact-C had one `duplicate_action`.

## Interpretation

The compact interface is not merely passive token compression. Changing the
layout changes Luna's policy trajectory. Skill C is the cleanest token result
because the call count is identical: the compact layout is measurably cheaper
and accuracy changed by only one paired item. Skill B demonstrates that a
shorter prompt can still cost more end-to-end when it induces additional
SEARCH/EXPAND decisions.

V3 remains substantially more compact than V2 Compact for Skill B because V3
also changes state semantics: automatic semantic memory, context-local indices,
chunk folding, and no selected-evidence/typed-handle legality interface. V2
Compact deliberately does not adopt those mechanisms, so it is still a V2
ablation rather than a renamed V3.

The accuracy changes are one-run observations over 20 questions and should not
be treated as statistically established causal effects. Repeated runs or a
frozen-context decision replay are needed to separate layout effects from Luna
run-to-run variation.

## Artifacts

- Valid B run: `runs/hotpotqa_resample20_seed20260805/luna/v2_compact_valid/b_chunk_entity_expert`
- Valid C run: `runs/hotpotqa_resample20_seed20260805/luna/v2_compact_valid/c_posthoc_guarded`
- Judge: `runs/hotpotqa_resample20_seed20260805/luna_judge_v2_compact_20260805`

An earlier directory at
`runs/hotpotqa_resample20_seed20260805/luna/v2_compact/b_chunk_entity_expert`
contains a 0-call environment-key failure. It is retained only as an audit
artifact and is excluded from every result above.
