# Qwen V3.1 Typed-Reference Ablation — 2026-08-06

## Result

V3.1 did not improve Qwen. Across the same 20 HotpotQA questions and A/B/C
skills, Luna-as-judge accuracy fell from 13/60 to 4/60. Invalid attempts rose
from 454 to 517 and total tokens rose from 1,726,718 to 1,815,749.

The one-namespace interface reduced node-type and repeated-read errors, but
created a larger failure class: Qwen generated E#/S#/C# references that were
not present in the current frozen snapshot. These availability errors grew
from 34 V3 index-out-of-range errors to 112 V3.1 reference-not-available
errors.

## Controlled setup

- Questions: `hotpotqa_resample20_seed20260805`, 20 questions.
- Corpus/substrate: `artifacts/hotpotqa_benchmark_exact`.
- Policy model: Ollama `qwen3.5:9b`, temperature 0, think disabled.
- Same embedding, substrate, enabled expansions, 10 environment steps,
  12 policy attempts, and 12,000 retrieved-token budget as baseline V3.
- Skills A/B/C preserve their V3 retrieval strategies; only reference
  terminology and recovery codes were adapted to typed refs.
- Judge: OpenAI `gpt-5.6-luna`, generated answer plus gold answer only; the
  judge did not see retrieved evidence. Blank answers were automatically wrong.
- This is one run per condition, so results are an ablation observation rather
  than a statistical causal estimate.

## Accuracy and cost

| Skill | Workflow | Luna judge | Contain | Exact | Finished episodes | Invalid | Ref errors | Calls | Total tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | V3 | 6/20 | 6/20 | 2/20 | 6 | 137 | 58 | 194 | 522,875 |
| A | V3.1 | 3/20 | 3/20 | 1/20 | 4 | 143 | 48 | 202 | 545,780 |
| B | V3 | 3/20 | 2/20 | 0/20 | 3 | 174 | 144 | 228 | 677,806 |
| B | V3.1 | 0/20 | 0/20 | 0/20 | 0 | 200 | 181 | 240 | 682,930 |
| C | V3 | 4/20 | 4/20 | 3/20 | 5 | 143 | 39 | 197 | 526,037 |
| C | V3.1 | 1/20 | 0/20 | 0/20 | 2 | 174 | 49 | 220 | 587,039 |
| **Total** | **V3** | **13/60** | **12/60** | **5/60** | **14** | **454** | **241** | **619** | **1,726,718** |
| **Total** | **V3.1** | **4/60** | **3/60** | **1/60** | **6** | **517** | **278** | **662** | **1,815,749** |

Aggregate V3.1 changes relative to V3:

- Luna judge: -9 answers (-69.2%).
- Finished episodes: 14 to 6.
- Invalid attempts: +63 (+13.9%).
- Policy calls: +43 (+6.9%).
- Total tokens: +89,031 (+5.2%).
- Retrieved tokens: 99,174 to 78,705 (-20.6%).
- Mean policy input per call: 2,648.8 to 2,555.6 (-3.5%).

The token regression was therefore not caused by a longer prompt. V3.1 was
slightly cheaper per call, but made more calls and produced 42.3% more output
tokens overall.

## Action behavior

| Metric | V3 | V3.1 | Change |
|---|---:|---:|---:|
| SEARCH | 323 | 337 | +14 |
| EXPAND | 89 | 173 | +84 |
| READ | 179 | 116 | -63 |
| FINISH attempts | 28 | 25 | -3 |
| SEARCH after observation | 263 | 277 | +14 |
| SEARCH after no progress | 234 | 251 | +17 |

The largest behavioral shift was EXPAND. In Skill B, EXPAND attempts increased
from 59 to 135 while SEARCH fell from 63 to 43. Qwen frequently constructed an
E# source even when that entity ref was not visible, so the typed prefix acted
more like a generatable action placeholder than a grounded pointer.

## Invalid-attempt attribution

| Error family | V3 | V3.1 | Change |
|---|---:|---:|---:|
| Index out of range / reference not available | 34 | 112 | +78 |
| Node/reference type mismatch | 96 | 80 | -16 |
| Chunk not readable | 111 | 86 | -25 |
| Duplicate action | 213 | 228 | +15 |
| Invalid structured policy response | 0 | 11 | +11 |

V3.1's 112 `reference_not_available` attempts split as follows:

| Skill | Total | Previously visible but now folded | Never visible/invented | Action family |
|---|---:|---:|---:|---|
| A | 13 | 11 | 2 | 13 FINISH |
| B | 93 | 0 | 93 | 91 EXPAND, 2 FINISH |
| C | 6 | 1 | 5 | 2 EXPAND, 4 FINISH |
| **Total** | **112** | **12** | **100** | **93 EXPAND, 19 FINISH** |

Two different interface failures occurred:

1. After READ folded sentences into a full chunk, Skill A often emitted an old
   S# instead of citing the visible read C#. These refs had existed earlier in
   the episode but were intentionally absent from the current frozen map.
2. Skill B invented E# refs and attempted entity expansion without first
   retrieving a currently visible entity. Typed prefixes made the required
   node type legible, but did not ground the numeric target.

## Answer flips

There were no incorrect-to-correct Luna-judge flips from V3 to V3.1. V3.1 lost
nine previously correct cases. Examples include:

- A, Peter Daou website slogan: correct `media for the 65.8 million` became the
  unrelated `A willing foe, and sea room`.
- A, Honky Tonk Angels record label and detective-novelist parent: both became
  blank after budget exhaustion.
- B lost all three prior correct answers: Atlantic Ocean, Kingscliff, and
  political activist.
- C lost Atlantic Ocean, Monkey Gland, and political activist; its one judged
  correct answer (`villanelle`) was already correct under baseline V3.

## Interpretation

The original V3 dual namespace was not the dominant cause of Qwen's failures.
Its context-local references had an important grounding property: every number
referred to something visible in the current screen, and READ replaced folded
sentence citations with current chunk citations.

V3.1 made type semantics clearer but made handles look reusable or
constructible. Qwen learned the shape `E#`, `S#`, or `C#` without reliably
copying an available target. The B procedure amplified this because it strongly
encouraged entity expansion. C's explicit reference gate did not solve the
broader duplicate/action-grounding problem.

Do not replace baseline V3 with this V3.1 implementation for Qwen. A useful
next ablation would retain semantic memory and unrestricted SEARCH while
constraining only each structured action field to the refs visible in the
current frozen snapshot. That would test grounded field-level enums without
adding a Policy-visible Action Catalog or action-scope bias.

## Artifacts and verification

- Qwen results: `runs/hotpotqa_resample20_seed20260805/qwen/v3_typed_refs/`.
- Luna judge: `runs/hotpotqa_resample20_seed20260805/luna_judge_v3_typed_refs_20260806_retry/`.
- Config: `configs/agentic_hotpotqa_qwen35_9b_v3_typed_refs.yaml`.
- Skills: `skills/hotpotqa_v31_{a,b,c}_*.md`.
- Full pytest suite passed; four pre-existing conditional tests were skipped.
- One failed judge output directory from the sandbox-blocked first attempt is
  retained for audit and excluded from all results.
