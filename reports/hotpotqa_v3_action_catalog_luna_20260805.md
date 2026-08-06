# HotpotQA Resample-20: Luna V3 Valid-Action Catalog Variation

## Variation

This variation keeps the V3 Semantic Memory, State Management, frozen
memory/citation maps, Policy schema, retrieval environment, Skill text, and
budgets unchanged. It adds one Policy-visible `valid_action_catalog` generated
from the same frozen context snapshot:

- all six SEARCH method/target templates on every turn;
- each structurally valid EXPAND kind/source-memory-index pair;
- every currently readable unread Chunk memory index;
- FINISH plus the currently available citation indices when citations exist.

The catalog reports structural executability, not semantic relevance or answer
sufficiency. Natural-language query choices and duplicate-action semantics
cannot be exhaustively enumerated.

## Experimental contract

- Questions: `data/evaluations/hotpotqa_resample20_seed20260805/questions.jsonl`
- Policy model: `gpt-5.6-luna`
- Baseline config: `configs/agentic_hotpotqa_luna_v3.yaml`
- Variation config: `configs/agentic_hotpotqa_luna_v3_action_catalog.yaml`
- Skills: unchanged V3 B and C Skill files
- Judge: same answer-vs-GT Luna judge; supporting evidence is not visible

## Results

| Skill | Version | Luna judge | Contain | Exact | Invalid | Calls | Input/call | Total tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B | V3 baseline | 18/20 | 16/20 | 9/20 | 1 | 98 | 3,472.1 | 356,430 |
| B | V3 + catalog | 18/20 | 16/20 | 7/20 | 0 | 94 | 4,112.5 | 402,714 |
| C | V3 baseline | 19/20 | 17/20 | 10/20 | 0 | 93 | 3,445.6 | 339,476 |
| C | V3 + catalog | 20/20 | 19/20 | 14/20 | 0 | 83 | 4,052.5 | 352,019 |
| B+C | V3 baseline | 37/40 | 33/40 | 19/40 | 1 | 191 | 3,459.2 | 695,906 |
| B+C | V3 + catalog | 38/40 | 35/40 | 21/40 | 0 | 177 | 4,084.3 | 754,733 |

The catalog reduced calls by 14 (7.3%) and invalid attempts from one to zero,
but raised average input per call by 18.1% and total tokens by 8.5%. Judge
accuracy increased by one question.

## Action behavior

| Version | SEARCH | EXPAND | READ | FINISH | SEARCH after observation | SEARCH after no progress |
|---|---:|---:|---:|---:|---:|---:|
| V3 baseline B+C | 108 | 7 | 36 | 40 | 68 | 8 |
| V3 + catalog B+C | 94 | 4 | 39 | 40 | 54 | 5 |

The catalog shifted the Policy toward fewer new searches/expansions and three
additional READ actions. This is evidence of the anticipated action-target
scope effect, although it did not reduce answer accuracy in this run.

## Context-length distribution

| Skill/version | Mean | Median | P90 | Max | First call mean | Last call mean |
|---|---:|---:|---:|---:|---:|---:|
| V3-B | 3,472.1 | 3,218 | 5,031.7 | 6,159 | 2,131.3 | 4,519.7 |
| Catalog-B | 4,112.5 | 4,197 | 5,864.0 | 8,147 | 2,388.3 | 5,167.4 |
| V3-C | 3,445.6 | 3,217 | 4,937.0 | 6,021 | 2,285.3 | 4,088.2 |
| Catalog-C | 4,052.5 | 3,672 | 6,004.8 | 7,941 | 2,542.3 | 4,674.6 |

Even with empty memory, the fixed SEARCH catalog and catalog protocol add
about 257 input tokens to the first call. Later prompts grow further because
every valid EXPAND source pair and READ target is enumerated.

## Paired answer change

Only one Luna-judge verdict changed across B and C. The original V3-C answered
the noodle translation question with `To cut`; the catalog variation answered
`Little hairs`, matching the GT.

The variation used five calls instead of eight:

1. BM25 Sentence search
2. BM25 Sentence search
3. Dense Sentence search
4. READ a catalog-listed Chunk
5. FINISH with the Chunk citation

The catalog made the readable target explicit, but it did not identify the
semantically correct Chunk. Because the baseline already exposed the same
memory item and index, one run cannot establish that the catalog caused the
referent correction rather than ordinary Luna trajectory variation.

## Interpretation

For Luna on this 40-episode sample, showing valid actions was slightly better
for answer accuracy and operational validity, and reduced Policy calls. It was
clearly worse for token efficiency. It also reduced post-observation SEARCH,
confirming that explicit targets influence action selection.

The result does not justify replacing baseline V3 yet. The accuracy difference
is one paired item, while the token increase is systematic. Repeated runs or a
frozen-context action-choice replay are required before attributing the one
answer improvement causally to the catalog.

## Artifacts

- B run: `runs/hotpotqa_resample20_seed20260805/luna/v3_action_catalog/b_chunk_entity_expert`
- C run: `runs/hotpotqa_resample20_seed20260805/luna/v3_action_catalog/c_posthoc_guarded`
- Judge: `runs/hotpotqa_resample20_seed20260805/luna_judge_v3_action_catalog_20260805`
