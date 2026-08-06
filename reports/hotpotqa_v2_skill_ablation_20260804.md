# HotpotQA V2 Skill Ablation

Date: 2026-08-04

## Evaluation contract

- Dataset: unchanged HotpotQA Test-6
- Architecture: `single_agent_v2`
- Policy models: Ollama `qwen3.5:9b` and OpenAI `gpt-5.6-luna`
- Retrieval substrate: `artifacts/hotpotqa_benchmark_exact`
- Qwen config: `configs/agentic_hotpotqa_qwen35_9b_v2.yaml`
- Luna config: `configs/agentic_hotpotqa_luna_v2.yaml`
- Max environment steps: 10
- Max policy attempts: 12
- Max retrieved tokens: 12,000
- Skill A: evidence-adaptive policy
- Skill B: fixed Chunk-to-Entity expert procedure
- Skill C: post-hoc Evidence/Progress/Novelty/Handle gates
- Embedding model, substrate, decoding settings, and Test-6 order were fixed.
- Hugging Face was placed in offline mode after one excluded preliminary B run
  encountered blocked cache metadata requests. The valid runs use the same
  already-cached embedding weights.

## Aggregate results

Official metrics use the supplied gold strings. Evidence-consistent accuracy
uses the local HotpotQA supporting evidence and treats incomplete compound
answers as incorrect.

| Model | Skill | Official Contain | Official Exact | Evidence-consistent | Invalid | Repairs | Calls | Input | Output | Total tokens | Retrieved |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen 9B | A | 3/6 | 2/6 | 2/6 | 27 | 6 | 51 | 155,679 | 8,405 | 164,084 | 8,171 |
| Qwen 9B | B | 2/6 | 0/6 | 2/6 | 32 | 5 | 56 | 276,387 | 8,320 | 284,707 | 18,732 |
| Qwen 9B | C | **4/6** | **2/6** | **3/6** | 27 | 10 | 53 | 172,678 | 7,669 | 180,347 | 8,053 |
| Luna | A | 4/6 | 2/6 | 5/6 | **0** | 0 | 25 | 91,058 | 5,053 | **96,111** | 4,559 |
| Luna | B | 4/6 | 2/6 | 5/6 | 2 | 0 | 29 | 132,728 | 5,447 | 138,175 | 14,538 |
| Luna | C | **5/6** | **2/6** | **6/6** | **0** | 0 | **23** | 95,186 | 4,518 | 99,704 | 5,554 |

## Action distribution

Counts include invalid Policy decisions, because they still consume one Policy
call and are part of the Skill behaviour.

| Model | Skill | SEARCH | EXPAND | READ | FINISH |
|---|---|---:|---:|---:|---:|
| Qwen 9B | A | 38 | 4 | 3 | 6 |
| Qwen 9B | B | 12 | 28 | 11 | 5 |
| Qwen 9B | C | 16 | 14 | 13 | 10 |
| Luna | A | 17 | 1 | 1 | 6 |
| Luna | B | 12 | 3 | 7 | 7 |
| Luna | C | 12 | 3 | 2 | 6 |

B followed its required first action for both models: all 6/6 questions began
with `SEARCH BM25 -> CHUNK`.

## Invalid-error distribution

### Qwen 9B

- A: `duplicate_action=21`, `unknown_handle=4`,
  `finish_evidence_not_selected=2`.
- B: `duplicate_action=15`, `unknown_handle=12`,
  `source_not_complete=2`, `finish_evidence_not_selected=1`,
  `handle_type_mismatch=1`, `selected_evidence_not_eligible=1`.
- C: `unknown_handle=11`, `duplicate_action=10`,
  `finish_evidence_not_selected=6`.

### Luna

- A: no invalid actions.
- B: `selected_evidence_not_eligible=2`.
- C: no invalid actions.

## Per-question evidence audit

| Evidence-supported answer | Qwen A | Qwen B | Qwen C | Luna A | Luna B | Luna C |
|---|---|---|---|---|---|---|
| more than 330 million people | Empty | Empty | Empty | Correct | Correct | Correct |
| American schoolteacher and publisher | Missing `American` | Only `schoolteacher` | Empty | Missing `American` | Missing `American` | Correct |
| Matt Groening | Correct | Empty | Correct | Correct | Correct | Correct |
| Livin La Vida Loco | Answers official `Loca` | Core `Loco`, verbose | Answers official `Loca` | Correct | Correct | Correct |
| Midnight Madness | Correct | Correct | Correct | Correct | Correct | Correct |
| Tori Amos | Empty | Empty | Correct | Correct | Correct | Correct |

## V2 interpretation

1. Qwen benefits most from C on accuracy, but the improvement does not come
   from reliable self-validation: C still has 27 invalid attempts and 10
   repairs. It mainly forces more varied retrieval before budget exhaustion.
2. Qwen B is the worst efficiency result. Its fixed procedure induces 28
   EXPAND decisions, including 12 unknown handles and 15 duplicates. It costs
   74% more tokens than Qwen A without improving evidence accuracy.
3. Luna can execute all three V2 Skills. C is the only V2 variation that
   preserves every required compound qualifier and reaches evidence-consistent
   6/6.
4. Luna B follows the intended procedure, but V2 makes it expensive: it reads
   seven Chunks and retrieves 14,538 tokens. Its accuracy remains 5/6 because
   it drops `American` from the occupation answer.
5. The model gap is primarily protocol execution. Qwen repeatedly chooses
   unknown handles, duplicate semantic actions, and unselected evidence. Luna
   almost eliminates those failures under the same Controller and retrieval
   environment.

## V2 versus V3

| Model | Skill | V2 evidence-correct | V3 evidence-correct | V2 invalid | V3 invalid | V2 tokens | V3 tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen 9B | A | 2/6 | 2/6 | 27 | 19 | 164,084 | 110,691 |
| Qwen 9B | B | 2/6 | 0/6 | 32 | 44 | 284,707 | 211,951 |
| Qwen 9B | C | 3/6 | 1/6 | 27 | 38 | 180,347 | 152,577 |
| Luna | A | 5/6 | 5/6 | 0 | 0 | 96,111 | 95,507 |
| Luna | B | 5/6 | **6/6** | 2 | 1 | 138,175 | **83,810** |
| Luna | C | **6/6** | 5/6 | 0 | 1 | 99,704 | 89,747 |

V3 is clearly better for Luna B: it obtains 6/6 with 39% fewer tokens than V2
B. V2 C obtains Luna's best V2 completeness, while V3 C is cheaper but misses
one compound answer. For Qwen, V2 improves B/C answer accuracy but remains
substantially more expensive; neither protocol solves the small model's
bookkeeping failures.

## Item-4 annotation defect

The official gold is the Ricky Martin song `Livin' la Vida Loca`. The source
record's supporting evidence says the Coal Chamber tour is `Livin La Vida
Loco`. The question asks for the tour, so evidence-consistent scoring accepts
`Loco` and rejects answers that identify only `Loca`. Official Contain and
Exact are retained separately for reproducibility.

## Artifacts

- `skills/hotpotqa_v2_a_evidence_adaptive.md`
- `skills/hotpotqa_v2_b_chunk_entity_expert.md`
- `skills/hotpotqa_v2_c_posthoc_guarded.md`
- `runs/hotpotqa_v2_skill_ablation_20260804/qwen/a_evidence_adaptive/summary.json`
- `runs/hotpotqa_v2_skill_ablation_20260804/qwen/b_chunk_entity_expert_valid/summary.json`
- `runs/hotpotqa_v2_skill_ablation_20260804/qwen/c_posthoc_guarded/summary.json`
- `runs/hotpotqa_v2_skill_ablation_20260804/luna/a_evidence_adaptive/summary.json`
- `runs/hotpotqa_v2_skill_ablation_20260804/luna/b_chunk_entity_expert/summary.json`
- `runs/hotpotqa_v2_skill_ablation_20260804/luna/c_posthoc_guarded/summary.json`

## Luna-as-Judge follow-up

The six result sets were subsequently scored by the existing A-RAG-compatible
`gpt-5.6-luna` semantic judge using generated answer and gold answer only.

| Target model | A | B | C |
|---|---:|---:|---:|
| Qwen 9B | 4/6 | 2/6 | 4/6 |
| Luna | 6/6 | 6/6 | 6/6 |

The full verdict matrix, Judge usage, and interpretation are in
`reports/hotpotqa_v2_luna_judge_20260805.md`.
