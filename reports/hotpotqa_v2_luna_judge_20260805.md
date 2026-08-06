# HotpotQA V2 Luna-as-Judge Evaluation

Date: 2026-08-05

## Judge contract

- Judge: OpenAI Responses `gpt-5.6-luna`
- Input: generated answer and gold answer only
- Supporting evidence visible to Judge: no
- Question visible to Judge prompt: no
- Rule: accept when the generated answer contains the key gold information,
  is factually consistent with it, and contains no contradiction
- Output: structured boolean `correct`
- Blank predictions: automatic incorrect without an API call
- Source predictions: the six valid V2 ablation runs

This is the existing A-RAG-compatible semantic judge protocol. It is separate
from normalized Exact, Contain, and evidence-consistent manual auditing.

## Aggregate results

| Target model | Skill | Luna Judge | Contain | Exact | Judge calls |
|---|---|---:|---:|---:|---:|
| Qwen 9B | A evidence-adaptive | 4/6 | 3/6 | 2/6 | 4 |
| Qwen 9B | B Chunk-to-Entity | 2/6 | 2/6 | 0/6 | 3 |
| Qwen 9B | C guarded | 4/6 | 4/6 | 2/6 | 4 |
| Luna | A evidence-adaptive | **6/6** | 4/6 | 2/6 | 6 |
| Luna | B Chunk-to-Entity | **6/6** | 4/6 | 2/6 | 6 |
| Luna | C guarded | **6/6** | 5/6 | 2/6 | 6 |

Across all six runs, the Judge made 29 provider calls and used 5,126 tokens:
4,206 input, 920 output, including 421 reasoning tokens reported within the
output usage.

## Per-question verdict matrix

| Question / Gold | Qwen A | Qwen B | Qwen C | Luna A | Luna B | Luna C |
|---|---|---|---|---|---|---|
| Q1 / more than 330 million people | Incorrect | Incorrect | Incorrect | Correct | Correct | Correct |
| Q2 / American schoolteacher and publisher | Correct | Correct | Incorrect | Correct | Correct | Correct |
| Q3 / Matt Groening | Correct | Incorrect | Correct | Correct | Correct | Correct |
| Q4 / Livin' la Vida Loca | Correct | Incorrect | Correct | Correct | Correct | Correct |
| Q5 / Midnight Madness | Correct | Correct | Correct | Correct | Correct | Correct |
| Q6 / Tori Amos | Incorrect | Incorrect | Correct | Correct | Correct | Correct |

## Notable Judge decisions

- The Judge accepts `schoolteacher and publisher` against `American
  schoolteacher and publisher`.
- It also accepts Qwen B's narrower `schoolteacher` answer. The protocol is
  therefore lenient about omitted compound qualifiers.
- It accepts a concise `Livin La Vida Loco` against gold `Livin' la Vida Loca`,
  treating the final-letter difference as semantically harmless.
- It rejects Qwen B's longer item-4 answer, which mentions the `Loca` song but
  then states that the tour is `Livin' La Vida Loco Tour`. The extra conflicting
  title makes this less equivalent to the gold than the concise variant.
- Seven blank predictions were assigned incorrect deterministically and did
  not consume Judge calls.

## Interpretation

1. Under A-RAG-style Luna LLM-Acc, all three Luna target runs score 6/6. This
   Judge therefore does not distinguish Luna A, B, and C on answer correctness;
   calls, invalid actions, and target-token cost remain necessary ablation
   metrics.
2. Qwen A and C both score 4/6, while B remains worst at 2/6. The semantic Judge
   does not change the conclusion that fixed Chunk-to-Entity procedure is hard
   for Qwen under V2.
3. Luna Judge is materially more permissive than Exact and sometimes more
   permissive than the intended compound-answer contract. Report LLM-Acc,
   Contain, and Exact together rather than replacing them with one score.
4. Because the Judge does not receive the question or supporting evidence, it
   cannot detect the Test-6 item-4 gold/evidence inconsistency. It only judges
   semantic compatibility with the supplied gold string.

## Artifacts

- `runs/hotpotqa_v2_skill_ablation_20260804/luna_judge_20260805/summary.json`
- `runs/hotpotqa_v2_skill_ablation_20260804/luna_judge_20260805/qwen_a.json`
- `runs/hotpotqa_v2_skill_ablation_20260804/luna_judge_20260805/qwen_b.json`
- `runs/hotpotqa_v2_skill_ablation_20260804/luna_judge_20260805/qwen_c.json`
- `runs/hotpotqa_v2_skill_ablation_20260804/luna_judge_20260805/luna_a.json`
- `runs/hotpotqa_v2_skill_ablation_20260804/luna_judge_20260805/luna_b.json`
- `runs/hotpotqa_v2_skill_ablation_20260804/luna_judge_20260805/luna_c.json`
