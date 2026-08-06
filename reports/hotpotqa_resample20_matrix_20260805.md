# HotpotQA Resample-20：V2/V3 × Skill A/B/C × Qwen/Luna

## Experiment contract

- Dataset: 20 newly sampled HotpotQA questions, excluding the earlier Train-20, Val-6, and Test-6 sets.
- Sample seed: `20260805`.
- Sample composition: 16 bridge questions and 4 comparison questions.
- Sample SHA-256: `57fd2bf9871cd7fd310f5937810caa715539f55f7ffc8134103d0755865fec75`.
- Shared substrate: `artifacts/hotpotqa_benchmark_exact`.
- Models: Ollama `qwen3.5:9b` and OpenAI `gpt-5.6-luna`.
- Architectures: Single-Agent V2 and Single-Agent V3.
- Skills: A (adaptive baseline), B (chunk-first expert procedure), C (post-hoc guarded procedure).
- Total episodes: 2 models × 2 architectures × 3 skills × 20 questions = 240.
- Semantic answer judge: `gpt-5.6-luna`, comparing saved generated answers against gold answers. Blank answers are automatically incorrect.

## Results

| Model | Arch | Skill | Luna judge | Contain | Exact | Invalid | Policy calls | Total tokens | Retrieved tokens | SEARCH | EXPAND | READ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Luna | V2 | A | 17/20 | 16/20 | 8/20 | 0 | 113 | 483,161 | 30,079 | 78 | 6 | 9 |
| Luna | V2 | B | 18/20 | 16/20 | 4/20 | 7 | 102 | 491,030 | 47,280 | 45 | 11 | 22 |
| Luna | V2 | C | 17/20 | 17/20 | 8/20 | 0 | 86 | 333,616 | 25,013 | 54 | 4 | 8 |
| Luna | V3 | A | 17/20 | 17/20 | 8/20 | 0 | 81 | 274,435 | 24,035 | 53 | 0 | 8 |
| Luna | V3 | B | 18/20 | 16/20 | 9/20 | 1 | 98 | 356,430 | 58,534 | 47 | 6 | 25 |
| Luna | V3 | C | **19/20** | 17/20 | **10/20** | 0 | 93 | 339,476 | 32,170 | 61 | 1 | 11 |
| Qwen | V2 | A | 5/20 | 3/20 | 1/20 | 109 | 202 | 741,957 | 32,178 | 143 | 26 | 18 |
| Qwen | V2 | B | 5/20 | 5/20 | 0/20 | 113 | 199 | 882,034 | 45,749 | 63 | 92 | 33 |
| Qwen | V2 | C | 4/20 | 4/20 | 0/20 | 103 | 196 | 703,781 | 27,251 | 104 | 75 | 11 |
| Qwen | V3 | A | **6/20** | **6/20** | 2/20 | 137 | 194 | 522,875 | 32,569 | 116 | 10 | 62 |
| Qwen | V3 | B | 3/20 | 2/20 | 0/20 | 174 | 228 | 677,806 | 43,456 | 63 | 59 | 89 |
| Qwen | V3 | C | 4/20 | 4/20 | **3/20** | 143 | 197 | 526,037 | 23,149 | 144 | 20 | 28 |

### A-RAG reference runs

The same Resample-20 was also run through the upstream A-RAG workflow with its
native append-only tool history and 10-loop/128k-token limits. Both runs use
the same 1,311-chunk corpus and `all-MiniLM-L6-v2` sentence index.

| Model | Luna judge | Contain | Exact | Blank | LLM calls | LLM tokens | Retrieved tokens | Keyword | Semantic | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A-RAG + Qwen | 6/20 | 6/20 | 0/20 | 13 | 81 | 217,831 | 53,854 | 29 | 6 | 26 |
| A-RAG + Luna | **19/20** | 17/20 | 0/20 | 0 | 75 | **173,408** | 48,634 | 16 | 19 | 23 |

A-RAG + Luna ties Luna V3-C for the best semantic answer accuracy (19/20),
while reporting 166,068 fewer LLM tokens and 18 fewer calls. This token
comparison is informative but not perfectly apples-to-apples: A-RAG and V3
serialize context differently and obtain usage through different provider
adapters.

A-RAG + Qwen reaches 6/20, tying Qwen V3-A. Its main failure is different:
13 episodes terminate with an empty answer after tool use, rather than being
rejected by a controller validator. Thus A-RAG removes structured-action
validation failures but does not guarantee reliable termination for the 9B
model.

### V2-2 progressive-skill reference runs

V2-2 uses its existing `agentic-rag-v2-2` skill bundle. Each action cycle first
selects an action family, intent, and pending evidence. Only then is the chosen
SEARCH, EXPAND, READ, or FINISH skill disclosed for parameter construction. A
third model call is used only for repair.

| Model | Luna judge | Contain | Exact | Invalid | Policy calls | Total tokens | Retrieved tokens | SEARCH | EXPAND | READ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2-2 + Qwen | **10/20** | 7/20 | 2/20 | 54 | 270 | 620,359 | 8,595 | 54 | 24 | 1 |
| V2-2 + Luna | 16/20 | 16/20 | 6/20 | **0** | 196 | 596,735 | 22,106 | 48 | 23 | 7 |

V2-2 is the strongest Qwen configuration by semantic answer accuracy: 10/20,
compared with 6/20 for both V3-A and A-RAG. Relative to V3-A, it adds 4 correct
answers while using 76 more calls and 97,484 more tokens. Its progressive
contract reduces the undifferentiated action problem, but does not make Qwen
cheap or fully reliable.

The Qwen stage audit records 134 action cycles, 42 initially invalid parameter
drafts, 41 repair attempts, 27 successful repairs, 51 reselections, and 15
unresolved invalid cycles. Luna records 98 action cycles with no invalid draft,
repair, or reselection. This shows that the bundle is structurally clear to
Luna but still difficult for Qwen at the parameter stage.

For Luna, V2-2 is not competitive: it scores 16/20 versus 19/20 for V3-C and
A-RAG, while consuming 596,735 tokens. Its four judge failures are the band
comparison, the Motörhead comparison target, the `little hairs` translation,
and the football-manager birth date. All four are legal completed trajectories,
so they are retrieval/evidence-sufficiency errors rather than validator errors.

## Main findings

### 1. Luna benefits from V3, especially with Skill C

Across the three skills, Luna V2 obtains 52/60 judge-correct answers and Luna V3 obtains 54/60. More importantly, V3 uses 970,341 total tokens versus V2's 1,307,807, a 25.8% reduction. Policy calls fall from 301 to 272.

The best single configuration is Luna V3-C at 19/20 judge-correct. Skill C also gives the best overall Luna efficiency: its V2+V3 total is 673,092 tokens, versus 757,596 for A and 847,460 for B.

### 2. V3 does not solve Qwen's controller-policy interface

Across the three skills, Qwen V2 obtains 14/60 judge-correct answers and Qwen V3 obtains 13/60. V3 reduces tokens from 2,327,772 to 1,726,718 (25.8%), but policy calls increase from 597 to 619 and invalid attempts increase from 325 to 454.

Therefore V3 is a context/token-efficiency improvement for Qwen, not an accuracy improvement in its current form. Qwen V3-A is the best Qwen configuration at 6/20 judge-correct, but the absolute result remains poor.

The A-RAG reference clarifies the interpretation: A-RAG + Qwen is also 6/20.
Its simpler tool protocol avoids V3's invalid-index counts, but 13 blank final
answers replace those failures. Qwen's problem is therefore broader than the
V2/V3 validator alone; it includes tool-loop termination and deciding when to
synthesize an answer.

### 3. The failure type changes from handles to index/type use

Qwen V2 invalid errors:

- `duplicate_action`: 250
- `unknown_handle`: 62
- `finish_evidence_not_selected`: 11
- `source_not_complete`: 2

Qwen V3 invalid errors:

- `duplicate_action`: 213
- `chunk_not_readable`: 111
- `memory_node_type_mismatch`: 96
- `memory_index_out_of_range`: 20
- `citation_index_out_of_range`: 14

V3 successfully removes `unknown_handle` and the S#/C#/E# handle contract. However, Qwen still fails to interpret which semantic-memory item is readable or valid for an expansion, and it repeatedly retries identical searches. The remaining issue is not evidence semantics alone; it is the action-reference affordance and recovery behavior.

### 4. Skill effects depend strongly on the model

- Luna: Skill C is the strongest overall choice. B can be accurate but reads more chunks and uses more retrieval tokens.
- Qwen: Skill A is best. B is harmful, especially under V3 (3/20 judge-correct, 174 invalid attempts).
- The rigid chunk-first procedure in B increases READ pressure. Qwen V3-B attempts 89 READ actions and produces 60 `chunk_not_readable` errors. The skill and interface jointly amplify failure.

This means skill quality should not be reported as one model-independent ranking. The experiment shows a model × architecture × skill interaction.

V2-2 strengthens this conclusion. Progressive action-specific skills help
Qwen substantially, but hurt Luna relative to its simpler V3-C and A-RAG
workflows. More decomposition is therefore not universally better; its value
depends on whether the base model needs extra action scaffolding enough to
justify duplicated context and calls.

### 5. Exact and Contain are diagnostic, not sufficient correctness measures

Exact undercounts semantically correct answers with added context, punctuation, aliases, or shortened location names. Contain can also be wrong for comparison questions because an answer may mention both candidate names while selecting the wrong one. Luna judge is therefore the primary final-answer correctness metric for this matrix, with Exact and Contain retained as deterministic diagnostics.

The judge still evaluates only generated answer versus gold answer. It does not grade retrieval evidence quality, citation entailment, or whether the trajectory reached the answer for the right reason.

## Recommended next intervention

Do not redesign semantic memory again yet. The highest-value next change is a small deterministic recovery layer for Qwen:

1. On `duplicate_action`, block that exact structured action in the next schema/prompt and require one of: changed query, changed method/target, a legal expansion/read, or FINISH.
2. Render explicit capabilities on each memory item itself, for example `Can expand: ...` or `Readable: yes/no`, without adding a separate Action Targets section or hiding SEARCH.
3. Constrain READ and EXPAND indices dynamically to the valid current indices in the structured schema where the provider supports enums.
4. Do not charge an environment step for invalid actions, but add a small repeated-invalid limit so ten identical policy calls cannot consume the entire episode.
5. Re-run the same fixed 20 questions before drawing another sample. This isolates the recovery/interface intervention from sampling variance.

These changes preserve the research premise: the LLM still decides whether to SEARCH, EXPAND, READ, or FINISH and still learns retrieval policy from the skill. The controller only makes selected actions executable and prevents mechanically identical failures from looping.

## Artifacts

- Sample: `data/evaluations/hotpotqa_resample20_seed20260805/questions.jsonl`
- Sample manifest: `data/evaluations/hotpotqa_resample20_seed20260805/manifest.json`
- Run summaries and trajectories: `runs/hotpotqa_resample20_seed20260805/`
- Luna judge outputs: `runs/hotpotqa_resample20_seed20260805/luna_judge_20260805/`
- A-RAG summaries and judge outputs: `runs/hotpotqa_resample20_seed20260805/arag/`
- A-RAG raw trajectories: `A-RAG/results/hotpotqa_resample20_seed20260805/`
- V2-2 summaries, trajectories, and judge outputs: `runs/hotpotqa_resample20_seed20260805/v2_2/`
- Matrix configuration: `configs/hotpotqa_resample20_matrix_20260805.yaml`
- Resumable runner: `scripts/run_hotpotqa_resample20_matrix.ps1`
