# HotpotQA V3 Skill Variation Test-6

Date: 2026-08-04

## Fixed evaluation contract

- Dataset: unchanged HotpotQA Test-6
- Agent policy and answer model: Ollama `qwen3.5:9b`
- OpenAI/GPT calls: none (`gpt_or_openai_used=false` in every run)
- Retrieval substrate: `artifacts/hotpotqa_benchmark_exact`
- Config: `configs/agentic_hotpotqa_qwen35_9b_v3.yaml`
- Architecture: Single-Agent V3 semantic memory with separate memory and citation indices
- Budgets, embedding model, substrate, and decoding settings were held constant.
- The expanded Train-20 split was not used during this Test-6 inference comparison.

## Variations

- A, baseline clean: current V3 procedure with the Test-6-specific answer phrase removed.
- B, chunk/entity expert: always starts with BM25 Chunk, then uses Entity and Sentence navigation, reading a Chunk only when a promising Sentence lacks context.
- C, post-hoc guarded: uses Evidence, Progress, Novelty, and Reference gates derived from prior Test-6 failures. This is deliberately test-informed and must not be reported as held-out generalization.

## Aggregate results

| Skill | Contain | Exact | Evidence-consistent correct | Invalid attempts | Context-index errors | Policy calls | Input tokens | Output tokens | Total tokens | Retrieved tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A baseline clean | 4/6 | 0/6 | 2/6 | 19 | 13 | 41 | 104,577 | 6,114 | 110,691 | 7,648 |
| B chunk/entity expert | 1/6 | 0/6 | 0/6 | 44 | 42 | 65 | 203,372 | 8,579 | 211,951 | 20,461 |
| C post-hoc guarded | 0/6 | 0/6 | 1/6 | 38 | 20 | 54 | 142,221 | 10,356 | 152,577 | 9,301 |

Evidence-consistent correctness treats contradictory answers as incorrect even if the gold string is present. It accepts a concise or verbose answer when its meaning is fully correct, and resolves the known Test-6 item-4 gold/evidence inconsistency in favour of its supporting evidence.

## Action and error distribution

| Skill | SEARCH | EXPAND | READ | FINISH | Error distribution |
|---|---:|---:|---:|---:|---|
| A | 20 | 5 | 11 | 5 | `memory_node_type_mismatch=11`, `duplicate_action=6`, `chunk_not_readable=2` |
| B | 13 | 20 | 23 | 9 | `memory_node_type_mismatch=20`, `chunk_not_readable=14`, `citation_index_out_of_range=8`, `duplicate_action=2` |
| C | 27 | 13 | 11 | 3 | `duplicate_action=18`, `memory_node_type_mismatch=17`, `chunk_not_readable=3` |

B followed its defining first-step instruction in all 6/6 questions: every first action was `SEARCH BM25 -> CHUNK`. Its poor result therefore occurred after the initial search, not because the Skill failed to load.

## Per-question semantic audit

| Gold answer | A | B | C |
|---|---|---|---|
| more than 330 million people | Empty | Empty | Empty |
| American schoolteacher and publisher | `schoolteacher` only; incomplete | Empty | `schoolteacher` only; incomplete |
| Matt Groening | Correct but verbose | Empty | `David X. Cohen`; stopped at the bridge |
| Gold: Livin' la Vida Loca; evidence: Livin La Vida Loco | Adds the unsupported suffix `World Tour` | Empty | Evidence-consistent `Livin La Vida Loco` |
| Midnight Madness | Contains the gold string but contradicts the dates | Contains the gold string but contradicts the dates | Empty |
| Tori Amos | Correct but verbose | Empty | Empty |

## Interpretation

1. A is the strongest Ollama variation. Its 0/6 Exact is partly an answer-contract problem: correct answers are often wrapped in full sentences. Its 4/6 Contain overstates true correctness because the movie comparison is self-contradictory and the tour answer adds an unsupported title suffix.
2. B's rigid Chunk-first workflow is actively harmful for the Ollama 9B model and budget. Chunk results enlarge context, after which the policy repeatedly READs already read or non-readable chunks, applies Entity expansions to the wrong node type, and cites invalid indices. It costs 91% more total tokens than A while reducing evidence-consistent correctness from 2/6 to 0/6.
3. C's textual gates do not provide reliable control. The model can recite the recovery rule but still repeats the same invalid EXPAND or SEARCH. It also finishes too early at a bridge entity or incomplete compound occupation.
4. The common bottleneck is not the absence of strategy prose. It is that the 9B policy must both reason about the question and perform exact protocol bookkeeping. Natural-language instructions alone do not deterministically enforce node-type compatibility, novelty, or complete multi-hop evidence.
5. The next experiment should keep A's flexible retrieval strategy and move only mechanical recovery into the runtime: after a type mismatch, suppress that exact invalid action for the current snapshot and expose a fresh valid decision opportunity without spending another environment step. Answer evaluation should also add semantic/consistency checks alongside Contain and Exact.

## Luna V3 results

The same three Skills were rerun with `gpt-5.6-luna`. Dataset, substrate,
embedding model, V3 architecture, and budgets were unchanged.

| Skill | Official Contain | Official Exact | Evidence-consistent correct | Invalid attempts | Policy calls | Input tokens | Output tokens | Total tokens | Retrieved tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A baseline clean | 4/6 | 1/6 | 5/6 | 0 | 26 | 90,902 | 4,605 | 95,507 | 6,371 |
| B chunk/entity expert | 5/6 | 3/6 | **6/6** | 1 | 25 | 80,302 | 3,508 | **83,810** | 16,206 |
| C post-hoc guarded | 4/6 | 3/6 | 5/6 | 1 | 25 | 85,430 | 4,317 | 89,747 | 6,027 |

### Luna action and error distribution

| Skill | SEARCH | EXPAND | READ | FINISH | Error distribution |
|---|---:|---:|---:|---:|---|
| A | 18 | 0 | 2 | 6 | none |
| B | 11 | 1 | 7 | 6 | `memory_node_type_mismatch=1` |
| C | 14 | 3 | 2 | 6 | `duplicate_action=1` |

B followed `SEARCH BM25 -> CHUNK` on the first action for all 6/6 questions.
Unlike the Ollama run, Luna then completed the procedure with only one invalid
attempt across the full set.

### Ollama versus Luna

| Skill | Ollama evidence-correct | Luna evidence-correct | Ollama invalid | Luna invalid | Ollama tokens | Luna tokens |
|---|---:|---:|---:|---:|---:|---:|
| A | 2/6 | 5/6 | 19 | 0 | 110,691 | 95,507 |
| B | 0/6 | **6/6** | 44 | 1 | 211,951 | **83,810** |
| C | 1/6 | 5/6 | 38 | 1 | 152,577 | 89,747 |

The model choice changes the conclusion about B. Chunk-first is not inherently
bad: Luna executes it efficiently and obtains every evidence-supported answer.
The Ollama 9B policy fails mainly at protocol bookkeeping after the initial
Chunk search. C is unnecessary for Luna and costs more than B without improving
accuracy.

### Test-6 item-4 annotation defect

The official answer is `Livin' la Vida Loca`, the Ricky Martin song. The source
record's supporting evidence says the Coal Chamber tour is `Livin La Vida
Loco`, a play on that song title. Since the question asks for the concert tour,
`Loco` is evidence-consistent. Official Contain and Exact nevertheless score it
as wrong. This accounts for Luna B showing official 5/6 but evidence-consistent
6/6.

## Artifacts

- `skills/hotpotqa_v3_a_baseline_clean.md`
- `skills/hotpotqa_v3_b_chunk_entity_expert.md`
- `skills/hotpotqa_v3_c_posthoc_guarded.md`
- `runs/hotpotqa_v3_skill_variations_20260804/a_baseline_clean/summary.json`
- `runs/hotpotqa_v3_skill_variations_20260804/b_chunk_entity_expert/summary.json`
- `runs/hotpotqa_v3_skill_variations_20260804/c_posthoc_guarded/summary.json`
- `runs/hotpotqa_v3_skill_variations_luna_20260804/a_baseline_clean/summary.json`
- `runs/hotpotqa_v3_skill_variations_luna_20260804/b_chunk_entity_expert/summary.json`
- `runs/hotpotqa_v3_skill_variations_luna_20260804/c_posthoc_guarded/summary.json`
