# HotpotQA multi-substrate retrieval policy

Build a complete evidence chain before finishing. At every turn, identify the
one fact that is still missing, preserve evidence for facts already established,
and choose one legal action that is most likely to establish that missing fact.

## Interpret the substrate correctly

Treat every handle as episode-local and copy it exactly from the current policy
state. Never invent a handle or reconstruct a database ID.

| Level | Handle | Meaning | Evidence status |
| --- | --- | --- | --- |
| Entity | `E#` | A canonical identity or alias cluster and a graph navigation anchor. An Entity does not assert a fact. | Never evidence. |
| Sentence | `S#` | An atomic textual proposition inside one Chunk. | Evidence only when `can_use_as_evidence=true`; a preview is not evidence. |
| Chunk | `C#` | An ordered context window inside one Document. Search and expansion normally expose only previews. | Evidence only after `READ`, when `has_been_read=true`. |
| Document | no policy handle | The implicit container for ordered Chunks and their title. | Navigate through parent Chunk handles and Chunk adjacency. |

Entity-Sentence mention links are structural candidate paths, not proof of the
relationship asked by the question. Co-occurrence means only that two entities
occur in the same Sentence; read the bridge Sentence text before treating any
relationship as established.

## Maintain an evidence-obligation ledger

Before every action, decompose the question into required facts.

- For a bridge question, require both `question subject -> bridge identity` and
  `bridge identity -> requested attribute`.
- For a comparison, require the comparable value for each subject and then the
  comparison implied by those values.
- For a conjunctive or multi-role question, require evidence for every requested
  role, qualifier, date, or named item.
- For a direct question, require a Sentence or READ Chunk that explicitly states
  the requested fact, not merely the topic or work identity.

Track which eligible evidence ref supports each obligation. The current
`selected_evidence_refs` replaces the complete working set; therefore carry
forward every earlier ref that still supports a required fact. Never discard a
correct first-hop ref merely because the next action targets another fact.

## Choose and execute actions

### `SEARCH`

Use `SEARCH` to enter a new part of the substrate or to retrieve a missing fact
that local graph traversal cannot directly provide. Keep `top_k=5`.

- Use `LEXICAL -> ENTITY` only with an explicit entity name or alias. It is an
  exact normalized alias lookup, not a free-text search.
- Use `BM25 -> SENTENCE` for a direct fact when names and likely corpus wording
  are known. A returned complete Sentence can be evidence immediately.
- Use `DENSE -> SENTENCE` when the fact is likely paraphrased or BM25 wording has
  failed.
- Use `BM25 -> CHUNK` or `DENSE -> CHUNK` only when a list, chronology, pronoun,
  qualifier, or surrounding context is likely to span several Sentences. The
  returned Chunk and previews remain navigation-only until `READ`.
- Use `DENSE -> ENTITY` only when an entity is conceptually described but its
  exact alias is unknown.

Start with the original complete natural-language question. After discovering a
bridge entity, ask a complete subquestion that names that entity and requests the
still-missing attribute. Do not repeat the original query when it has already
established only the first hop, and do not reduce BM25 or Dense queries to
disconnected keywords.

### `EXPAND`

Use `EXPAND` only for a local structural hop. Copy `source_id` from
`policy_state.allowed_expansions[kind]`; this mapping is the authoritative list
of legal sources. Set `query` to a complete missing-fact subquestion to rank the
local neighbours, or `null` to rank with the original question. Keep `top_k=5`.

The current HotpotQA profile enables exactly these paths:

| Kind | Legal source | Result and use |
| --- | --- | --- |
| `ENTITY_MENTIONED_IN_SENTENCE` | `E#` | Return complete Sentences that mention the Entity. Use this for `Entity -> facts about that entity`; returned Sentences are eligible evidence. |
| `SENTENCE_MENTIONS_ENTITY` | complete eligible `S#` | Return Entities named in that Sentence. Use this to extract a specific bridge identity; returned Entities are navigation-only. |
| `ENTITY_CO_OCCURS_ENTITY_SENTENCE` | `E#` | Traverse `Entity -> bridge Sentence -> other Entity`. The bridge Sentence may be evidence, while the target Entity remains navigation-only. Verify the text actually expresses the needed relation. |
| `CHUNK_ADJACENT_CHUNK` | `C#` | Return the previous, next, or both neighbouring Chunks in the same Document. Set `direction` to `PREV`, `NEXT`, or `BOTH`; results remain unread navigation-only Chunks. |

For the first three kinds set `direction=null`. Do not swap the inverse
Sentence/Entity paths: `ENTITY_MENTIONED_IN_SENTENCE` requires `E#`, while
`SENTENCE_MENTIONS_ENTITY` requires `S#`. Do not rely on controller repair to
correct a semantically wrong expansion.

When a Sentence contains several Entities, choose the one that satisfies the
first-hop relation in the text. Do not expand every visible Entity or abandon a
bridge already established by the Sentence.

### `READ`

Use `READ` with one visible `C#` when complete local context is needed. Prefer the
parent Chunk of a useful Sentence or preview. READ when:

- a pronoun or reference is ambiguous;
- a preview omits the needed proposition;
- a list, date, qualifier, or second supporting Sentence may be nearby;
- comparison evidence requires exact values or chronology.

After READ, all contained Sentences become complete and eligible and the Chunk
itself becomes eligible. Do not READ the same Chunk again. Do not READ an
unrelated Chunk merely because it is visible.

### `FINISH`

Use `FINISH` only when `assessment.status=SUFFICIENT`. Submit 1-20 eligible
Sentence or READ Chunk refs, and include every submitted ref in this turn's
`selected_evidence_refs`. Entities, previews, and unread Chunks can never be
submitted.

Before FINISH, run this coverage check:

1. Map every evidence obligation to explicit text in an eligible ref.
2. Confirm both hops of a bridge or both sides of a comparison are present.
3. Confirm the evidence states the requested property, not only a related title,
   episode, person, or topic.
4. For dates or numeric comparisons, retain the values for both subjects and
   verify which value satisfies the question.
5. Keep the smallest evidence set that still covers every obligation.

If any obligation lacks support, keep the assessment `INSUFFICIENT` or
`UNCERTAIN`, name the exact missing fact, and retrieve it instead of finishing.

## Recover without wasting attempts

- Inspect `attempted_actions` before acting. Never repeat an identical action.
- After irrelevant or empty results, change the query, retrieval method, target,
  or bridge entity according to the missing fact.
- After a duplicate or invalid action, copy a legal handle from
  `allowed_expansions` or choose a new SEARCH; do not retry the same payload.
- Prefer a targeted missing-fact SEARCH over speculative expansion when the
  current graph neighbourhood does not contain the required attribute.
- Spend the final available step on completing a known evidence chain or
  FINISHing sufficient evidence, not on a low-confidence detour.
