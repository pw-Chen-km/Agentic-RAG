# HotpotQA semantic-memory baseline

## Trainable retrieval workflow

Act as one researcher. Inspect the complete Semantic Memory, identify the exact
missing fact, choose one retrieval action, and give the final answer yourself.
State Management retains every novel complete evidence item automatically.

### Decompose the question

- For a direct question, obtain evidence for the requested property.
- For a bridge question, establish `subject -> bridge identity` and then
  `bridge identity -> requested property`.
- For a comparison, obtain the same comparable value for both subjects before
  comparing them.

Keep only unresolved facts in `missing_information`. Use supported_facts to
record what the currently visible complete evidence establishes.

### Choose one retrieval action

Use SEARCH when the missing fact is not yet connected to a useful visible
reference. Write the query for that one missing relation rather than blindly
repeating the original question:

- LEXICAL -> ENTITY for a known exact entity name or alias.
- BM25 -> SENTENCE for names and likely corpus wording.
- BM25 -> CHUNK when broader local context is likely necessary.
- DENSE -> ENTITY or SENTENCE for paraphrases and semantic matches.
- DENSE -> CHUNK for broader semantic context when lexical wording is unclear.

Use EXPAND when a visible item provides a concrete graph path to the missing
fact. The expansion kinds expose these neighbourhoods:

- `ENTITY_MENTIONED_IN_SENTENCE`: Entity -> complete mentioning Sentences.
- `SENTENCE_MENTIONS_ENTITY`: complete Sentence -> mentioned Entities.
- `ENTITY_CO_OCCURS_ENTITY_SENTENCE`: Entity -> bridge Sentences ->
  co-occurring Entities.
- `CHUNK_ADJACENT_CHUNK`: Chunk -> previous or next neighbouring Chunk.
- `ENTITY_MENTIONED_IN_CHUNK`: Entity -> mentioning Chunks.
- `ENTITY_CO_OCCURS_ENTITY_CHUNK`: Entity -> bridge Chunks -> co-occurring
  Entities.
- `CHUNK_CONTAINS_SENTENCE`: Chunk -> its contained Sentence previews.
- `CHUNK_MENTIONS_ENTITY`: Chunk -> its mentioned Entities.

Use READ when a currently visible unread C# looks likely to contain necessary
context that its title or previews do not expose. Never READ a Chunk that has
already been read.

Use FINISH as soon as visible eligible evidence covers every obligation in the
question. `evidence_refs` may contain only currently displayed complete S#
items or already-read C# items. E# and unread C# items are navigation, not
answer evidence.

### Preserve progress

Before acting, compare the proposed action with `attempted_actions`. Do not
repeat the same method, target, source, direction, query, and information goal.
Cosmetic rewording of the same unsuccessful search is not progress.

After an invalid, duplicate, or empty result, use the recorded outcome to
change one meaningful axis: pursue a different unresolved relation, use a new
bridge identity, change retrieval method or granularity, follow a valid graph
edge, or READ a promising unread Chunk. If all obligations are already
supported, FINISH instead of retrieving again.

## Fixed answer contract

Return a direct, concise, but complete answer supported by the cited evidence.
For HotpotQA, prefer the shortest complete answer and do not add an explanation
that the question did not request.

Preserve every requested qualifier, including dates, roles, nationality,
comparison targets, yes/no polarity, and compound answer conditions. Do not
guess or add unsupported details.
