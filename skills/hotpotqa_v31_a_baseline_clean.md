# HotpotQA semantic-memory typed-reference strategy

Act as one researcher. Inspect the complete semantic memory, identify the exact
missing fact, choose one retrieval action, and give the final answer yourself.
State Management retains every novel complete evidence item automatically.

## Plan the evidence chain

- Direct question: obtain one sentence that explicitly states the requested
  property.
- Bridge question: establish both `subject -> bridge identity` and
  `bridge identity -> requested property`.
- Comparison: obtain the comparable value for both named subjects and compare
  the values before answering.
- Compound answer: preserve every role, qualifier, nationality, date, or item
  explicitly supported by the evidence.

At each turn, write the exact remaining fact in `missing_information`. Judge
the next action from that missing fact, not from which memory items happen to
be present.

## Choose the next action

- SEARCH when semantic memory does not yet contain a clear path to the missing
  fact. SEARCH remains available after any number of previous retrievals.
- EXPAND only when one known entity, sentence, or chunk provides a concrete
  local bridge to the missing fact.
- READ an unread parent chunk when pronouns, lists, qualifiers, nearby dates,
  or surrounding context are missing.
- FINISH only when the displayed evidence fully supports the answer.

Start with a complete BM25 Sentence question when names or likely wording are
known. Use Dense Sentence search for paraphrases. Use Lexical Entity only for
an exact entity name or alias.

After a sentence reveals a bridge identity, a new targeted SEARCH is often
better than expanding unrelated nodes. Do not let the presence of existing
memory items prevent a fresh SEARCH.

Use graph expansion only when its source expresses a relevant connection:

- `SENTENCE_MENTIONS_ENTITY`: complete S# -> mentioned Entities.
- `ENTITY_MENTIONED_IN_SENTENCE`: E# -> complete mentioning Sentences.
- `ENTITY_CO_OCCURS_ENTITY_SENTENCE`: E# -> bridge Sentence -> co-occurring
  Entities.
- `CHUNK_ADJACENT_CHUNK`: C# -> previous or next Chunk.

Do not repeat the same SEARCH, EXPAND, or READ. After an invalid action, inspect
the current semantic memory and either choose a different legal source or issue
a new targeted SEARCH.

## Finish with typed evidence refs

FINISH as soon as every required fact is covered. Include the shortest complete
answer and copy only currently displayed S# refs or already-read C# refs into
`evidence_refs`. E# and unread C# refs are navigation only and never evidence.

For comparisons, calculate from both values before naming the result. When
evidence states multiple coordinated roles or qualifiers, preserve all of them
instead of returning only one. Never FINISH using evidence that states only the
topic or identity but not the requested property.
