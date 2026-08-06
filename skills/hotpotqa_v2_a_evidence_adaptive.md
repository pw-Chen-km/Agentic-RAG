# HotpotQA V2 evidence-adaptive strategy

Act as one researcher. Identify the exact missing fact, retrieve it, retain its
evidence, and give the final answer yourself. The Controller handles storage,
handle allocation, duplicate checks, and budgets.

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
the next action from that missing fact, not from which handles happen to be
present.

## Choose the next action

- SEARCH when retained evidence does not yet contain a clear path to the
  missing fact. SEARCH remains available after previous retrievals.
- EXPAND only when one visible handle provides a concrete local bridge to the
  missing fact and appears under the selected kind in `allowed_expansions`.
- READ an unread visible C# when pronouns, lists, qualifiers, nearby dates, or
  surrounding context are missing.
- FINISH only when eligible retained evidence fully supports the answer.

Start with a complete BM25 Sentence question when names or likely wording are
known. Use Dense Sentence for paraphrases. Use Lexical Entity only for an exact
entity name or alias.

After a Sentence reveals a bridge identity, a new targeted SEARCH is often
better than expanding an unrelated handle. Do not let existing handles prevent
a fresh SEARCH.

Use graph expansion only from a currently legal source:

- `SENTENCE_MENTIONS_ENTITY`: eligible S#.
- `ENTITY_MENTIONED_IN_SENTENCE`: E#.
- `ENTITY_CO_OCCURS_ENTITY_SENTENCE`: E# whose neighbourhood may express the
  missing relation.
- `CHUNK_ADJACENT_CHUNK`: C# with PREV, NEXT, or BOTH.

Do not repeat the same SEARCH, EXPAND, or READ. After an invalid action, copy a
currently legal handle from the error details or choose a new targeted SEARCH.

## Retain and finish

Add every newly useful eligible S# or already-read C# to
`selected_evidence_refs`; do not select Entities or unread Chunks. FINISH as
soon as every required fact is covered. Put the shortest complete answer in
`action.answer` and cite the retained evidence in `action.evidence_refs`.

For comparisons, calculate from both values before naming the result. When
evidence states multiple coordinated roles or qualifiers, preserve all of
them. Never FINISH from evidence that states only a topic or identity but not
the requested property.
