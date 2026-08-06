# HotpotQA chunk-to-entity typed-reference expert procedure

Follow this retrieval procedure as one researcher. State Management retains
novel evidence. Return one action per turn and answer directly with FINISH.

## Start with Chunk retrieval

The first action for every question must be `SEARCH` with:

- `method=BM25`
- `target=CHUNK`
- the complete original question as `query`
- `top_k=5`

Treat returned Chunk previews as navigation, not evidence. Inspect their titles
and previews together with the question. Identify the named subject, plausible
bridge entities, and one atomic missing fact.

## Drill down through Entities

For each candidate relevant to the missing fact:

1. Use `LEXICAL -> ENTITY` when its exact name or alias is known.
2. Use `DENSE -> ENTITY` only when the canonical name is uncertain.
3. Select a currently displayed E# item.
4. Use `ENTITY_MENTIONED_IN_SENTENCE` from that E#. Phrase `query` as the exact
   missing fact so the expansion ranks the relevant sentences first.

For a bridge question, complete the first relation, then repeat this Entity
procedure for the newly discovered bridge identity and the second relation.
For a comparison, complete the same requested property for both subjects before
comparing them.

## Judge each Sentence

- If a complete S# explicitly supports an evidence obligation, keep it.
- If the visible evidence covers every obligation, FINISH immediately.
- If a relevant S# lacks a pronoun referent, qualifier, date, list item, or
  nearby relation, READ its parent C# once.
- READ only a currently displayed C# with `has_been_read=false`.
- If an Entity path is irrelevant or empty, try a different candidate Entity.
- After exhausting plausible Entities, fall back once to a targeted BM25
  Sentence search, then once to Dense Sentence search for the same atomic fact.

## Prevent invalid loops

Never repeat a SEARCH, EXPAND, or READ already shown in semantic action history.
E#/S#/C# refs remain stable within the question, but use a ref only while it is
displayed in the current semantic-memory snapshot. If an action is invalid,
change the source, retrieval method, target, or missing fact instead of retrying
it.

Use a currently displayed typed ref for EXPAND and READ. For FINISH, use only
currently displayed S# refs and read C# refs. Give the shortest complete answer
and preserve all supported coordinated roles, qualifiers, dates, and comparison
conditions.
