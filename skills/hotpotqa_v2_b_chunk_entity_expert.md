# HotpotQA V2 chunk-to-entity expert procedure

Follow this retrieval procedure as one researcher. Return one action per turn,
retain useful evidence, and answer directly with FINISH.

## Start with Chunk retrieval

The first action for every question must be `SEARCH` with:

- `method=BM25`
- `target=CHUNK`
- the complete original question as `query`
- `top_k=5`

Treat returned Chunk previews and C# handles as navigation, not evidence.
Inspect their titles and previews together with the question. Identify the
named subject, plausible bridge entities, and one atomic missing fact.

## Drill down through Entities

For each candidate relevant to the missing fact:

1. Use `LEXICAL -> ENTITY` when its exact name or alias is known.
2. Use `DENSE -> ENTITY` only when the canonical name is uncertain.
3. Select a visible E# that is listed for
   `ENTITY_MENTIONED_IN_SENTENCE` in `allowed_expansions`.
4. EXPAND from that exact E#. Phrase `query` as the missing fact so the local
   neighbourhood ranks relevant Sentences first.

For a bridge question, complete the first relation, then repeat this Entity
procedure for the discovered bridge identity and second relation. For a
comparison, complete the same requested property for both subjects.

## Judge each Sentence

- Add a complete eligible S# to `selected_evidence_refs` when it explicitly
  supports an evidence obligation.
- If retained evidence covers every obligation, FINISH immediately.
- If a relevant Sentence lacks a referent, qualifier, date, list item, or
  nearby relation, READ its visible parent C# once.
- READ only a currently visible unread C#.
- If an Entity path is irrelevant or empty, try a different candidate Entity.
- After exhausting plausible Entities, fall back once to targeted BM25
  Sentence and then once to Dense Sentence for the same atomic fact.

## Prevent invalid loops

Never repeat a SEARCH, EXPAND, or READ from semantic action history. Copy only
currently displayed handles; do not reconstruct or modify E#/S#/C#. If an
action is invalid, change the source, retrieval method, target, or missing fact
instead of retrying it.

FINISH with the shortest complete `action.answer`. Its `evidence_refs` must be
eligible retained S# or already-read C# handles. Preserve all supported roles,
qualifiers, dates, and comparison conditions.
