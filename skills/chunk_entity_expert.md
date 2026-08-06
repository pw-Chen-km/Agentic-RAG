# HotpotQA chunk-to-entity expert procedure

Return one action per turn and answer directly with FINISH. State Management
retains novel evidence automatically.

The first action must be BM25 CHUNK SEARCH using the complete question and
`top_k=5`. Treat previews as navigation. Identify candidate entities and one
atomic missing fact.

For each promising candidate:

1. SEARCH LEXICAL -> ENTITY when the exact name is known; otherwise use DENSE.
2. EXPAND `ENTITY_MENTIONED_IN_SENTENCE` from the visible E#.
3. Phrase the expansion query as the missing fact.
4. If an S# explicitly satisfies an obligation, keep it. If it needs nearby
   context, READ its visible unread parent C# once.

For bridges, repeat this entity procedure for the newly found identity. For
comparisons, obtain the same property for both subjects. If entity paths fail,
fall back to a targeted BM25 Sentence search, then Dense Sentence search.

Never repeat an action. Use refs only while visible in the current snapshot.
FINISH with the shortest complete answer and only visible S# or read C#
`evidence_refs`.
