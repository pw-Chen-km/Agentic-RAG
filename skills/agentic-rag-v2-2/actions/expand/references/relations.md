# Enabled EXPAND relations

Only these four relations are enabled in V2-2:

- `ENTITY_MENTIONED_IN_SENTENCE`: visible Entity -> complete Sentences that
  mention it. Use an `E#` source and `direction=null`.
- `SENTENCE_MENTIONS_ENTITY`: visible Sentence -> Entities mentioned in it. Use
  an `S#` source and `direction=null`.
- `ENTITY_CO_OCCURS_ENTITY_SENTENCE`: visible Entity -> bridge Sentence ->
  co-occurring Entities. Use an `E#` source and `direction=null`.
- `CHUNK_ADJACENT_CHUNK`: visible Chunk -> adjacent Chunk in the same document.
  Use a `C#` source and `direction=PREV`, `NEXT`, or `BOTH`.

Entity results are navigation nodes rather than answer evidence. Complete
Sentence results may be eligible evidence. Returned Chunk previews require READ
before the Chunk can be cited.
