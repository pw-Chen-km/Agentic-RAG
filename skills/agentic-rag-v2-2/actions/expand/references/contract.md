# EXPAND parameter contract

Return exactly:

```json
{
  "kind": "ENTITY_MENTIONED_IN_SENTENCE",
  "source_id": "E2",
  "direction": null,
  "query": "sentences describing Moses Harman's occupation",
  "top_k": 5
}
```

- `kind` must be one enabled relation described in `relations.md`.
- `source_id` must be one visible `E#`, `S#`, or `C#` handle of the source type
  required by that relation.
- `direction` must be `null` except for `CHUNK_ADJACENT_CHUNK`, where it must be
  `PREV`, `NEXT`, or `BOTH`.
- `query` is optional. Omit it or use a nonblank semantic phrase; never use a
  handle as the meaning of the query.
- `top_k` must be `5`.
