# READ parameter contract

Return exactly:

```json
{
  "chunk_id": "C1"
}
```

`chunk_id` must be a visible `C#` handle for a Chunk that has not already been
read. Use the Chunk title and previews to choose it; do not invent a handle.
