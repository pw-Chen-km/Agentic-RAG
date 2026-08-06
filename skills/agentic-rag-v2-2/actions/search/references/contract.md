# SEARCH parameter contract

Return exactly:

```json
{
  "query": "semantic retrieval query",
  "method": "BM25",
  "target": "SENTENCE",
  "top_k": 5
}
```

- `query` must be nonblank and use real names or semantic text.
- `top_k` must be `5`.
- The supported method-target pairs are:
  - `LEXICAL` + `ENTITY`
  - `BM25` + `SENTENCE`
  - `BM25` + `CHUNK`
  - `DENSE` + `ENTITY`
  - `DENSE` + `SENTENCE`
  - `DENSE` + `CHUNK`

Prefer lexical Entity search for an exact name, BM25 for likely wording, and
dense retrieval for paraphrases.
