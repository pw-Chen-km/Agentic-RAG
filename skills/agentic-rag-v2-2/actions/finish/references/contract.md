# FINISH parameter contract

Return exactly:

```json
{
  "answer": "American schoolteacher and publisher",
  "evidence_refs": [
    {"unit": "SENTENCE", "id": "S4"}
  ]
}
```

- `answer` must be nonblank and contain only claims supported by cited evidence.
- `evidence_refs` must be nonempty and unique.
- Every citation must be an exact member of the inherited Stage 1
  `selected_evidence_refs`; citations may be a sufficient subset of that pending
  selection.
