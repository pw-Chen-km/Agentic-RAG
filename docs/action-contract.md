# Action contract

The provider exposes all four action families on every Policy call. Existing
memory never removes SEARCH from the schema.

## SEARCH

```json
{
  "assessment": {
    "status": "INSUFFICIENT",
    "supported_facts": [],
    "missing_information": ["Peter Daou's relationship to Verrit"]
  },
  "action": {
    "type": "SEARCH",
    "query": "Peter Daou Verrit created",
    "method": "BM25",
    "target": "SENTENCE",
    "top_k": 5
  }
}
```

SEARCH is unrestricted by existing refs. Legal methods are BM25, DENSE, and
LEXICAL where supported by the target; legal targets are ENTITY, SENTENCE, and
CHUNK according to the retriever contract.

## EXPAND

```json
{
  "assessment": {
    "status": "INSUFFICIENT",
    "supported_facts": ["Peter Daou created Verrit."],
    "missing_information": ["The requested property of Verrit"]
  },
  "action": {
    "type": "EXPAND",
    "kind": "ENTITY_MENTIONED_IN_SENTENCE",
    "source_ref": "E1",
    "query": "requested property of Verrit",
    "direction": null,
    "top_k": 5
  }
}
```

The resolver trims whitespace and normalizes case only. It never guesses a
number, fuzzy-matches text, or repairs a graph handle.

## READ

```json
{
  "assessment": {
    "status": "INSUFFICIENT",
    "supported_facts": [],
    "missing_information": ["The sentence's missing qualifier"]
  },
  "action": {"type": "READ", "chunk_ref": "C1"}
}
```

The chunk must be visible, in scope, and unread.

## FINISH

```json
{
  "assessment": {
    "status": "SUFFICIENT",
    "supported_facts": ["Peter Daou created Verrit."],
    "missing_information": []
  },
  "action": {
    "type": "FINISH",
    "answer": "Peter Daou",
    "evidence_refs": ["S1"]
  }
}
```

Only visible complete S# items and visible read C# items are evidence. E# and
unread C# are navigation only.

## Invalid attempts

Interface errors include `reference_not_available`,
`reference_type_mismatch`, `reference_not_evidence`, `chunk_not_readable`,
`expansion_not_valid_for_node`, and `duplicate_action`. An invalid attempt is
recorded with the raw decision and frozen map. It consumes one Policy attempt
and its LLM tokens, but no retrieval step or retrieved-token budget.
