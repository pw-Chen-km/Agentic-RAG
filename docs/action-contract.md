# Action contract

Every Policy turn derives one immutable `AvailableActionSpace` from the
current state and frozen typed-reference map. The same action space is rendered
as optional prompt guidance and compiled into the provider response schema.
Action families with no legal structural variant are omitted from that turn's
schema. Set `agent.use_state_conditioned_schema: false` to retain the legacy
static schema for ablations.

## SEARCH

```json
{
  "assessment": {
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

SEARCH remains available while retrieval budgets are open. Its six legal
method/target pairs are separate schema variants. Query text stays free-form,
so exact duplicate SEARCH remains a post-generation Validator rule.

## EXPAND

```json
{
  "assessment": {
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

Only enabled relationships with a compatible visible source are emitted.
`source_ref` is an enum of the compatible refs for that relationship. The
resolver still trims whitespace and normalizes case only; it never guesses a
number, fuzzy-matches text, or repairs a graph handle.

## READ

```json
{
  "assessment": {
    "supported_facts": [],
    "missing_information": ["The sentence's missing qualifier"]
  },
  "action": {"type": "READ", "chunk_ref": "C1"}
}
```

`chunk_ref` is an enum of visible, in-scope, unread Chunks. READ is omitted when
that enum would be empty.

## FINISH

```json
{
  "assessment": {
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

Only visible complete S# items and visible read C# items are emitted as allowed
evidence values. E# and unread C# are navigation only. When retrieval closes,
budget finalization uses a FINISH-only action space and schema.

## Invalid attempts

The schema prevents most structural interface errors before parsing. The
Resolver and Validator remain defense-in-depth. Their errors include `reference_not_available`,
`reference_type_mismatch`, `reference_not_evidence`, `chunk_not_readable`,
`expansion_not_valid_for_node`, and `duplicate_action`. An invalid attempt is
recorded with the raw decision and frozen map. It consumes one Policy attempt
and its LLM tokens, but no retrieval step or retrieved-token budget.
