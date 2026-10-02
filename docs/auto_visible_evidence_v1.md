# Auto-visible evidence contract

Retrieval and answer generation use different responsibilities.

During retrieval, the model chooses one available operation and updates the
information-gap assessment. When the episode enters answer stage, the harness
builds one complete source view from all source units that were actually shown
to the model. It removes exact duplicate source units and gives a passage
precedence over its sentence children.

The `finish` tool therefore requires only:

```json
{"answer": "..."}
```

The model no longer has to select a citation subset. The episode artifact saves:

- `visible_source_refs`: the deduplicated source units shown in answer stage;
- `evidence_refs`: the same programmatically collected references for backward
  artifact compatibility;
- `evidence_refs_source`: `programmatic_visible_source_refs`.

Entity references are never evidence references. A source reference is eligible
only when its complete text is present in the answer-stage Policy input. This
field measures evidence acquisition; it must not be interpreted as a model
citation-selection metric. Citation selection is a separate capability and is
not evaluated by the new contract.
