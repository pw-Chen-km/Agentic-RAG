# Information-gap assessment v2

The assessment is a model-generated working state, not source evidence. It is
returned in the same native tool call as the retrieval action:

```json
{
  "assessment": {
    "resolved_gaps": [],
    "missing_information": [
      "determine the main industry of Mary Town"
    ]
  }
}
```

`resolved_gaps` is cumulative within an episode. `missing_information` is
replaced on each turn with the complete current list. Items should preserve the
exact entity names, relationships, dates, locations, and other qualifiers from
the question and shown source text. For example, use `determine the main
industry of Mary Town`, not `determine the industry`.

When resolving an earlier item, begin with its original wording and append the
newly established value after an em dash. For example, `identify the town where
the festival was held — Mary Town` preserves the link to the earlier gap.

The runtime stores both lists in the trajectory and shows the latest assessment
to the next policy turn. It does not treat either list as evidence and does not
add assessment text to evaluator source spans. There is no fixed three-item
limit; the provider output budget is the operational bound.

The previous `supported_facts`-only contract remains historical and cannot be
used to resume a run with this contract.
