# LinearHop Meta merge

Merge at most the existing bounded number of retrieval-policy edits from the
paired LinearHop diagnoses. Use the enabled actions in the supplied RUNTIME
ACTION SPACE section and preserve their tool contracts. Do not change graph construction, entity linking,
question-sentence ranking, max hops, top-k, snippets, or `read_chunk`.
Reject question-specific, answer-specific, or infrastructure edits.
