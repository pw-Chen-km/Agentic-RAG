# LinearHop Meta merge

Merge at most the existing bounded number of retrieval-policy edits from the
paired LinearHop diagnoses. Preserve the fixed A-RAG tools and the separate
Direct-BM25 control. Do not change graph construction, entity linking,
question-sentence ranking, max hops, top-k, snippets, or `read_chunk`.
Reject question-specific, answer-specific, or infrastructure edits.
