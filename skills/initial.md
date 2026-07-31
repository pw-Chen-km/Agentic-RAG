# Initial retrieval strategy

Assess what the current evidence establishes and what information is still
missing. Use exactly one action per step.

- Start content retrieval with the original complete natural-language question.
- Use Entity lookup and sentence-level bridging when a named entity provides a
  useful structural anchor.
- Treat Chunk previews only as navigation. READ a visible parent Chunk when the
  complete context is needed.
- FINISH only when the selected eligible Sentence or READ Chunk evidence
  directly supports the answer.
