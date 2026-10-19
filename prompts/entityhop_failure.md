Diagnose failures before proposing a policy hypothesis.  The fixed action space
is keyword_search, semantic_search, read_chunk, entity_hop, and stop-and-answer.
Identify what limited evidence acquisition: a missing or premature hop, an
unread useful result, repeated retrieval, or an early answer.  A hypothesis must
be grounded in repeated failures from at least two question IDs and must remain
general across states.  Do not edit the EntityHop backend, entity substrate,
BM25, controller, tool schema, or evaluator; do not include specific entities,
queries, chunks, answers, or E/S/C references.  Return at most two native edits
or no_change, with only the four-field Skill rule format.
