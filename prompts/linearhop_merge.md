# LinearHop SkillOpt merge

Merge only cross-question retrieval-policy edits supported by the recorded
diagnoses. The public actions are `semantic_search`, `entity_hop`,
`linear_hop`, `read_chunk`, and stopping to answer. Keep graph construction,
question-sentence ranking, hop limits, top-k, aliases and Chunk IDs fixed.
Reject edits that mention a specific question, answer, entity, chunk, graph
ID, backend setting or infrastructure detail. Keep the existing bounded edit
and validation rules.
