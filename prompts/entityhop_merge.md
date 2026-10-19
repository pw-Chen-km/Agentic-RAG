Merge successful-pattern findings and failure diagnoses conservatively.  Retain
only a policy-level difference supported by multiple independent questions and
check for conflicting paths and counterexamples.  The only actions are
keyword_search, semantic_search, read_chunk, entity_hop, and stop-and-answer.
EntityHop may be used only when a concrete entity is already visible and the
remaining information need plausibly depends on it.  Do not mandate a fixed
workflow, modify the backend/index/schema/controller, or hard-code an entity,
query, chunk, answer, or E/S/C reference.  Keep at most three batch edits and
use only add_rule, replace_rule, delete_rule, or no_change.
