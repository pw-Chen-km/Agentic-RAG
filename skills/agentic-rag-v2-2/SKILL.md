---
name: agentic-rag-v2-2
description: Select the next high-level Agentic RAG action, explain its information-seeking intent, and retain eligible evidence before an action-specific skill constructs executable parameters.
---

# Select the next retrieval action

Choose strategy, not function parameters. Inspect the question, accumulated
evidence, latest observation, trajectory, budget, and visible node semantics.

## Procedure

1. Identify the exact fact still needed to answer the question.
2. Select every visible, complete evidence item that should remain accumulated.
3. Choose exactly one high-level action:
   - `SEARCH`: formulate a new retrieval request when the visible graph does not
     provide a useful starting node or a revised query is needed.
   - `EXPAND`: follow a useful visible Entity, Sentence, or Chunk to connected
     nodes.
   - `READ`: obtain the full text of a visible unread Chunk when its preview is
     insufficient.
   - `FINISH`: answer only when the selected evidence directly supports every
     required fact.
4. State the information gap and why the chosen action is appropriate in
   `action_intent`.

## Output boundary

Return only:

- `action_type`: `SEARCH`, `EXPAND`, `READ`, or `FINISH`.
- `action_intent`: a concise semantic explanation of the missing fact and the
  reason for choosing this action.
- `selected_evidence_refs`: the complete Sentence or read Chunk evidence to
  retain transactionally if the resulting action becomes valid.

Do not choose a query, method, target, expansion kind, source handle, direction,
Chunk handle, answer, or citations here. Do not use an opaque handle such as
`E2` as the meaning of the intent; use the visible entity name or text instead.

Evidence refs must be visible, unique, and eligible. Never select an Entity, a
Sentence preview, or an unread Chunk as evidence. An empty evidence selection is
valid while relevant evidence has not yet been found.
