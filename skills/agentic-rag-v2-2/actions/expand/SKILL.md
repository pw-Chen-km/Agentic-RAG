---
name: agentic-rag-v2-2-expand
description: Convert an inherited V2-2 EXPAND intent into one executable traversal from a visible Entity, Sentence, or Chunk using the enabled graph relations.
---

# Construct an expand action

Use the inherited intent and the semantic content of `visible_handles` to choose
one source and one relationship.

1. Read [the parameter contract](references/contract.md).
2. Read [the enabled relations](references/relations.md).
3. Choose a visible source whose meaning supports the inherited intent.
4. Put the handle only in `source_id`; write any optional query semantically.
5. Return only the parameter object; the Harness supplies `type=EXPAND`.

Do not choose another action. Do not return assessment or evidence fields. Read
[the recovery guide](references/recovery.md) only after the Validator rejects a
draft.
