---
name: agentic-rag-v2-2-read
description: Convert an inherited V2-2 READ intent into one executable request for the full text of a visible unread Chunk.
---

# Construct a read action

Use the inherited intent and `visible_handles` to identify the unread Chunk whose
full text can resolve the information gap.

1. Read [the parameter contract](references/contract.md).
2. Choose one visible unread Chunk from its title and preview semantics.
3. Return only the parameter object; the Harness supplies `type=READ`.

Do not choose another action. Do not return assessment or evidence fields. Read
[the recovery guide](references/recovery.md) only after the Validator rejects a
draft.
