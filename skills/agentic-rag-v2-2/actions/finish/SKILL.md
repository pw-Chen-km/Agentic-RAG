---
name: agentic-rag-v2-2-finish
description: Convert an inherited V2-2 FINISH intent and its pending evidence selection into a grounded answer with eligible citations.
---

# Construct a finish action

Answer only from the inherited pending evidence selection.

1. Read [the parameter contract](references/contract.md).
2. Apply [the evidence rules](references/evidence.md).
3. Compose the complete answer and cite the smallest sufficient nonempty subset
   of selected evidence.
4. Return only the parameter object; the Harness supplies `type=FINISH`.

Do not choose another action. Do not add evidence that Stage 1 did not select.
Read [the recovery guide](references/recovery.md) only after the Validator
rejects a draft.
