---
name: agentic-rag-v2-2-search
description: Convert an inherited V2-2 SEARCH intent into one executable lexical, BM25, or dense retrieval request without selecting another action.
---

# Construct a search action

Use the inherited `action_intent`, question, evidence, trajectory, and visible
node semantics to formulate one focused retrieval request.

1. Read [the parameter contract](references/contract.md).
2. Write a semantic query using real names and facts, never a handle as a
   substitute for meaning.
3. Choose the method and granularity most likely to retrieve the missing fact.
4. Return only the parameter object; the Harness supplies `type=SEARCH`.

Do not choose another action. Do not return assessment or evidence fields. Read
[the recovery guide](references/recovery.md) only after the Validator rejects a
draft.
