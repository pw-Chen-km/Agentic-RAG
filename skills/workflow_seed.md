# Agentic-RAG Workflow Skill

## Fixed runtime guidance
The Agent must output one legal PolicyDecision at a time. Follow the action
schema and use only references visible in the current state.

<!-- RETRIEVAL_POLICY_START -->
## Retrieval policy
Learn a general sequence of retrieval actions from the current state. Do not
memorize question-specific queries, answers, or reference IDs.
<!-- RETRIEVAL_POLICY_END -->

<!-- RECOVERY_POLICY_START -->
## Recovery policy
After an empty, duplicate, or invalid action, choose a meaningful alternative
path based on the current visible evidence and available actions.
<!-- RECOVERY_POLICY_END -->

<!-- ANSWER_POLICY_START -->
## Answer policy
When the legal evidence covers the question, FINISH and give a concise,
complete answer using the appropriate evidence references.
<!-- ANSWER_POLICY_END -->

## Fixed answer contract
The final answer must be supported by legal evidence. Never guess, fabricate,
or add claims that the evidence does not support.
