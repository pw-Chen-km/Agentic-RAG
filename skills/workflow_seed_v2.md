# Agentic-RAG Workflow Skill (workflow-skill-v2)

<!-- FIXED_RUNTIME_GUIDANCE_START -->
## Fixed runtime guidance
The Agent must output one legal PolicyDecision at a time. Follow the action schema and use only references visible in the current state.
<!-- FIXED_RUNTIME_GUIDANCE_END -->

<!-- RETRIEVAL_POLICY_START -->
## Retrieval policy
### [R01] Start from the current question
- When: No useful evidence is currently visible.
- Action sequence:
  1. Use SEARCH to identify the main entity, relation, or requested attribute.
  2. Choose the target that matches the question's requested level of detail.
- Stop or recovery: Inspect the returned material before choosing the next action.

### [R02] Use a sentence when it already answers the question
- When: A visible Sentence directly states the requested fact and covers the question's conditions.
- Action sequence:
  1. Check the subject, relation, and requested answer level.
  2. Use FINISH with the smallest legal evidence set.
- Stop or recovery: Do not retrieve more material after all answer conditions are covered.
- Exceptions:
  - Continue when a multi-hop question still lacks evidence for another hop.

### [R03] Read a useful visible chunk
- When: A visible unread Chunk may contain the missing fact and no complete Sentence is yet sufficient.
- Action sequence:
  1. READ the most relevant unread Chunk.
  2. Inspect the newly visible Sentences before another SEARCH.
- Stop or recovery: FINISH when the newly visible legal evidence covers the question; otherwise change the missing part of the search.

### [R04] Resolve a bridge before the next hop
- When: The question needs two or more facts and the current evidence identifies an intermediate entity or relation.
- Action sequence:
  1. EXPAND or READ the visible source that can identify the intermediate entity.
  2. SEARCH for the remaining attribute only after the bridge is explicit.
- Stop or recovery: FINISH only after every required hop has legal supporting evidence.
- Exceptions:
  - Skip expansion when the question or visible evidence already names the intermediate entity unambiguously.
<!-- RETRIEVAL_POLICY_END -->

<!-- RECOVERY_POLICY_START -->
## Recovery policy
### [R05] Change path after no progress
- When: The last SEARCH or EXPAND did not add useful information for the missing part of the question.
- Action sequence:
  1. Check for an unread Chunk or expandable visible source.
  2. Use READ or EXPAND when one is available.
  3. Otherwise change the searched entity, attribute, target, or method.
- Stop or recovery: Do not repeat a semantically equivalent SEARCH without a changed target or missing fact.

### [R06] Avoid repeated actions
- When: The previous action was duplicate, invalid, or returned only material already visible.
- Action sequence:
  1. Read the current available-action list.
  2. Choose a different legal path that can add information.
- Stop or recovery: FINISH when existing legal evidence is sufficient; otherwise keep the next action different in purpose.

### [R07] Keep a finishing opportunity
- When: Legal evidence covers all requested conditions or the remaining budget is small.
- Action sequence:
  1. Check the available evidence references.
  2. FINISH with a concise supported answer when possible.
- Stop or recovery: Do not spend the last policy attempts on another search when a legal answer can already be formed.
- Exceptions:
  - Continue only when a required condition is visibly unsupported.
<!-- RECOVERY_POLICY_END -->

<!-- ANSWER_POLICY_START -->
## Answer policy
### [A01] Answer from legal evidence
- When: FINISH is legal and the selected evidence covers the question.
- Action sequence:
  1. Extract the requested answer from the cited evidence.
  2. Preserve required dates, people, relations, and comparison conditions.
  3. Keep the answer direct and concise.
- Stop or recovery: Do not add content that the legal evidence does not support.
<!-- ANSWER_POLICY_END -->

<!-- FIXED_ANSWER_CONTRACT_START -->
## Fixed answer contract
The final answer must be supported by legal evidence. Never guess, fabricate, or add claims that the evidence does not support.
<!-- FIXED_ANSWER_CONTRACT_END -->
