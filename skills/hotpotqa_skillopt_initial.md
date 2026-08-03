# HotpotQA retrieval strategy

Gather the smallest complete evidence chain needed to answer the question. Base every action on the current missing information and the nodes already visible to you.

## Choose the first retrieval

Start with the original complete question.

- Use BM25 Sentence search when the question is likely to share names or wording with the corpus.
- Use Dense Sentence search when the needed fact may be expressed with different wording.
- Use Lexical Entity search only when an explicit entity name or alias is a useful graph anchor.
- Use Chunk search only when the question asks for broad context unlikely to be resolved by one sentence.

Do not shorten a natural-language question into disconnected keywords for BM25 or Dense search.

## Follow multi-hop evidence

Treat a retrieved sentence as a possible fact, not automatically as sufficient evidence.

For bridge questions:

1. Identify what the first sentence establishes.
2. Identify the entity that connects the known fact to the missing fact.
3. Use `SENTENCE_MENTIONS_ENTITY` when a complete sentence contains a useful bridge entity.
4. Use `ENTITY_MENTIONED_IN_SENTENCE` to find complete sentences about that entity.
5. Use `ENTITY_CO_OCCURS_ENTITY_SENTENCE` when local Entity–Sentence–Entity traversal can reveal the next entity and its bridge sentence.
6. Ask the next retrieval as a complete natural-language subquestion focused on the missing fact.

For comparison questions, gather separate evidence for both compared subjects before deciding the comparison result. Do not finish after finding evidence for only one side.

## Read context selectively

Every complete Sentence exposes its parent Chunk ID. READ that Chunk when:

- the sentence contains an ambiguous pronoun or reference;
- the sentence alone does not establish the required relationship;
- surrounding sentences may contain the second part of a fact;
- the answer depends on a list, qualifier, date, or comparison context.

Use `CHUNK_ADJACENT_CHUNK` only when the relevant context appears to cross a Chunk boundary. Chunk previews are navigation hints, not proof.

## Maintain evidence

Keep previously selected evidence that still supports a required fact because each assessment replaces the complete selected evidence set.

Prefer a small evidence set in which every selected Sentence or READ Chunk contributes directly to the answer. Do not select an Entity, an unread Chunk, a preview, or a high-scoring result whose text does not support a required fact.

Before FINISH, verify:

- the evidence answers the exact requested answer type;
- every required hop or comparison side is supported;
- no selected evidence contradicts the proposed answer;
- the FINISH refs are the minimal sufficient subset of the selected evidence.

If evidence is incomplete, state the specific missing fact and retrieve that fact. If a search fails, change method or ask a clearer complete subquestion instead of repeating the same action.
