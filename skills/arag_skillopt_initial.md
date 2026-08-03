# A-RAG retrieval strategy

Gather the smallest evidence set that completely answers the question. Base
each action on the current missing information and the semantic handles already
visible to you.

## Start with the complete question

- Prefer BM25 Sentence search when names or wording are likely to appear in the
  corpus.
- Prefer Dense Sentence search when the same fact may be phrased differently.
- Use Lexical Entity search only for an explicit entity name or alias.
- Use Chunk search when one sentence is unlikely to contain enough context.

Keep BM25 and Dense queries as complete natural-language questions or focused
subquestions. Do not replace them with disconnected keywords.

## Build the evidence chain

Treat a retrieval result as a candidate fact, not automatic proof. For
multi-hop questions, identify the entity connecting the established fact to the
missing fact, then use Sentence-to-Entity or Entity-to-Sentence expansion. Use
sentence-scope Entity-to-Entity bridging when the bridge sentence directly
connects two useful entities.

For comparison questions, obtain evidence for every compared subject before
deciding. For synthesis, explanation, or generation questions, retrieve the
separate facts needed to support every material claim.

## Read only when context is needed

Every complete Sentence exposes its parent Chunk. READ it when a pronoun,
qualifier, list, chronology, or surrounding explanation is needed. Chunk
previews are navigation hints and cannot serve as evidence until the Chunk is
READ. Use adjacent Chunk expansion only when the relevant context crosses a
boundary.

## Finish with grounded evidence

Each assessment replaces the complete selected evidence set, so keep earlier
evidence that still supports a required fact. Select only eligible Sentences or
READ Chunks that directly contribute to the answer.

Before FINISH, verify that every requested hop, comparison side, explanation,
or conclusion is supported and that no selected evidence contradicts the
answer. If information is missing, name the exact missing fact and retrieve it
with a complete subquestion instead of repeating the same action.
