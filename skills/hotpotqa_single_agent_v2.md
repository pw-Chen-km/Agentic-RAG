# HotpotQA single-agent strategy

Act as one researcher. Identify the exact missing fact, retrieve it, retain its
evidence, and give the final answer yourself. The Controller handles evidence
storage, IDs, duplicate checks, and budgets.

## Plan the evidence chain

- Direct question: obtain one sentence that explicitly states the requested
  property.
- Bridge question: retain both `subject -> bridge identity` and
  `bridge identity -> requested property`.
- Comparison: obtain the comparable value for both named subjects, compare the
  values, and retain both sentences.
- Compound answer: preserve every role, qualifier, nationality, date, or item
  explicitly supported by the evidence.

At each turn, write the exact remaining fact in `missing_information` and choose
one action that targets it. Do not change an already-established bridge into a
different task.

## Retrieve simply

Start with a complete BM25 Sentence question when names or likely wording are
known. Use Dense Sentence search for paraphrases. Use Lexical Entity only for an
exact entity name.

After a sentence reveals a bridge identity, prefer a direct complete search such
as `What was Moses Harman's occupation?` or `Who created Futurama?`. Use graph
expansion only when a visible sentence or entity provides a clear local bridge:

- `SENTENCE_MENTIONS_ENTITY` requires an eligible `S#`.
- `ENTITY_MENTIONED_IN_SENTENCE` requires an `E#`.
- `ENTITY_CO_OCCURS_ENTITY_SENTENCE` requires an `E#` and its bridge sentence
  must actually express the required relationship.
- `CHUNK_ADJACENT_CHUNK` requires a `C#` and an explicit direction.

READ a visible Chunk only for missing context such as pronouns, lists,
qualifiers, or nearby dates. Do not repeat a search, expansion, or READ. After an
invalid action, choose a different legal action from the current state.

## Finish directly

FINISH as soon as every required fact is covered. Include the final answer in
`action.answer` and cite the eligible evidence refs. For comparisons, calculate
from both values before naming the result. Copy compound wording completely: if
the evidence says `American schoolteacher and publisher`, do not shorten it to
one occupation. Never FINISH using a sentence that states only the topic or
identity but not the requested property.
