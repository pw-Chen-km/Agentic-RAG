# Experimental guarded HotpotQA procedure

This is a post-hoc research condition derived from observed failure patterns;
it is not the neutral baseline. Apply four gates before selecting exactly one
SEARCH, EXPAND, READ, or FINISH action.

## Evidence gate

Split the question into atomic obligations. Keep only unresolved obligations
in `missing_information`. FINISH as soon as visible eligible evidence covers
all obligations.

## Progress gate

With empty memory, BM25-search one distinctive atomic relation as SENTENCE.
After discovering a bridge identity, search its requested property rather than
repeating the original query. For comparisons, retrieve each subject's value
separately. Use Entity expansion only for a concrete known path; READ only when
a promising sentence lacks nearby context.

If there is no progress, change axis: BM25 Sentence, Dense Sentence, Entity
search plus expansion, then BM25 Chunk plus one READ.

## Novelty gate

Do not repeat the same method, target, source, and information goal. Cosmetic
rewording is still a repeat.

## Reference gate

EXPAND requires a currently visible source ref. READ requires a visible unread
C#. FINISH accepts only visible S# and read C# evidence. After an invalid ref,
copy a valid visible ref or SEARCH; do not guess or repair a hidden identifier.

Return the shortest answer that satisfies the question and preserve supported
qualifiers, dates, roles, and comparison conditions.
