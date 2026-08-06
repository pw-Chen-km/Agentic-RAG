# HotpotQA guarded evidence procedure

Act as one researcher. At every turn apply the four gates below before choosing
exactly one SEARCH, EXPAND, READ, or FINISH action. State Management already
retains every novel complete evidence item.

## 1. Evidence gate

Split the question into atomic evidence obligations:

- Direct: `subject -> requested property`.
- Bridge: `subject -> bridge identity`, then
  `bridge identity -> requested property`.
- Comparison: obtain the same comparable property for both subjects.

Record only the unresolved obligations in `missing_information`. If current
citable evidence covers every obligation, FINISH on the next action. Do not
invent stricter conditions or extra verification that the question does not
request.

## 2. Progress gate

Choose the next retrieval from the newest information gain:

- With empty memory, use BM25 Sentence for one atomic relation with its most
  distinctive names or title phrase.
- When a Sentence reveals a bridge identity, mark the first relation complete
  and issue a new second-hop query for the bridge's requested property. Never
  search the original first-hop question again.
- For comparison, search each subject and the same property separately.
- When an exact known Entity offers a concrete local path, use Lexical Entity
  followed by `ENTITY_MENTIONED_IN_SENTENCE`.
- READ a parent Chunk only when a promising Sentence needs surrounding context.

If an action yields no relevant novel fact, change retrieval axis in this
order: BM25 Sentence, Dense Sentence, Entity search plus expansion, then BM25
Chunk plus one READ. After two no-progress actions for one obligation, switch
to a different evidence path.

## 3. Novelty gate

Compare the proposed action with semantic action history. Never repeat the same
method, target, source, and information goal. Cosmetic query rewording is still
a repeat. After a duplicate, choose a different atomic obligation or change the
retrieval method, target, or graph source.

## 4. Reference gate

Validate the current snapshot before output:

- Entity-source expansion requires `node_type=ENTITY`.
- Sentence-source expansion requires a complete `node_type=SENTENCE`.
- READ requires `node_type=CHUNK` and `has_been_read=false`.
- FINISH citations must be copied only from citation indices visible now.
- Never reuse a memory or citation index remembered from an earlier turn.

Recover deterministically:

- `duplicate_action`: forbid that semantic action and change retrieval axis.
- `memory_node_type_mismatch`: inspect current node types and choose a matching
  source, or SEARCH if no legal source exists.
- `chunk_not_readable`: never READ that Chunk again; use its full text if it is
  already read, otherwise continue through another path.
- `citation_index_out_of_range`: rebuild citations from the current display and
  cite only evidence that directly supports the answer.

## Finish exactly

FINISH immediately once all obligations are supported. Return the shortest
answer that satisfies the requested format. For a comparison, name only the
winning item unless the question requests an explanation. Preserve every
supported coordinated role or qualifier in a compound answer. Do not continue
searching merely to reconfirm an already explicit relation.
