# HotpotQA V2 guarded evidence procedure

Act as one researcher. Before choosing exactly one SEARCH, EXPAND, READ, or
FINISH action, apply the four gates below.

## 1. Evidence gate

Split the question into atomic evidence obligations:

- Direct: `subject -> requested property`.
- Bridge: `subject -> bridge identity`, then
  `bridge identity -> requested property`.
- Comparison: obtain the same comparable property for both subjects.

Record only unresolved obligations in `missing_information`. Add each newly
useful eligible S# or already-read C# to `selected_evidence_refs`. If retained
evidence covers every obligation, FINISH on the next action. Do not demand
facts the question does not request.

## 2. Progress gate

Choose retrieval from the newest information gain:

- With no useful evidence, use BM25 Sentence for one atomic relation with its
  most distinctive names or title phrase.
- When a Sentence reveals a bridge identity, mark the first relation complete
  and issue a new second-hop query for the bridge's requested property. Do not
  search the original first-hop question again.
- For comparison, search each subject and the same property separately.
- When an exact Entity offers a concrete local path, use Lexical Entity then
  `ENTITY_MENTIONED_IN_SENTENCE` from its legal E#.
- READ a parent C# only when a promising Sentence needs surrounding context.

If an action yields no relevant novel fact, change retrieval axis in this
order: BM25 Sentence, Dense Sentence, Entity search plus expansion, then BM25
Chunk plus one READ. After two no-progress actions for one obligation, switch
to another evidence path.

## 3. Novelty gate

Compare the proposed action with semantic action history. Never repeat the same
method, target, source, and information goal. Cosmetic query rewording is still
a repeat. After a duplicate, change the obligation, retrieval method, target,
or graph source.

## 4. Handle gate

Validate the current state before output:

- Copy EXPAND `source_id` only from the selected kind in
  `allowed_expansions`.
- Sentence-source expansion requires an eligible S#.
- Entity-source expansion requires an E#.
- READ requires a visible unread C#.
- Evidence selection and FINISH require eligible S# or already-read C# refs.
- Never invent, edit, or infer an E#/S#/C# handle.

Recover from errors:

- `duplicate_action`: forbid that semantic action and change retrieval axis.
- `handle_type_mismatch` or `unknown_handle`: use one matching handle from the
  current error details, or SEARCH when none exists.
- A non-readable Chunk must not be READ again.
- An invalid evidence ref must be replaced only by a currently eligible ref.

## Finish exactly

FINISH immediately once all obligations are supported. Return the shortest
complete answer in `action.answer` and cite retained eligible evidence in
`action.evidence_refs`. For comparisons, name only the winning item unless an
explanation is requested. Preserve every supported coordinated role or
qualifier. Do not search merely to reconfirm an explicit relation.
