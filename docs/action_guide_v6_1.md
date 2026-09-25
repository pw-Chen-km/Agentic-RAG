# Neutral capability cards (v6.1)

The study uses one shared task skill for every condition. The skill explains the
task, the current observation, and the rule that one operation is selected per
turn. It does not tell the model which retrieval route to try first.

Each condition then supplies a state-specific `OPERATIONS AVAILABLE NOW` section.
This section is a capability registry, not a plan. It contains only operations
that the condition and the currently visible references allow. The list order has
no meaning. Every card uses the same fields:

- `Search scope`: the candidate text units that can be considered;
- `How it works`: how the query or visible entity limits and ranks candidates;
- `Returns`: the complete text unit made visible to the model;
- `Limitation`: what the operation cannot reach.

The exact operation name is shown before the description, for example
`find_sentences(query)`. The constrained decision schema uses the same operation
catalog for its name descriptions, so the prompt and validator cannot silently
describe different capabilities. Conditions change membership in the catalog,
not the wording of a card.

`E#` is an episode-local name reference. Its meaning comes from the displayed
canonical name; it is not an evidence citation. A follow operation accepts only
an E# that is visible in the current observation. Passage and sentence labels
remain the only source labels that can be cited in `evidence_refs`.

The renderer returns complete passages or complete sentences and does not add a
passage preview or truncate a sentence. It separately records returned
references and newly visible source text for audit, while the model sees only the
source text and compact action history needed for the next decision.
