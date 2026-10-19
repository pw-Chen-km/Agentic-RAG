YOUR TASK: USE THE FINDINGS TO IMPROVE THE CURRENT SKILL
You receive explanations from the previous review calls, including observed
actions and possible improvements. Read them together with the complete
Current Skill and the descriptions of the available tools. Decide what
instructions would help the agent handle future questions better.

Create the formal edits in this call. A review may suggest wording, but you
must decide how the idea fits the complete Skill. Add a rule when it supplies
useful guidance that is missing. Rewrite an existing rule when it covers the
situation but needs a change. Delete a rule only when the findings justify
removing it. Any of these choices is allowed; do not prefer one rule ID or
require every finding to match an existing rule.

One question may justify testing an improvement. Different edits can address
different problems; they do not need a common theme. Remove duplicate changes
and resolve instructions that would tell the agent to do opposite things in
the same situation. Check the whole resulting Skill, including rules you keep.
Do not make up evidence to resolve a conflict. If it cannot be resolved, omit
the conflicting change. Do not replace or delete the same rule twice.

THE FORMAT OF THE SKILL
The Skill contains a retrieval_policy list. Each rule has exactly four fields:
- rule_id is the rule's identifier, such as R01. Use an existing identifier to
rewrite or delete that rule. Use a new identifier for a new rule.
- title briefly names what the rule helps the agent do.
- when describes a situation the agent can recognize from its question,
visible information, and previous actions.
- action_sequence is a list of written instructions for choosing what to do
in that situation. It can say "if this happens, do this next". It does not require
every question to follow the same fixed sequence. Several listed actions do
not automatically mean that they should be chosen together.

RETURN THE EDITS USING THESE TOOLS
- add_rule: add a new rule. Set section to retrieval_policy and supply a new
rule_id plus rule containing title, when, and action_sequence.
- replace_rule: supply a current rule_id and its complete replacement rule
containing title, when, and action_sequence.
- delete_rule: supply the current rule_id to remove.
- no_change: explain why no useful change is justified. Use this alone.
For add_rule and replace_rule, put rule_id in the outer tool arguments. The
program inserts it into the final four-field rule. Keep question IDs and your
explanation outside rule. Do not put specific training answers, names, queries,
passage IDs, or answers or evidence supplied only for this review in a rule.

For each edit, use reason to explain which finding motivated it, which
instruction you are adding, clarifying, or removing, and how the agent's
behavior should change. Explain expected benefits and relevant risks without
claiming that an untested change has worked. supporting_case_ids must name at
least one supplied question listed in the observations that support that
finding. These IDs identify the evidence for the finding; they do not prove
that your proposed change will work. Do not cite a question mentioned only as
a limitation or a question with no reported observation.

Return at most three edits to create one proposed version of the Skill for
testing on separate questions.
Three is a limit, not a target. Each added or rewritten rule may use at most
300 tokens, measured by the program. The
whole set of editable rules may use at most 1500 tokens. The program checks
format, identifiers, and length; it cannot establish that your explanation is
correct. A test on separate questions measures whether the proposed Skill helps.
