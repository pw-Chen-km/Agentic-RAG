YOUR TASK IN THIS CALL: IMPROVE THE CURRENT SKILL FROM DIAGNOSES
You receive reports written by the Reflection stage. Each report describes
observed decisions, possible limitations and improvement ideas. Read these
reports together with the complete Current Skill and Fixed tool guidance.

Decide how the Current Skill could be improved so that the agent makes better
decisions in future questions. You create the edits in this stage; you are not
selecting pre-written edits. A diagnosis from one question may justify testing
a general rule. You do not need another question showing the same problem.
Different edits may address entirely different problems. Do not reject useful
diagnoses merely because they do not form a shared theme.

For each change, explain in reason which supplied report/diagnosis it comes
from, what the current rule is missing or gets wrong, what behavior the edit is
intended to change, and the expected answer-quality/cost tradeoff. Say which
effects are observed and which remain untested hypotheses. supporting_case_ids
must contain at least one eligible supplied question ID whose SUPPORT observation
grounds the diagnosis. Multiple reports or branches of one question are still
one question. Do not cite neutral or counterexample observations as support.
Read them as caveats when deciding what to change. Layer and basis labels are
advisory, not automatic vetoes.

Before returning edits, remove duplicate intentions and check the resulting
Skill for contradictions. Different actions under different conditions can
coexist. Opposite instructions under the same conditions require resolution.
You may clarify conditions when the diagnoses support that distinction; do not
invent evidence to reconcile a conflict. If it cannot be resolved, omit the
conflicting change. Do not replace or delete the same rule twice in one batch.
Do not merely restate an existing rule.

RETURN ONLY THE ALLOWED EDIT TOOL CALLS
- add_rule: add a new rule. Set section to retrieval_policy, give it a new
rule_id, and supply rule with title, when and action_sequence.
- replace_rule: use the ID of a current rule and supply its complete replacement
title, when and action_sequence.
- delete_rule: identify a current rule that the diagnoses give a reason to remove.
- no_change: provide a reason when no useful, permitted change is justified.
This call must appear alone.

For add/replace, rule_id is supplied outside rule in the tool arguments. The
program places that ID into the final four-field rule. Do not add extra runtime
fields. Keep reasons, question IDs and diagnosis references outside rule.

Return at most THREE edits, producing ONE candidate for later validation. Three
is a limit, not a target. If more improvements seem useful, prioritize the ones
whose connection to a diagnosed decision and expected benefit you can explain
most clearly. Do not invent a shared problem to justify independent edits.
Each new/replaced rule body may use at most 300 o200k_base tokens. The whole
rendered mutable Skill may use at most 1500 tokens. The program checks format,
IDs and these limits; it cannot prove your interpretation or rule is correct.
Validation, not this discussion, will measure whether the candidate helps.
