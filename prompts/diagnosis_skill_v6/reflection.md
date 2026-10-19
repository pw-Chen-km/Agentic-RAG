YOUR TASK: EXPLAIN WHAT HAPPENED AND WHAT GUIDANCE MIGHT HELP
Read the saved runs and describe useful findings about how the agent chose
its actions. A finding may concern a problem, an effective choice worth
preserving, or an uncertain result that needs further testing.

Call report_diagnoses exactly once. Return at most two diagnoses. Here a
diagnosis means a written explanation of a finding and a possible way to
improve the agent's instructions. An empty list is valid; explain why in
reason. One question is enough to suggest a change worth testing.

Each diagnosis has four parts:
- observations: For each question that directly supports your finding, copy
its exact case_id and write state, action, and result. In state, explain what
the agent already knew and what information was still missing, if the record
lets you tell. In action, explain what the agent actually chose, including
relevant information sent to the tools and tools chosen together. In result, describe what happened
afterward and any relevant measured costs. Use one observation per question.
Describe the events in full sentences. Round numbers are optional; if useful,
write "Decision 6", not an unexplained abbreviation such as "D6". For two runs
of the same question, explicitly identify run A and run B in your explanation.
- analysis: Explain what the observations show about the agent's decisions,
why that matters, and what remains an interpretation. Discuss the behavior
before deciding how any existing rule should be edited. You do not need to
assign the finding to a rule ID or find a defect in an existing rule.
- possible_skill_improvement: In ordinary language, describe what guidance
might help: what situation the agent should recognize, how it could choose
its next action, and why this might improve its sequence of choices. You may suggest
concrete wording as an idea. Leave this empty if no change is justified.
Do not choose add_rule or replace_rule, assign rule IDs, or write Skill JSON.
The next call will decide how to fit an idea into the complete Skill.
- limitations: Explain relevant uncertainty, other possible causes, or cases
that challenge the finding. Leave this empty when there is nothing relevant
to add. A question that only challenges the finding must not be listed as a
supporting observation; explain it here instead.

Keep useful detail in the explanation, but describe each fact once. Do not
repeat a whole action sequence in every field or copy long passage text.
Do not invent what the agent was thinking, an information gap not supported
by the record, or the outcome of an action that was never tried. No formal
rule-editing tools are available in this call.
