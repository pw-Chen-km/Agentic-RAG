"""A-RAG-specific reflection prompts.

Action availability comes from the supplied runtime menu and tool schemas.
These strings remain hashable experiment inputs, not a second tool inventory.
"""
from __future__ import annotations

RETRIEVAL_PROMPT = r"""You improve a small formatted Skill for the A-RAG document agent.

Use exactly the enabled actions in the supplied RUNTIME ACTION SPACE section,
including stopping to answer. It can call more than one document tool in a loop. Your input is a
set of completed training rollouts and the current four-field rules (rule_id,
title, when, action_sequence).

Study complete paths, not isolated actions.  For each path ask:
1. Was the selected search or enabled hop a sensible retrieval method?
2. After a candidate chunk appeared, was read_chunk used at the right time?
3. Did the agent repeat a search without new information, stop too early, or
   spend calls/tokens without improving the evidence?
4. Did the proposed change help across at least two independent questions?

Return at most two small edits using the native tools add_rule,
replace_rule, delete_rule, or no_change. Prefer replacing or deleting an
existing rule over adding a near-duplicate. An edit must be a general policy
for the enabled actions, not a question-specific query, answer, chunk ID, or
dataset fact. If the cases do not support a repeated pattern, call no_change.
Do not modify the tool interface, evaluator, or answer contract. The rule
fields are exactly title, when, and action_sequence; do not add other fields.
"""

META_PROMPT = r"""You are the meta analyst for an A-RAG SkillOpt experiment.

Compare parent-Skill and candidate-Skill trajectories for the same question.
Use exactly the enabled actions in the supplied RUNTIME ACTION SPACE section
and their tool schemas. Use the paired paths as evidence about complete retrieval workflow:
which path found useful chunks earlier, which path repeated searches or read
too late, and which path used fewer calls/tokens for comparable answer quality.

A difference is actionable only when it repeats across multiple independent
questions. Do not claim that one action is a guaranteed cause. Return at most
two edits with add_rule, replace_rule, delete_rule, or no_change. Rules must be
general and use only the fields title, when, action_sequence. Never include a
question, answer, query, chunk identifier, or change to the A-RAG tools.
When the evidence is weak or mixed, call no_change.
"""

