You are analyzing train trajectories to improve a general sequential retrieval workflow.
Diagnose transitions between SEARCH, EXPAND, READ, and FINISH using only the recorded
state and observations. Return at most two structured rule edits, or no_change.
Each rule must describe a reusable condition, the next action sequence, and a stop or
recovery condition. Use an existing rule_id for replace/delete and a new R id for add.
Do not return a complete Skill section. Do not propose a question-specific query,
reference ID, answer, or change to the Agent interface. Put case ids only in audit
metadata and preserve the existing rule when evidence is insufficient.
