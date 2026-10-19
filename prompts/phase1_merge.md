Merge the diagnoses from the entire rollout batch into at most THREE edits and
ONE candidate. This is the first point at which rule edits are produced. Analysts
reported observations and tentative hypotheses, not pre-approved modifications.
Read success AND failure groups, including singleton diagnoses, empty reports,
neutral cases and counterexamples. Find the same limiting decision pattern in at
least two DIFFERENT questions; unrelated diagnoses do not become corroboration
merely because both mention search/read. failure_layer and basis are advisory,
not hard exclusions: an interface/answer/uncertain label can coexist with a
modifiable retrieval decision. If citing such a diagnosis, explain in reason why
the proposed rule can change that decision without changing the tools, backend,
controller, answer contract or evaluator. A purely unavailable-tool/data problem
is not repaired by a rule. Existing wording may need refinement, but simply
restating a rule without specifying changed behavior is not an improvement.

For each edit, use reason to connect: supplied diagnosis/group IDs -> observed
decision pattern -> current rule limitation -> intended changed behavior ->
accuracy/call/token tradeoff and counterexamples. supporting_case_ids must cite
actual SUPPORT observations, not neutral or counterexample IDs. State which part
is observed and which is an untested hypothesis. Do not invent new evidence.
Use the explicitly supplied eligible supporting_case_ids; they include SUPPORT
cases from all layer/basis classifications. The list checks provenance and
observation labels only, not whether two cases support the same proposed change.
You must still establish that shared decision pattern; do not pad it with
unrelated cases or treat uncertain classifications as proven causal facts.
Compare the rule before and after: does this repair the diagnosed decision? Do
not re-propose a costly historical rule while claiming it demonstrated savings.
Do not credit READ for a benefit caused by an earlier search-target change.

Prefer refinement/replacement or deletion over redundant additions. Combine
duplicate intentions, narrow supported conditions, or discard conflicting ideas.
Do not silently overwrite one rule twice. Use no_change when no shared defensible
change survives; three is a ceiling, not a target. The existing gate tests ONE
candidate after the whole batch; there is no per-diagnosis adoption.
