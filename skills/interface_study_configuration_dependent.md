TASK AND OBSERVATION CONTRACT

Answer the original question using only source text shown in this conversation.
OPERATIONS AVAILABLE NOW is the complete list of operations that may be requested in
the current state. An operation not listed there is unavailable. The operation cards
describe each operation's search scope, query meaning, returned text unit, and limits.
Only displayed source text can support the answer. Operation descriptions, action
history, and information-gap assessments are working guidance, not evidence.

POLICY

At each step, make exactly one native tool call from the current tool registry.
Include the information-gap assessment in the same tool call when it is requested.
Use only references and entity names shown in the current observation.

During normal retrieval:

- If no source text is visible, or missing_information is non-empty, retrieve more
  information. Do not call finish while a specific information gap remains.
- If source text is visible and missing_information is empty, finish with an answer.
  The harness records all deduplicated source units shown in the answer stage;
  do not construct a citation list.
- If the previous operation was rejected, repeated, empty, or added no new source
  text, reassess the same concrete gap. Do not submit the same tool and arguments
  again.

CONFIGURATION-DEPENDENT ROUTING

After a retrieval result, first compare the current missing_information with the
visible source text and the visible entity names.

- When an entity-follow operation is listed and a visible, navigable entity is a
  plausible anchor for the unresolved fact or relationship, prefer that entity-follow
  operation for the next retrieval. Choose the entity yourself from the visible E#
  references. Use the landing unit provided by the operation: it is a passage in a
  passage-following condition and a sentence in a sentence-following condition.
- Do not follow an entity merely because it is visible. The entity should have a
  reasonable connection to the specific unresolved information.
- When no visible entity is a plausible anchor, use a global search operation with a
  materially different query that states the unresolved fact or relationship.
- When no entity-follow operation is listed, global query reformulation is the
  available route for an unresolved gap. An annotation-only E# cannot be followed.
- If a local hop adds no useful new source text, reassess the gap. Another visible
  entity or a legal global search may be selected; do not repeat the same operation.

This routing rule applies equally to every configuration. The configuration changes
which operations are listed, not the rule used to choose among them. Do not retrieve
more only to exercise a tool that happens to be available.

INFORMATION-GAP ASSESSMENT

When assessment is requested, produce resolved_gaps and missing_information in the
same tool call as the action. Keep resolved_gaps cumulative. Replace
missing_information on every turn with the complete set of specific facts or
relationships still needed to answer the original question.

Write each gap as a precise, searchable statement. Preserve the exact names,
relations, dates, locations, and other qualifiers from the question or visible source.
When a gap is resolved, begin the resolved item with the earlier gap wording and append
the established value after an em dash. An assessment is a working judgment, not
source evidence, and does not create new references.

FIXED ACTION RULES

E# labels identify entity names, while C# and S# labels identify source units. The
harness records visible source units automatically, so finish does not require an
evidence_refs list. Do not invent hidden sources, unavailable operations, entity names,
or references. A budget-finalize turn is different: retrieval is closed and the only
legal call is finish, even if a gap remains.
