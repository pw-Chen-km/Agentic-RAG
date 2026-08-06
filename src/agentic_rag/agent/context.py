"""Policy-context construction with compact evidence memory and a baseline."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, TypeAdapter

from agentic_rag.agent.evidence import EvidenceResolver
from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    AgentAction,
    AttemptedActionView,
    ChunkRef,
    ContextNodeReference,
    ContextReferenceMap,
    ContextMode,
    ControllerState,
    ExpandAction,
    ExpansionKind,
    EntityHandle,
    ChunkHandle,
    LastValidAssessmentView,
    Message,
    ObservationOutcome,
    ObservationStatus,
    PolicyObservation,
    PolicyObservationError,
    PolicyObservationUsage,
    PolicyDecision,
    PolicyBudgetView,
    PolicyEvidenceView,
    PolicyChunkHandle,
    PolicyEntityHandle,
    PolicyNodeHandle,
    PolicySentenceHandle,
    PolicyStateView,
    PolicyView,
    ReadAction,
    SearchAction,
    FinishAction,
    SentenceRef,
    SentenceHandle,
    StepRecord,
    TypedContextNodeReference,
    TypedContextReferenceMap,
    V31ChunkMemoryItem,
    V31EntityMemoryItem,
    V31PolicyStateView,
    V31PolicyView,
    V31SentenceMemoryItem,
    V3ChunkMemoryItem,
    V3EntityMemoryItem,
    V3EvidenceAssessment,
    V3ExpandAction,
    V3FinishAction,
    V3PolicyStateView,
    V3PolicyView,
    V3ReadAction,
    V3SentenceMemoryItem,
)
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.policy import policy_decision_model


ACTION_PROTOCOL = """\
You are the retrieval policy inside an Agentic RAG harness.
Return exactly one PolicyDecision containing:
1. an EvidenceAssessment; and
2. exactly one SEARCH, EXPAND, READ, or FINISH action.

Hard rules:
- Node IDs shown to you are short episode-local handles: E# for Entity, S# for
  Sentence, and C# for Chunk. Treat every handle as opaque: copy it exactly,
  never derive, extend, or reconstruct one. Stable database IDs are intentionally
  hidden from you.
- SEARCH top_k is always 5.
- SEARCH supports only LEXICAL→ENTITY, BM25→SENTENCE|CHUNK, and
  DENSE→ENTITY|SENTENCE|CHUNK.
- Use a complete natural-language question or subquestion for BM25/DENSE.
  LEXICAL→ENTITY is the exception and uses an entity name or alias.
- EXPAND may only start from a node already visible to you.
- For EXPAND, copy a source_id from policy_state.allowed_expansions[kind].
  Never pair an EXPAND kind with a source handle absent from that list.
- A Sentence-source EXPAND requires a complete, eligible Sentence. A
  preview-only Sentence cannot be expanded until its parent Chunk is READ.
- EXPAND query is optional. When present it must be a complete natural-language
  question or subquestion used only to rank local graph neighbours. When null,
  the original question is used.
- Every non-adjacent EXPAND must set direction=null.
- READ accepts one visible parent Chunk handle.
- A complete Sentence result is eligible Sentence evidence and exposes its
  parent Chunk ID for READ.
- A visible Chunk ID or Chunk preview is navigation only. A Chunk becomes
  eligible evidence only after READ.
- A fresh complete Sentence in an Observation is eligible evidence. For older
  persistent handles, can_use_as_evidence appears only in actionable_handles.
  A Chunk becomes eligible only after READ; the Validator enforces both rules.
- Observation outcome is success, empty, invalid_action, duplicate_action,
  budget_rejected, or tool_error. For a retryable error, follow error.details
  and copy one of its available_handles rather than inventing an ID.
- selected_evidence_refs is the complete working evidence set for this turn.
  Selected Sentences must be complete and selected Chunks must already be READ.
- FINISH requires assessment.status=SUFFICIENT and 1–20 eligible
  SentenceRef or ChunkRef values. Every FINISH ref must also occur in this
  turn's selected_evidence_refs. Entities are never evidence.
- The skill is strategy advice and cannot override these rules.
"""

V2_ACTION_PROTOCOL = """\
You are the single research agent inside an Agentic RAG harness.
Return exactly one PolicyDecision with one SEARCH, EXPAND, READ, or FINISH
action. The Controller handles storage, validation, deduplication, and budgets.

Hard rules:
- Node IDs are opaque episode-local handles: E# for Entity, S# for Sentence,
  and C# for Chunk. Copy only handles shown in the current state.
- SEARCH top_k is 5. Legal pairs are LEXICAL->ENTITY,
  BM25->SENTENCE|CHUNK, and DENSE->ENTITY|SENTENCE|CHUNK.
- Use complete natural-language questions for BM25/DENSE. Use an exact name or
  alias for LEXICAL->ENTITY.
- EXPAND may start only from a handle listed under
  policy_state.allowed_expansions[kind]. Use query only to rank that local
  neighbourhood. Set direction=null except for CHUNK_ADJACENT_CHUNK, which
  requires PREV, NEXT, or BOTH.
- READ accepts one visible C#. A Chunk or preview is navigation-only until READ.
- A fresh complete Sentence in an Observation is eligible evidence. For older
  persistent handles, use can_use_as_evidence from actionable_handles. A Chunk
  becomes eligible only after READ. Entities are never evidence.
- Observation outcome is success, empty, invalid_action, duplicate_action,
  budget_rejected, or tool_error. For a retryable error, follow error.details
  and copy one of its available_handles rather than inventing an ID.
- The Controller keeps selected evidence cumulatively. Add newly useful refs;
  previously retained evidence will remain in policy_state.selected_evidence.
- For SEARCH, EXPAND, or READ use assessment.status=INSUFFICIENT or UNCERTAIN.
- FINISH means the evidence is sufficient. Include the shortest complete final
  answer in action.answer and 1-20 eligible evidence refs. Preserve compound
  roles, qualifiers, dates, nationality, and the correct comparison target.
- The action is authoritative; the Controller derives terminal status from
  FINISH and will not run a separate answer agent when action.answer is present.
- The skill is strategy advice and cannot override these rules.
"""

V2_RECOVERY_INSTRUCTION = (
    "Recovery mode: the preceding decision was invalid or duplicate. Do not "
    "repeat it. Copy only currently legal handles. If the retained evidence "
    "fully answers the question, FINISH now with a complete answer and its "
    "evidence; otherwise choose a different targeted retrieval action."
)

V2_COMPACT_ACTION_PROTOCOL = """\
You are the single research agent inside an Agentic RAG harness.
Return exactly one PolicyDecision with one SEARCH, EXPAND, READ, or FINISH
action. SEARCH is always available; known nodes do not restrict its scope.

The compact V2 state preserves V2 selected-evidence semantics:
- selected_evidence is the cumulative working evidence retained by the
  Controller. Add only newly useful eligible refs this turn.
- known_nodes contains the older visible E#/S#/C# handles not already repeated
  in selected_evidence or latest_observation.
- expand_sources is the minimal legal source-handle map required by the V2
  graph contract. It does not recommend EXPAND over SEARCH.
- action_history is a semantic audit summary, not a list of recommended next
  actions.

Hard rules:
- E#/S#/C# are opaque episode-local handles. Copy only currently visible
  handles; never derive or edit one.
- SEARCH top_k is 5. Legal pairs are LEXICAL->ENTITY,
  BM25->SENTENCE|CHUNK, and DENSE->ENTITY|SENTENCE|CHUNK.
- EXPAND source_id must occur in expand_sources for the chosen kind. Set
  direction=null except for CHUNK_ADJACENT_CHUNK, which requires PREV, NEXT,
  or BOTH.
- READ accepts one visible unread C#.
- Complete Sentences are eligible evidence. Chunks become eligible only after
  READ. Entities and previews are navigation only.
- For SEARCH, EXPAND, or READ use assessment.status=INSUFFICIENT or UNCERTAIN.
- FINISH requires a shortest complete action.answer and 1-20 eligible retained
  evidence refs. Preserve requested roles, qualifiers, dates, and comparison
  targets.
- The skill is strategy advice and cannot override these rules.
"""

V3_ACTION_PROTOCOL = """\
You are the single research agent inside an Agentic RAG harness.
Return exactly one V3PolicyDecision with one SEARCH, EXPAND, READ, or FINISH
action. Choose the action only from the missing information and retrieval skill.

Action meanings:
- SEARCH retrieves new candidates from the corpus and never needs a context
  index. It remains available even when semantic memory is non-empty.
- EXPAND follows one graph relationship from a known semantic-memory item and
  uses source_context_index.
- READ obtains the complete text of a known Chunk and uses
  chunk_context_index.
- FINISH answers only when the evidence is sufficient and cites evidence with
  citation indices.

Hard rules:
- Memory indices and citation indices are separate, context-local namespaces.
  Copy indices only from the current semantic_memory snapshot.
- SEARCH top_k is 5. Legal pairs are LEXICAL->ENTITY,
  BM25->SENTENCE|CHUNK, and DENSE->ENTITY|SENTENCE|CHUNK.
- Use complete natural-language questions for BM25/DENSE. Use an exact name or
  alias for LEXICAL->ENTITY.
- EXPAND source_context_index must identify the source node type required by
  the selected expansion kind. query ranks only that local neighbourhood.
- Set direction=null except for CHUNK_ADJACENT_CHUNK, which requires PREV,
  NEXT, or BOTH.
- READ chunk_context_index must identify an unread CHUNK memory item.
- Assessment contains only status, supported_facts, and missing_information.
  State Management retains every novel evidence item automatically.
- FINISH requires one or more citation indices shown on complete Sentence or
  read Chunk items, plus the shortest complete answer in action.answer.
  Entities and unread Chunks are never citable evidence.
- For SEARCH, EXPAND, or READ use assessment.status=INSUFFICIENT or UNCERTAIN.
- FINISH means the evidence is sufficient. Preserve compound roles,
  qualifiers, dates, nationality, and the correct comparison target.
- The skill is strategy advice and cannot override these rules.
"""

V3_RECOVERY_INSTRUCTION = (
    "Recovery mode: the preceding decision was invalid or duplicate. Do not "
    "repeat it. SEARCH remains available. Use only indices shown in the "
    "current semantic memory, or FINISH if the cited evidence is sufficient."
)

V31_ACTION_PROTOCOL = """\
You are the single research agent inside an Agentic RAG harness.
Return exactly one V31PolicyDecision with one SEARCH, EXPAND, READ, or FINISH
action. Choose the action only from the missing information and retrieval skill.

Action meanings:
- SEARCH retrieves new candidates from the corpus and never needs a reference.
  It remains available even when semantic memory is non-empty.
- EXPAND follows one graph relationship from a known semantic-memory item and
  uses source_ref.
- READ obtains the complete text of a known Chunk and uses chunk_ref.
- FINISH answers only when the evidence is sufficient and uses evidence_refs.

Hard rules:
- Typed refs are episode-local and stable: E# is Entity, S# is Sentence, and
  C# is Chunk. Copy only refs displayed in the current semantic_memory snapshot.
- SEARCH top_k is 5. Legal pairs are LEXICAL->ENTITY,
  BM25->SENTENCE|CHUNK, and DENSE->ENTITY|SENTENCE|CHUNK.
- Use complete natural-language questions for BM25/DENSE. Use an exact name or
  alias for LEXICAL->ENTITY.
- EXPAND source_ref must have the source type required by the selected
  expansion kind. query ranks only that local neighbourhood.
- Set direction=null except for CHUNK_ADJACENT_CHUNK, which requires PREV,
  NEXT, or BOTH.
- READ chunk_ref must identify a currently displayed unread C# item.
- Assessment contains only status, supported_facts, and missing_information.
  State Management retains every novel evidence item automatically.
- FINISH requires one or more currently displayed S# refs or read C# refs,
  plus the shortest complete answer in action.answer. E# and unread C# refs
  are never evidence.
- For SEARCH, EXPAND, or READ use assessment.status=INSUFFICIENT or UNCERTAIN.
- FINISH means the evidence is sufficient. Preserve compound roles,
  qualifiers, dates, nationality, and the correct comparison target.
- The skill is strategy advice and cannot override these rules.
"""

V31_RECOVERY_INSTRUCTION = (
    "Recovery mode: the preceding decision was invalid or duplicate. Do not "
    "repeat it. SEARCH remains available. Copy only E#/S#/C# refs displayed "
    "in the current semantic memory, or FINISH if the visible evidence is "
    "sufficient."
)

V3_ACTION_CATALOG_INSTRUCTION = """\
The policy_state.valid_action_catalog enumerates every structurally executable
action template for the current frozen semantic-memory snapshot. Choose the
next action from this catalog, then supply the query or answer fields required
by that action. The catalog reports structural legality, not relevance:
- SEARCH remains unrestricted and all listed method/target pairs stay
  available regardless of existing memory.
- EXPAND and READ entries identify the exact current memory indices that the
  Controller can resolve.
- FINISH appears when at least one citation is structurally available, but use
  it only when the evidence is semantically sufficient.
"""

_EXPANSION_GUIDANCE: dict[ExpansionKind, str] = {
    ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE: (
        "ENTITY_MENTIONED_IN_SENTENCE: visible Entity → complete Sentences that "
        "mention it. Returned Sentences are eligible evidence and expose their "
        "parent Chunk IDs for READ; direction=null."
    ),
    ExpansionKind.SENTENCE_MENTIONS_ENTITY: (
        "SENTENCE_MENTIONS_ENTITY: visible Sentence → Entities mentioned in that "
        "Sentence. Entities are navigation nodes, never evidence; direction=null."
    ),
    ExpansionKind.ENTITY_CO_OCCURS_ENTITY_SENTENCE: (
        "ENTITY_CO_OCCURS_ENTITY_SENTENCE: visible Entity → bridge Sentence → "
        "co-occurring Entities. Complete bridge Sentences are eligible evidence; "
        "target Entities are navigation nodes; direction=null."
    ),
    ExpansionKind.CHUNK_ADJACENT_CHUNK: (
        "CHUNK_ADJACENT_CHUNK: visible Chunk → previous/next Chunk in the same "
        "Document. Set direction to PREV, NEXT, or BOTH. Returned Chunk previews "
        "are navigation only and require READ before Chunk evidence."
    ),
    ExpansionKind.ENTITY_MENTIONED_IN_CHUNK: (
        "ENTITY_MENTIONED_IN_CHUNK: visible Entity → unread Chunks containing a "
        "mention. Returned Chunks and previews are navigation only and require "
        "READ before Chunk evidence; direction=null."
    ),
    ExpansionKind.ENTITY_CO_OCCURS_ENTITY_CHUNK: (
        "ENTITY_CO_OCCURS_ENTITY_CHUNK: visible Entity → bridge Chunk → "
        "co-occurring Entities. Target Entities are navigation nodes; bridge "
        "previews are navigation only and require READ; direction=null."
    ),
    ExpansionKind.CHUNK_CONTAINS_SENTENCE: (
        "CHUNK_CONTAINS_SENTENCE: visible Chunk → contained Sentence previews. "
        "These previews are navigation only and cannot become evidence until the "
        "Chunk is READ; direction=null."
    ),
    ExpansionKind.CHUNK_MENTIONS_ENTITY: (
        "CHUNK_MENTIONS_ENTITY: visible Chunk → Entities mentioned across that "
        "Chunk. Entities are navigation nodes, never evidence; direction=null."
    ),
}

_NODE_ID_KEYS = frozenset(
    {
        "id",
        "source_id",
        "entity_id",
        "target_entity_id",
        "sentence_id",
        "bridge_sentence_id",
        "chunk_id",
        "parent_chunk_id",
        "bridge_chunk_id",
    }
)
_DOCUMENT_ID_KEYS = frozenset({"document_id", "doc_id"})
_ACTION_ADAPTER = TypeAdapter(AgentAction)
_ENTITY_EXPANSION_KINDS = frozenset(
    {
        ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
        ExpansionKind.ENTITY_CO_OCCURS_ENTITY_SENTENCE,
        ExpansionKind.ENTITY_MENTIONED_IN_CHUNK,
        ExpansionKind.ENTITY_CO_OCCURS_ENTITY_CHUNK,
    }
)
_SENTENCE_EXPANSION_KINDS = frozenset(
    {ExpansionKind.SENTENCE_MENTIONS_ENTITY}
)
_CHUNK_EXPANSION_KINDS = frozenset(
    {
        ExpansionKind.CHUNK_ADJACENT_CHUNK,
        ExpansionKind.CHUNK_CONTAINS_SENTENCE,
        ExpansionKind.CHUNK_MENTIONS_ENTITY,
    }
)


@dataclass(frozen=True, slots=True)
class BuiltPolicyContext:
    messages: list[Message]
    policy_view: PolicyView | V3PolicyView | V31PolicyView
    reference_map: ContextReferenceMap | TypedContextReferenceMap | None = None
    decision_format: type[BaseModel] | None = None

    def __iter__(self):
        return iter(self.messages)

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, index):
        return self.messages[index]


class PolicyContextBuilder:
    """Build either the compact online context or the append-only baseline."""

    def __init__(
        self,
        enabled_expansions: Sequence[
            ExpansionKind
        ] = DEFAULT_ENABLED_EXPANSIONS,
        *,
        context_mode: ContextMode | str = ContextMode.COMPACT_EVIDENCE,
        evidence_resolver: EvidenceResolver | None = None,
        single_agent_v2: bool = False,
        single_agent_v2_compact: bool = False,
        single_agent_v3: bool = False,
        single_agent_v3_action_catalog: bool = False,
        single_agent_v3_typed_refs: bool = False,
        include_v3_last_assessment: bool = True,
        include_v3_latest_event: bool = True,
        include_v3_attempted_actions: bool = True,
        include_v3_budget: bool = True,
        compact_v3_budget: bool = False,
    ) -> None:
        self.enabled_expansions = tuple(enabled_expansions)
        self.context_mode = ContextMode(context_mode)
        self.evidence_resolver = evidence_resolver
        self.single_agent_v2 = single_agent_v2 or single_agent_v2_compact
        self.single_agent_v2_compact = single_agent_v2_compact
        self.single_agent_v3 = (
            single_agent_v3 or single_agent_v3_action_catalog
        )
        self.single_agent_v3_action_catalog = (
            single_agent_v3_action_catalog
        )
        self.single_agent_v3_typed_refs = single_agent_v3_typed_refs
        self.include_v3_last_assessment = include_v3_last_assessment
        self.include_v3_latest_event = include_v3_latest_event
        self.include_v3_attempted_actions = include_v3_attempted_actions
        self.include_v3_budget = include_v3_budget
        self.compact_v3_budget = compact_v3_budget
        if self.single_agent_v3 and self.single_agent_v3_typed_refs:
            raise ValueError("V3 index and V3.1 typed-ref modes are exclusive")
        if self.compact_v3_budget and not self.single_agent_v3_typed_refs:
            raise ValueError("compact V3 budget requires typed-reference mode")
        if self.compact_v3_budget and not self.include_v3_budget:
            raise ValueError("compact V3 budget cannot also be omitted")
        self.semantic_memory_v3 = (
            self.single_agent_v3 or self.single_agent_v3_typed_refs
        )
        if self.single_agent_v2 and self.semantic_memory_v3:
            raise ValueError("single_agent_v2 and single_agent_v3 are exclusive")

    def build(
        self,
        query: str,
        skill: SkillDocument | str,
        state: ControllerState,
        trajectory: Sequence[StepRecord],
        *,
        scope_id: str | None = None,
    ) -> BuiltPolicyContext:
        skill_text = skill.content if isinstance(skill, SkillDocument) else skill
        if self.semantic_memory_v3:
            return self._build_v3(
                query,
                skill_text,
                state,
                trajectory,
                scope_id=scope_id,
            )
        policy_view = self.build_policy_view(
            state, trajectory, scope_id=scope_id
        )
        messages = self._base_messages(query, skill_text)
        if self.context_mode is ContextMode.APPEND_ONLY:
            messages.extend(self._append_only_history(trajectory, state))
            messages.append(
                Message(
                    role="user",
                    content=self._json(
                        {
                            "instruction": policy_view.instruction,
                            "current_state": self._append_only_state(
                                state, policy_view
                            ),
                        }
                    ),
                )
            )
        else:
            payload = (
                self._compact_v2_payload(policy_view, state, trajectory)
                if self.single_agent_v2_compact
                else policy_view.model_dump(
                    mode="json", exclude={"context_mode"}
                )
            )
            messages.append(
                Message(role="user", content=self._json(payload))
            )
        if (
            self.single_agent_v2
            and state.newest_observation is not None
            and state.newest_observation.status
            in {
                ObservationStatus.INVALID_ACTION,
                ObservationStatus.DUPLICATE_ACTION,
            }
        ):
            messages.append(
                Message(role="user", content=V2_RECOVERY_INSTRUCTION)
            )
        return BuiltPolicyContext(messages=messages, policy_view=policy_view)

    def _compact_v2_payload(
        self,
        policy_view: PolicyView,
        state: ControllerState,
        trajectory: Sequence[StepRecord],
    ) -> dict[str, Any]:
        """Serialize the same V2 state with less policy-facing UI overhead.

        The returned ``policy_view`` remains the full V2 projection so handle
        resolution and validation are unchanged.  Only the LLM-facing layout
        is compacted here.
        """

        projected = policy_view.policy_state
        return {
            "instruction": "Choose exactly one next action.",
            "state": {
                "step": projected.step,
                "policy_attempts": projected.policy_attempts,
                "last_assessment": (
                    projected.last_valid_assessment.model_dump(mode="json")
                    if projected.last_valid_assessment is not None
                    else None
                ),
                "selected_evidence": [
                    item.model_dump(mode="json")
                    for item in projected.selected_evidence
                ],
                "latest_observation": projected.latest_observation,
                "known_nodes": [
                    _compact_v2_handle(item)
                    for item in projected.actionable_handles
                ],
                "expand_sources": {
                    kind.value: handles
                    for kind, handles in projected.allowed_expansions.items()
                },
                "action_history": [
                    _compact_v2_attempt_summary(record, state)
                    for record in trajectory
                ],
                "budget": projected.budget.model_dump(mode="json"),
            },
        }

    def _build_v3(
        self,
        query: str,
        skill_text: str,
        state: ControllerState,
        trajectory: Sequence[StepRecord],
        *,
        scope_id: str | None,
    ) -> BuiltPolicyContext:
        if self.evidence_resolver is None:
            raise ValueError("V3 semantic memory requires an evidence resolver")
        if scope_id is None and (
            state.visible_entity_ids
            or state.visible_sentence_ids
            or state.visible_chunk_ids
        ):
            raise ValueError("V3 semantic memory requires scope_id")

        substrate = self.evidence_resolver.substrate
        visible_ids = (
            state.visible_entity_ids
            | state.visible_sentence_ids
            | state.visible_chunk_ids
        )
        ordered_ids = [
            node_id
            for node_id in state.semantic_memory_node_ids
            if node_id in visible_ids
        ]
        already_ordered = set(ordered_ids)
        ordered_ids.extend(
            sorted(visible_ids - already_ordered, key=_v3_node_sort_key(substrate))
        )

        covered_sentence_ids = {
            sentence.sentence_id
            for chunk_id in state.read_chunk_ids
            for sentence in substrate.sentences_by_chunk.get(chunk_id, [])
        }
        display_ids = []
        for node_id in ordered_ids:
            if node_id in state.visible_sentence_ids and (
                node_id not in state.eligible_sentence_ids
                or node_id in covered_sentence_ids
            ):
                continue
            display_ids.append(node_id)

        memory_map: dict[int, ContextNodeReference] = {}
        citations: dict[int, SentenceRef | ChunkRef] = {}
        typed_map: dict[str, TypedContextNodeReference] = {}
        semantic_memory: list[Any] = []
        if self.single_agent_v3_typed_refs:
            ref_by_stable_id: dict[str, str] = {}
            for node_id in display_ids:
                node_type = _v3_node_type(substrate, node_id)
                ref = state.handle_registry.handle_for(node_id, node_type)
                if ref is None:
                    raise ValueError(
                        "visible V3.1 node is missing its typed reference"
                    )
                ref_by_stable_id[node_id] = ref
                typed_map[ref] = TypedContextNodeReference(
                    node_type=node_type,
                    stable_id=node_id,
                    can_read=(
                        node_type == "CHUNK"
                        and node_id not in state.read_chunk_ids
                    ),
                    can_use_as_evidence=(
                        node_type == "SENTENCE"
                        or (
                            node_type == "CHUNK"
                            and node_id in state.read_chunk_ids
                        )
                    ),
                )

            for node_id in display_ids:
                ref = ref_by_stable_id[node_id]
                if node_id in state.visible_entity_ids:
                    entity = substrate.entity_by_id[node_id]
                    semantic_memory.append(
                        V31EntityMemoryItem(
                            ref=ref,
                            canonical_name=entity.canonical_name,
                            entity_type=entity.entity_type,
                        )
                    )
                    continue

                if node_id in state.visible_sentence_ids:
                    sentence = substrate.sentence_by_id[node_id]
                    chunk = substrate.chunk_by_id[sentence.chunk_id]
                    document = substrate.document_by_id[chunk.doc_id]
                    parent_ref = ref_by_stable_id.get(chunk.chunk_id)
                    if parent_ref is None:
                        raise ValueError(
                            "visible V3.1 Sentence is missing its parent Chunk"
                        )
                    semantic_memory.append(
                        V31SentenceMemoryItem(
                            ref=ref,
                            title=document.title,
                            text=sentence.text,
                            parent_chunk_ref=parent_ref,
                        )
                    )
                    continue

                chunk = substrate.chunk_by_id[node_id]
                document = substrate.document_by_id[chunk.doc_id]
                has_been_read = node_id in state.read_chunk_ids
                handle = state.node_handles.get(node_id)
                previews = (
                    [preview.text for preview in handle.previews]
                    if isinstance(handle, ChunkHandle)
                    else []
                )
                semantic_memory.append(
                    V31ChunkMemoryItem(
                        ref=ref,
                        title=document.title,
                        chunk_position=chunk.chunk_pos,
                        has_been_read=has_been_read,
                        text=chunk.text if has_been_read else None,
                        previews=previews,
                    )
                )
        else:
            memory_map = {
                index: ContextNodeReference(
                    node_type=_v3_node_type(substrate, node_id),
                    stable_id=node_id,
                    can_read=(
                        node_id in state.visible_chunk_ids
                        and node_id not in state.read_chunk_ids
                    ),
                )
                for index, node_id in enumerate(display_ids, start=1)
            }
            index_by_stable_id = {
                item.stable_id: index for index, item in memory_map.items()
            }
            next_citation = 1
            for context_index, node_id in enumerate(display_ids, start=1):
                if node_id in state.visible_entity_ids:
                    entity = substrate.entity_by_id[node_id]
                    semantic_memory.append(
                        V3EntityMemoryItem(
                            context_index=context_index,
                            canonical_name=entity.canonical_name,
                            entity_type=entity.entity_type,
                        )
                    )
                    continue

                if node_id in state.visible_sentence_ids:
                    sentence = substrate.sentence_by_id[node_id]
                    chunk = substrate.chunk_by_id[sentence.chunk_id]
                    document = substrate.document_by_id[chunk.doc_id]
                    parent_index = index_by_stable_id.get(chunk.chunk_id)
                    if parent_index is None:
                        raise ValueError(
                            "visible V3 Sentence is missing its parent Chunk"
                        )
                    citations[next_citation] = SentenceRef(id=node_id)
                    semantic_memory.append(
                        V3SentenceMemoryItem(
                            context_index=context_index,
                            citation_index=next_citation,
                            title=document.title,
                            text=sentence.text,
                            parent_chunk_context_index=parent_index,
                        )
                    )
                    next_citation += 1
                    continue

                chunk = substrate.chunk_by_id[node_id]
                document = substrate.document_by_id[chunk.doc_id]
                has_been_read = node_id in state.read_chunk_ids
                citation_index = None
                if has_been_read:
                    citation_index = next_citation
                    citations[next_citation] = ChunkRef(id=node_id)
                    next_citation += 1
                handle = state.node_handles.get(node_id)
                previews = (
                    [preview.text for preview in handle.previews]
                    if isinstance(handle, ChunkHandle)
                    else []
                )
                semantic_memory.append(
                    V3ChunkMemoryItem(
                        context_index=context_index,
                        citation_index=citation_index,
                        title=document.title,
                        chunk_position=chunk.chunk_pos,
                        has_been_read=has_been_read,
                        text=chunk.text if has_been_read else None,
                        previews=previews,
                    )
                )

        last_assessment = (
            V3EvidenceAssessment(
                status=state.last_assessment.status,
                supported_facts=list(state.last_assessment.supported_facts),
                missing_information=list(
                    state.last_assessment.missing_information
                ),
            )
            if state.last_assessment is not None
            else None
        )
        budget_view = PolicyBudgetView(
            remaining_steps=state.remaining_step_budget,
            remaining_policy_attempts=(
                state.remaining_policy_attempt_budget
            ),
            remaining_retrieved_tokens=(
                state.remaining_retrieved_token_budget
            ),
        )
        policy_state_fields = dict(
                step=state.step,
                policy_attempts=state.policy_attempts,
                last_assessment=(
                    last_assessment
                    if self.include_v3_last_assessment
                    else None
                ),
                semantic_memory=semantic_memory,
                latest_event=(
                    _v3_latest_event(state.newest_observation, substrate)
                    if self.include_v3_latest_event
                    else None
                ),
                attempted_actions=(
                    [
                        _v3_attempt_summary(record, substrate)
                        for record in trajectory
                    ]
                    if self.include_v3_attempted_actions
                    else []
                ),
                budget=(
                    _compact_v3_budget(budget_view)
                    if self.compact_v3_budget
                    else budget_view
                ),
        )
        view: V3PolicyView | V31PolicyView
        if self.single_agent_v3_typed_refs:
            view = V31PolicyView(
                context_mode=(
                    "semantic_memory_v3_2"
                    if self.compact_v3_budget
                    else "semantic_memory_v3_typed_refs"
                ),
                policy_state=V31PolicyStateView(**policy_state_fields)
            )
        else:
            view = V3PolicyView(
                policy_state=V3PolicyStateView(**policy_state_fields)
            )
        messages = self._base_messages(query, skill_text)
        payload = view.model_dump(mode="json", exclude={"context_mode"})
        if not self.include_v3_last_assessment:
            payload["policy_state"].pop("last_assessment", None)
        if not self.include_v3_latest_event:
            # The raw Observation remains in the trajectory artifact. This
            # ablation removes only its compact event projection from the
            # next Policy input.
            payload["policy_state"].pop("latest_event", None)
        if not self.include_v3_attempted_actions:
            payload["policy_state"].pop("attempted_actions", None)
        if not self.include_v3_budget:
            payload["policy_state"].pop("budget", None)
        if self.single_agent_v3_action_catalog:
            payload["policy_state"]["valid_action_catalog"] = (
                _v3_valid_action_catalog(
                    memory_map,
                    citations,
                    self.enabled_expansions,
                )
            )
        messages.append(
            Message(
                role="user",
                content=self._json(payload),
            )
        )
        if state.newest_observation is not None and (
            state.newest_observation.status
            in {
                ObservationStatus.INVALID_ACTION,
                ObservationStatus.DUPLICATE_ACTION,
            }
        ):
            recovery = (
                V31_RECOVERY_INSTRUCTION
                if self.single_agent_v3_typed_refs
                else V3_RECOVERY_INSTRUCTION
            )
            messages.append(Message(role="user", content=recovery))
        return BuiltPolicyContext(
            messages=messages,
            policy_view=view,
            reference_map=(
                TypedContextReferenceMap(typed_refs=typed_map)
                if self.single_agent_v3_typed_refs
                else ContextReferenceMap(
                    memory=memory_map,
                    citations=citations,
                )
            ),
            decision_format=policy_decision_model(
                self.enabled_expansions,
                direct_answer=True,
                semantic_memory_v3=not self.single_agent_v3_typed_refs,
                semantic_memory_v31=self.single_agent_v3_typed_refs,
            ),
        )

    def build_policy_view(
        self,
        state: ControllerState,
        trajectory: Sequence[StepRecord],
        *,
        scope_id: str | None,
    ) -> PolicyView:
        selected = self._selected_evidence(state, scope_id)
        latest = self.project_observation(
            state.newest_observation, state
        )
        represented_ids = {
            item.ref.id for item in selected
        }
        for item in selected:
            if item.parent_chunk_id:
                represented_ids.add(item.parent_chunk_id)
        represented_ids.update(_node_ids(latest))
        handles = [
            self._policy_handle(handle)
            for _, handle in sorted(
                state.node_handles.items(),
                key=lambda item: (item[1].node_type, item[0]),
            )
            if handle.id not in represented_ids
        ]
        last_assessment = (
            LastValidAssessmentView(
                status=state.last_assessment.status,
                missing_information=list(
                    state.last_assessment.missing_information
                ),
            )
            if state.last_assessment is not None
            else None
        )
        attempted = [
            AttemptedActionView(
                step=record.step,
                policy_attempt=record.policy_attempt,
                action=(
                    _project_action(record.decision.action, state)
                    if record.decision is not None
                    else None
                ),
                repaired_action=(
                    _project_action(record.repaired_decision.action, state)
                    if record.repaired_decision is not None
                    else None
                ),
                repair_code=record.repair_code,
                validation_status=record.validation_status,
                observation_status=(
                    record.observation.status
                    if record.observation is not None
                    else None
                ),
                error_code=(
                    record.observation.error_code
                    if record.observation is not None
                    else None
                ),
                message=(
                    _handle_safe_value(record.observation.message, state)
                    if record.observation is not None
                    else _handle_safe_value(record.validation_error, state)
                ),
                new_node_count=(
                    len(record.observation.novel_node_ids)
                    if record.observation is not None
                    else 0
                ),
                retrieved_tokens=(
                    record.observation.retrieved_tokens
                    if record.observation is not None
                    else 0
                ),
            )
            for record in trajectory
        ]
        return PolicyView(
            context_mode=self.context_mode,
            policy_state=PolicyStateView(
                step=state.step,
                policy_attempts=state.policy_attempts,
                last_valid_assessment=last_assessment,
                selected_evidence=selected,
                latest_observation=latest,
                actionable_handles=handles,
                allowed_expansions=self._allowed_expansions(state),
                attempted_actions=attempted,
                budget=PolicyBudgetView(
                    remaining_steps=state.remaining_step_budget,
                    remaining_policy_attempts=(
                        state.remaining_policy_attempt_budget
                    ),
                    remaining_retrieved_tokens=(
                        state.remaining_retrieved_token_budget
                    ),
                ),
            ),
        )

    def _allowed_expansions(
        self, state: ControllerState
    ) -> dict[ExpansionKind, list[str]]:
        allowed: dict[ExpansionKind, list[str]] = {}
        for kind in self.enabled_expansions:
            if kind in _ENTITY_EXPANSION_KINDS:
                source_type = "ENTITY"
            elif kind in _SENTENCE_EXPANSION_KINDS:
                source_type = "SENTENCE"
            else:
                source_type = "CHUNK"

            source_ids = []
            for handle in state.node_handles.values():
                if handle.node_type != source_type:
                    continue
                projected = self._policy_handle(handle)
                if projected.can_expand:
                    source_ids.append(handle.id)
            if source_ids:
                allowed[kind] = sorted(source_ids)
        return allowed

    def _policy_handle(self, handle: Any) -> PolicyNodeHandle:
        enabled = frozenset(self.enabled_expansions)
        if isinstance(handle, EntityHandle):
            return PolicyEntityHandle(
                id=handle.id,
                label=handle.label,
                entity_type=handle.entity_type,
                can_expand=bool(enabled & _ENTITY_EXPANSION_KINDS),
            )
        if isinstance(handle, SentenceHandle):
            return PolicySentenceHandle(
                id=handle.id,
                text=handle.text,
                parent_chunk_id=handle.parent_chunk_id,
                title=handle.title,
                can_use_as_evidence=handle.can_use_as_evidence,
                can_expand=(
                    handle.can_use_as_evidence
                    and bool(enabled & _SENTENCE_EXPANSION_KINDS)
                ),
            )
        if isinstance(handle, ChunkHandle):
            return PolicyChunkHandle(
                id=handle.id,
                title=handle.title,
                has_been_read=handle.has_been_read,
                can_read=not handle.has_been_read,
                can_expand=bool(enabled & _CHUNK_EXPANSION_KINDS),
                can_use_as_evidence=handle.can_use_as_evidence,
                previews=list(handle.previews),
            )
        raise TypeError(f"unsupported node handle: {type(handle).__name__}")

    def _selected_evidence(
        self,
        state: ControllerState,
        scope_id: str | None,
    ) -> list[PolicyEvidenceView]:
        if not state.selected_evidence_refs:
            return []
        if self.evidence_resolver is None or scope_id is None:
            raise ValueError(
                "selected evidence projection requires a resolver and scope_id"
            )
        resolved: list[PolicyEvidenceView] = []
        for ref in sorted(
            state.selected_evidence_refs,
            key=lambda item: (item.unit, item.id),
        ):
            for item in self.evidence_resolver.resolve(
                [ref], state, scope_id
            ):
                if isinstance(item.ref, SentenceRef):
                    handle = _require_handle(
                        state, item.ref.id, "SENTENCE"
                    )
                    projected_ref = SentenceRef(id=handle)
                else:
                    handle = _require_handle(
                        state, item.ref.id, "CHUNK"
                    )
                    projected_ref = ChunkRef(id=handle)
                parent_chunk_id = (
                    _require_handle(
                        state, item.parent_chunk_id, "CHUNK"
                    )
                    if item.parent_chunk_id is not None
                    else None
                )
                resolved.append(
                    PolicyEvidenceView(
                        ref=projected_ref,
                        text=item.text,
                        title=item.title,
                        parent_chunk_id=parent_chunk_id,
                        contained_sentence_ids=[
                            _require_handle(state, sentence_id, "SENTENCE")
                            for sentence_id in item.contained_sentence_ids
                        ],
                    )
                )
        return resolved

    def _base_messages(
        self, query: str, skill_text: str
    ) -> list[Message]:
        if self.single_agent_v2_compact:
            enabled = ", ".join(
                f"{kind.value}({_compact_expansion_source_type(kind)})"
                for kind in self.enabled_expansions
            ) or "none"
            protocol = (
                f"{self._action_protocol()}\n"
                f"Enabled EXPAND kinds and source types: {enabled}."
            )
        else:
            protocol = (
                f"{self._action_protocol()}\n"
                "Enabled EXPAND kinds for this run:\n"
                f"{self._expansion_guidance()}"
            )
        if self.single_agent_v3_action_catalog:
            protocol = f"{protocol}\n{V3_ACTION_CATALOG_INSTRUCTION}"
        return [
            Message(
                role="system",
                content=protocol,
            ),
            Message(
                role="system",
                content=(
                    "Current trainable retrieval skill (plain Markdown):\n\n"
                    f"{skill_text}"
                ),
            ),
            Message(role="user", content=f"Original question:\n{query}"),
        ]

    def _action_protocol(self) -> str:
        if self.single_agent_v3_typed_refs:
            return V31_ACTION_PROTOCOL
        if self.single_agent_v3:
            return V3_ACTION_PROTOCOL
        if self.single_agent_v2_compact:
            return V2_COMPACT_ACTION_PROTOCOL
        return V2_ACTION_PROTOCOL if self.single_agent_v2 else ACTION_PROTOCOL

    def project_observation(
        self,
        observation: Any,
        state: ControllerState,
    ) -> dict[str, Any] | None:
        """Project one raw observation using this run's action config."""

        return project_observation_for_policy(
            observation,
            state,
            enabled_expansions=self.enabled_expansions,
        )

    def _append_only_history(
        self,
        trajectory: Sequence[StepRecord],
        state: ControllerState,
    ) -> list[Message]:
        messages: list[Message] = []
        for record in trajectory:
            decision_payload = (
                _project_decision(record.decision, state)
                if record.decision is not None
                else {
                    "policy_error": _handle_safe_value(
                        record.validation_error, state
                    )
                    or "Policy did not produce a valid decision."
                }
            )
            messages.append(
                Message(
                    role="assistant",
                    content=self._json(
                        {
                            "step": record.step,
                            "policy_decision": decision_payload,
                        }
                    ),
                )
            )
            messages.append(
                Message(
                    role="user",
                    content=self._json(
                        {
                            "step": record.step,
                            "validation_status": record.validation_status,
                            "validation_error": _handle_safe_value(
                                record.validation_error, state
                            ),
                            "environment_observation": (
                                self.project_observation(
                                    record.observation, state
                                )
                                if record.observation is not None
                                else None
                            ),
                        }
                    ),
                )
            )
        return messages

    @staticmethod
    def _append_only_state(
        state: ControllerState,
        policy_view: PolicyView,
    ) -> dict[str, Any]:
        return {
            "step": state.step,
            "selected_evidence": [
                item.model_dump(mode="json")
                for item in policy_view.policy_state.selected_evidence
            ],
            "actionable_handles": [
                item.model_dump(mode="json")
                for item in policy_view.policy_state.actionable_handles
            ],
            "remaining_step_budget": state.remaining_step_budget,
            "remaining_retrieved_token_budget": (
                state.remaining_retrieved_token_budget
            ),
        }

    def _expansion_guidance(self) -> str:
        if not self.enabled_expansions:
            return "- (none; do not choose EXPAND)"
        if self.single_agent_v3:
            replacements = {
                "visible Entity": "known Entity memory item",
                "visible Sentence": "complete Sentence memory item",
                "visible Chunk": "known Chunk memory item",
                "parent Chunk IDs": "parent Chunk memory indices",
                "requires READ": "requires READ using its memory index",
            }
            lines = []
            for item in self.enabled_expansions:
                guidance = _EXPANSION_GUIDANCE[item]
                for old, new in replacements.items():
                    guidance = guidance.replace(old, new)
                lines.append(f"- {guidance}")
            return "\n".join(lines)
        return "\n".join(
            f"- {_EXPANSION_GUIDANCE[item]}"
            for item in self.enabled_expansions
        )

    @staticmethod
    def to_openai_input(
        messages: Sequence[Message],
    ) -> list[dict[str, str]]:
        return [message.as_openai_input() for message in messages]

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )


class AppendOnlyContextBuilder(PolicyContextBuilder):
    """Explicit baseline that preserves the original replay behavior."""

    def __init__(
        self,
        enabled_expansions: Sequence[
            ExpansionKind
        ] = DEFAULT_ENABLED_EXPANSIONS,
        *,
        evidence_resolver: EvidenceResolver | None = None,
    ) -> None:
        super().__init__(
            enabled_expansions,
            context_mode=ContextMode.APPEND_ONLY,
            evidence_resolver=evidence_resolver,
        )


def project_observation_for_policy(
    observation: Any,
    state: ControllerState,
    *,
    enabled_expansions: Sequence[
        ExpansionKind
    ] = DEFAULT_ENABLED_EXPANSIONS,
) -> dict[str, Any] | None:
    """Return the minimal handle-safe observation shown to the Policy.

    The complete :class:`Observation` remains in the trajectory for replay and
    audit.  This projection deliberately excludes Controller-owned novelty,
    visibility deltas, stable IDs, evidence-usability flags, and budget state.
    """

    if observation is None:
        return None
    projected_action = (
        _project_action(observation.action, state)
        if observation.action is not None
        else None
    )
    results = [
        _project_result(
            item,
            state=state,
            is_read=bool(
                projected_action is not None
                and projected_action.type == "READ"
            ),
            enabled_expansions=enabled_expansions,
        )
        for item in observation.results
    ]
    outcome = _policy_observation_outcome(observation)
    error = None
    if outcome not in {
        ObservationOutcome.SUCCESS,
        ObservationOutcome.EMPTY,
    }:
        code = observation.error_code or outcome.value
        message = _handle_safe_value(observation.message, state)
        error = PolicyObservationError(
            code=code,
            message=(
                str(message).strip()
                if message is not None and str(message).strip()
                else "The action was not executed successfully."
            ),
            retryable=outcome in {
                ObservationOutcome.INVALID_ACTION,
                ObservationOutcome.DUPLICATE_ACTION,
            },
            details=_policy_error_details(
                observation.action,
                state,
                code=code,
            ),
        )
    return PolicyObservation(
        action_id=observation.action_id,
        action=projected_action,
        outcome=outcome,
        results=results,
        usage=PolicyObservationUsage(
            retrieved_tokens=observation.retrieved_tokens
        ),
        error=error,
    ).model_dump(mode="json")


def _policy_observation_outcome(observation: Any) -> ObservationOutcome:
    if observation.status is ObservationStatus.OK:
        if observation.results or observation.metadata.get("finish") is True:
            return ObservationOutcome.SUCCESS
        return ObservationOutcome.EMPTY
    if observation.status is ObservationStatus.INVALID_ACTION:
        return ObservationOutcome.INVALID_ACTION
    if observation.status is ObservationStatus.DUPLICATE_ACTION:
        return ObservationOutcome.DUPLICATE_ACTION
    if (
        observation.error_code == "retrieved_token_budget_exceeded"
        or observation.metadata.get("retrieved_budget_exhausted") is True
    ):
        return ObservationOutcome.BUDGET_REJECTED
    return ObservationOutcome.TOOL_ERROR


def _policy_error_details(
    action: AgentAction | None,
    state: ControllerState,
    *,
    code: str,
) -> dict[str, Any]:
    if action is None:
        return {}
    expected_type = _expected_handle_type(action)
    if expected_type is None:
        return {}
    details: dict[str, Any] = {"expected_type": expected_type}
    received = getattr(action, "source_id", None) or getattr(
        action, "chunk_id", None
    )
    if isinstance(received, str):
        details["received_handle"] = _handle_safe_value(received, state)
    if code in {
        "unknown_handle",
        "handle_type_mismatch",
        "source_not_visible",
        "duplicate_action",
    }:
        details["available_handles"] = sorted(
            handle.id
            for handle in state.node_handles.values()
            if handle.node_type == expected_type
        )[:10]
    return details


def _expected_handle_type(action: AgentAction) -> str | None:
    if action.type == "READ":
        return "CHUNK"
    if action.type != "EXPAND":
        return None
    if action.kind in _ENTITY_EXPANSION_KINDS:
        return "ENTITY"
    if action.kind in _SENTENCE_EXPANSION_KINDS:
        return "SENTENCE"
    if action.kind in _CHUNK_EXPANSION_KINDS:
        return "CHUNK"
    return None


def _project_result(
    result: Mapping[str, Any],
    *,
    state: ControllerState,
    is_read: bool,
    enabled_expansions: Sequence[ExpansionKind],
) -> dict[str, Any]:
    projected = {
        key: _project_result_value(
            value,
            state=state,
            enabled_expansions=enabled_expansions,
        )
        for key, value in result.items()
        if key
        not in {
            "content_read",
            "evidence_eligible",
            *_DOCUMENT_ID_KEYS,
        }
    }
    if is_read and "sentences" in projected:
        projected.pop("text", None)
    return projected


def _project_result_value(
    value: Any,
    *,
    state: ControllerState,
    enabled_expansions: Sequence[ExpansionKind],
) -> Any:
    if isinstance(value, list):
        return [
            _project_result_value(
                item,
                state=state,
                enabled_expansions=enabled_expansions,
            )
            for item in value
        ]
    if not isinstance(value, Mapping):
        return _handle_safe_value(value, state)
    projected = {
        key: _project_result_value(
            child,
            state=state,
            enabled_expansions=enabled_expansions,
        )
        for key, child in value.items()
        if key
        not in {
            "content_read",
            "evidence_eligible",
            *_DOCUMENT_ID_KEYS,
        }
    }
    return projected


def _project_action(
    action: AgentAction,
    state: ControllerState,
) -> AgentAction:
    payload = _handle_safe_value(action.model_dump(mode="json"), state)
    return _ACTION_ADAPTER.validate_python(payload)


def _project_decision(
    decision: PolicyDecision,
    state: ControllerState,
) -> dict[str, Any]:
    return _handle_safe_value(decision.model_dump(mode="json"), state)


def _handle_safe_value(value: Any, state: ControllerState) -> Any:
    """Recursively replace registered stable node IDs with short handles."""

    if isinstance(value, Mapping):
        return {
            key: _handle_safe_value(child, state)
            for key, child in value.items()
            if key not in _DOCUMENT_ID_KEYS
        }
    if isinstance(value, list):
        return [_handle_safe_value(item, state) for item in value]
    if isinstance(value, tuple):
        return [_handle_safe_value(item, state) for item in value]
    if not isinstance(value, str):
        return value
    exact = state.handle_registry.handle_for(value)
    if exact is not None:
        return exact
    projected = value
    for stable_id, handle in sorted(
        state.handle_registry.stable_id_to_handle.items(),
        key=lambda item: (-len(item[0]), item[0]),
    ):
        projected = projected.replace(stable_id, handle)
    return projected


def _require_handle(
    state: ControllerState,
    stable_id: str,
    expected_type: str,
) -> str:
    handle = state.handle_registry.handle_for(stable_id, expected_type)
    if handle is None:
        raise ValueError(
            f"missing {expected_type} handle for visible stable node"
        )
    return handle


def _v3_node_sort_key(substrate: Any):
    type_rank = {"ENTITY": 0, "SENTENCE": 1, "CHUNK": 2}

    def key(node_id: str) -> tuple[int, str]:
        return (type_rank[_v3_node_type(substrate, node_id)], node_id)

    return key


def _v3_node_type(substrate: Any, node_id: str) -> str:
    if node_id in substrate.entity_by_id:
        return "ENTITY"
    if node_id in substrate.sentence_by_id:
        return "SENTENCE"
    if node_id in substrate.chunk_by_id:
        return "CHUNK"
    raise ValueError("semantic memory contains an unknown stable node")


def _v3_node_label(substrate: Any, node_id: str) -> dict[str, Any]:
    if node_id in substrate.entity_by_id:
        entity = substrate.entity_by_id[node_id]
        return {
            "node_type": "ENTITY",
            "canonical_name": entity.canonical_name,
        }
    if node_id in substrate.sentence_by_id:
        sentence = substrate.sentence_by_id[node_id]
        chunk = substrate.chunk_by_id[sentence.chunk_id]
        document = substrate.document_by_id[chunk.doc_id]
        return {
            "node_type": "SENTENCE",
            "title": document.title,
            "text": sentence.text,
        }
    if node_id in substrate.chunk_by_id:
        chunk = substrate.chunk_by_id[node_id]
        document = substrate.document_by_id[chunk.doc_id]
        return {
            "node_type": "CHUNK",
            "title": document.title,
            "chunk_position": chunk.chunk_pos,
        }
    return {"node_type": "UNKNOWN"}


def _v3_action_summary(action: Any, substrate: Any) -> dict[str, Any] | None:
    if action is None:
        return None
    if isinstance(action, SearchAction):
        return {
            "type": "SEARCH",
            "method": action.method.value,
            "target": action.target.value,
            "query": action.query,
        }
    if isinstance(action, ExpandAction):
        return {
            "type": "EXPAND",
            "kind": action.kind.value,
            "source": _v3_node_label(substrate, action.source_id),
            "query": action.query,
            "direction": (
                action.direction.value if action.direction is not None else None
            ),
        }
    if isinstance(action, V3ExpandAction):
        return {
            "type": "EXPAND",
            "kind": action.kind.value,
            "query": action.query,
            "direction": (
                action.direction.value if action.direction is not None else None
            ),
        }
    if isinstance(action, ReadAction):
        return {
            "type": "READ",
            "chunk": _v3_node_label(substrate, action.chunk_id),
        }
    if isinstance(action, V3ReadAction):
        return {"type": "READ"}
    if isinstance(action, FinishAction):
        return {
            "type": "FINISH",
            "answer": action.answer,
            "citation_count": len(action.evidence_refs),
        }
    if isinstance(action, V3FinishAction):
        return {
            "type": "FINISH",
            "answer": action.answer,
            "citation_count": len(action.citations),
        }
    return {"type": str(getattr(action, "type", type(action).__name__))}


def _v3_safe_message(message: str | None, substrate: Any) -> str | None:
    if message is None:
        return None
    projected = message
    for node_id in sorted(
        (
            *substrate.entity_by_id,
            *substrate.sentence_by_id,
            *substrate.chunk_by_id,
        ),
        key=len,
        reverse=True,
    ):
        if node_id not in projected:
            continue
        label = _v3_node_label(substrate, node_id)
        replacement = (
            label.get("canonical_name")
            or label.get("title")
            or label.get("node_type")
            or "known node"
        )
        projected = projected.replace(node_id, str(replacement))
    return projected


def _v3_latest_event(observation: Any, substrate: Any) -> dict[str, Any] | None:
    if observation is None:
        return None
    error = None
    if observation.error_code or observation.message:
        error = {
            "code": observation.error_code or "tool_error",
            "message": _v3_safe_message(observation.message, substrate)
            or "The action failed.",
            "retryable": observation.status
            in {
                ObservationStatus.INVALID_ACTION,
                ObservationStatus.DUPLICATE_ACTION,
            },
        }
    return {
        "action_id": observation.action_id,
        "action": _v3_action_summary(observation.action, substrate),
        "outcome": _policy_observation_outcome(observation).value,
        "usage": {"retrieved_tokens": observation.retrieved_tokens},
        "error": error,
    }


def project_v3_observation_for_policy(
    observation: Any,
    substrate: Any,
) -> dict[str, Any] | None:
    """Return the V3 event metadata shown beside materialized memory."""

    return _v3_latest_event(observation, substrate)


def _v3_attempt_summary(record: StepRecord, substrate: Any) -> dict[str, Any]:
    action = (
        record.resolved_decision.action
        if record.resolved_decision is not None
        else (
            record.decision.action if record.decision is not None else None
        )
    )
    return {
        "policy_attempt": record.policy_attempt,
        "action": _v3_action_summary(action, substrate),
        "outcome": (
            _policy_observation_outcome(record.observation).value
            if record.observation is not None
            else "tool_error"
        ),
        "error_code": (
            record.observation.error_code
            if record.observation is not None
            else None
        ),
    }


def _compact_v3_budget(budget: PolicyBudgetView) -> str:
    return (
        f"Budget: {budget.remaining_steps} steps, "
        f"{budget.remaining_policy_attempts} attempts, "
        f"{budget.remaining_retrieved_tokens} retrieval tokens left"
    )


def _compact_v2_handle(handle: PolicyNodeHandle) -> dict[str, Any]:
    """Return semantic handle content without repeated capability flags."""

    if isinstance(handle, PolicyEntityHandle):
        return {
            "type": "ENTITY",
            "id": handle.id,
            "name": handle.label,
            **(
                {"entity_type": handle.entity_type}
                if handle.entity_type is not None
                else {}
            ),
        }
    if isinstance(handle, PolicySentenceHandle):
        return {
            "type": "SENTENCE",
            "id": handle.id,
            "title": handle.title,
            "text": handle.text,
            "parent_chunk": handle.parent_chunk_id,
            "eligible_evidence": handle.can_use_as_evidence,
        }
    if isinstance(handle, PolicyChunkHandle):
        return {
            "type": "CHUNK",
            "id": handle.id,
            "title": handle.title,
            "read": handle.has_been_read,
            "eligible_evidence": handle.can_use_as_evidence,
            "previews": [
                preview.model_dump(mode="json") for preview in handle.previews
            ],
        }
    raise TypeError(f"unsupported policy handle: {type(handle).__name__}")


def _compact_v2_attempt_summary(
    record: StepRecord,
    state: ControllerState,
) -> dict[str, Any]:
    """Keep semantic novelty history while omitting the verbose audit record."""

    decision = (
        record.resolved_decision
        or record.repaired_decision
        or record.decision
    )
    action = (
        _project_action(decision.action, state).model_dump(
            mode="json", exclude_none=True
        )
        if decision is not None
        else None
    )
    return {
        "attempt": record.policy_attempt,
        "action": action,
        "outcome": (
            _policy_observation_outcome(record.observation).value
            if record.observation is not None
            else record.validation_status.value
        ),
        "error_code": (
            record.observation.error_code
            if record.observation is not None
            else None
        ),
    }


def _compact_expansion_source_type(kind: ExpansionKind) -> str:
    if kind in _ENTITY_EXPANSION_KINDS:
        return "ENTITY"
    if kind in _SENTENCE_EXPANSION_KINDS:
        return "SENTENCE"
    return "CHUNK"


def _v3_valid_action_catalog(
    memory_map: Mapping[int, ContextNodeReference],
    citations: Mapping[int, SentenceRef | ChunkRef],
    enabled_expansions: Sequence[ExpansionKind],
) -> dict[str, Any]:
    """Enumerate structurally executable V3 actions for one frozen snapshot."""

    search = [
        {
            "type": "SEARCH",
            "method": method,
            "target": target,
            "top_k": 5,
        }
        for method, target in (
            ("LEXICAL", "ENTITY"),
            ("BM25", "SENTENCE"),
            ("BM25", "CHUNK"),
            ("DENSE", "ENTITY"),
            ("DENSE", "SENTENCE"),
            ("DENSE", "CHUNK"),
        )
    ]
    expand: list[dict[str, Any]] = []
    for kind in enabled_expansions:
        source_type = _compact_expansion_source_type(kind)
        for context_index, node in sorted(memory_map.items()):
            if node.node_type != source_type:
                continue
            template: dict[str, Any] = {
                "type": "EXPAND",
                "kind": kind.value,
                "source_context_index": context_index,
                "top_k": 5,
            }
            if kind is ExpansionKind.CHUNK_ADJACENT_CHUNK:
                template["valid_directions"] = ["PREV", "NEXT", "BOTH"]
            else:
                template["direction"] = None
            expand.append(template)

    read = [
        {"type": "READ", "chunk_context_index": context_index}
        for context_index, node in sorted(memory_map.items())
        if node.node_type == "CHUNK" and node.can_read
    ]
    finish = (
        [
            {
                "type": "FINISH",
                "available_citation_indices": sorted(citations),
            }
        ]
        if citations
        else []
    )
    return {
        "SEARCH": search,
        "EXPAND": expand,
        "READ": read,
        "FINISH": finish,
    }


def _node_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, list):
        for item in value:
            found.update(_node_ids(item))
        return found
    if not isinstance(value, Mapping):
        return found
    for key, child in value.items():
        if key in _NODE_ID_KEYS and isinstance(child, str):
            found.add(child)
        elif key in {
            "node_ids",
            "bridge_sentence_ids",
            "bridge_chunk_ids",
        } and isinstance(child, list):
            found.update(item for item in child if isinstance(item, str))
        else:
            found.update(_node_ids(child))
    return found
