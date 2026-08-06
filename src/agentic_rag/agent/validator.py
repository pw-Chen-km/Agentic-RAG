"""State-dependent validation for policy decisions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Any

from agentic_rag.agent.handle_resolution import (
    HandleResolutionError,
    resolve_evidence_ref,
    resolve_policy_decision,
)
from agentic_rag.agent.models import (
    AssessmentStatus,
    ActionSelection,
    ChunkHandle,
    ChunkRef,
    ControllerState,
    EntityHandle,
    EvidenceRef,
    ExpandAction,
    ExpansionDirection,
    ExpansionKind,
    FinishAction,
    PolicyDecision,
    ReadAction,
    SearchAction,
    SentenceHandle,
    SentenceRef,
    VALID_SEARCH_PAIRS,
    action_signature,
)
from agentic_rag.storage import Substrate


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    code: str | None = None
    message: str | None = None
    signature: str | None = None
    resolved_decision: PolicyDecision | None = None

    @classmethod
    def valid(
        cls, signature: str, resolved_decision: PolicyDecision
    ) -> "ValidationResult":
        return cls(
            ok=True,
            signature=signature,
            resolved_decision=resolved_decision,
        )

    @classmethod
    def invalid(
        cls,
        code: str,
        message: str,
        *,
        signature: str | None = None,
        resolved_decision: PolicyDecision | None = None,
    ) -> "ValidationResult":
        return cls(
            ok=False,
            code=code,
            message=message,
            signature=signature,
            resolved_decision=resolved_decision,
        )


@dataclass(frozen=True, slots=True)
class EvidenceSelectionValidation:
    """Stage-1 evidence validation with controller-internal resolved refs."""

    ok: bool
    code: str | None = None
    message: str | None = None
    resolved_refs: tuple[EvidenceRef, ...] = ()

    @classmethod
    def valid(
        cls, resolved_refs: Sequence[EvidenceRef]
    ) -> "EvidenceSelectionValidation":
        return cls(ok=True, resolved_refs=tuple(resolved_refs))

    @classmethod
    def invalid(
        cls, code: str, message: str
    ) -> "EvidenceSelectionValidation":
        return cls(ok=False, code=code, message=message)


_ENTITY_SOURCE_KINDS = frozenset(
    {
        ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
        ExpansionKind.ENTITY_CO_OCCURS_ENTITY_SENTENCE,
        ExpansionKind.ENTITY_MENTIONED_IN_CHUNK,
        ExpansionKind.ENTITY_CO_OCCURS_ENTITY_CHUNK,
    }
)
_SENTENCE_SOURCE_KINDS = frozenset(
    {ExpansionKind.SENTENCE_MENTIONS_ENTITY}
)
_CHUNK_SOURCE_KINDS = frozenset(
    {
        ExpansionKind.CHUNK_ADJACENT_CHUNK,
        ExpansionKind.CHUNK_CONTAINS_SENTENCE,
        ExpansionKind.CHUNK_MENTIONS_ENTITY,
    }
)


class LegalActionCatalog:
    """Build handle-safe recovery options from the validator's predicates.

    The catalog deliberately exposes policy handles and semantic labels only.
    Stable substrate IDs are used internally to evaluate visibility, scope,
    eligibility, and duplicate state, but never occur in the returned payload.
    """

    _ACTION_TYPES = frozenset({"SEARCH", "EXPAND", "READ", "FINISH"})

    def __init__(
        self,
        substrate: Substrate,
        enabled_expansions: tuple[ExpansionKind, ...],
    ) -> None:
        self.substrate = substrate
        self.enabled_expansions = frozenset(enabled_expansions)

    def options(
        self,
        action_type: object,
        state: ControllerState,
        scope_id: str,
        *,
        selected_evidence_refs: Sequence[EvidenceRef] = (),
    ) -> list[dict[str, Any]]:
        """Return state-dependent recovery choices for one action family."""

        self.substrate.require_scope(scope_id)
        normalized = self._normalize_action_type(action_type)
        if normalized == "SEARCH":
            return self._search_options()
        if normalized == "EXPAND":
            return self._expand_options(state, scope_id)
        if normalized == "READ":
            return self._read_options(state, scope_id)
        return self._finish_options(
            selected_evidence_refs, state, scope_id
        )

    def validate_expand(
        self,
        action: ExpandAction,
        state: ControllerState,
        scope_id: str,
    ) -> tuple[str, str] | None:
        """Apply the same structural predicate used by EXPAND options."""

        if action.kind not in self.enabled_expansions:
            return (
                "disabled_expansion",
                f"Expansion {action.kind.value} is disabled for this run",
            )

        if action.kind in _ENTITY_SOURCE_KINDS:
            return self._source_error(
                action.source_id,
                visible_ids=state.visible_entity_ids,
                scope_ids=self.substrate.entity_ids_by_scope[scope_id],
                source_name="Entity",
            )
        if action.kind in _SENTENCE_SOURCE_KINDS:
            source_error = self._source_error(
                action.source_id,
                visible_ids=state.visible_sentence_ids,
                scope_ids=self.substrate.sentence_ids_by_scope[scope_id],
                source_name="Sentence",
            )
            if source_error is not None:
                return source_error
            if action.source_id not in state.eligible_sentence_ids:
                return (
                    "source_not_complete",
                    (
                        "Sentence source is only visible as a navigation preview; "
                        "READ its parent Chunk before expanding it: "
                        f"{action.source_id}"
                    ),
                )
            return None
        if action.kind in _CHUNK_SOURCE_KINDS:
            return self._source_error(
                action.source_id,
                visible_ids=state.visible_chunk_ids,
                scope_ids=self.substrate.chunk_ids_by_scope[scope_id],
                source_name="Chunk",
            )
        return (
            "unsupported_expansion",
            f"Unknown expansion kind: {action.kind}",
        )

    def validate_read(
        self,
        action: ReadAction,
        state: ControllerState,
        scope_id: str,
    ) -> tuple[str, str] | None:
        """Apply the same structural predicate used by READ options."""

        source_error = self._source_error(
            action.chunk_id,
            visible_ids=state.visible_chunk_ids,
            scope_ids=self.substrate.chunk_ids_by_scope[scope_id],
            source_name="Chunk",
            subject="Chunk",
        )
        if source_error is not None:
            return source_error
        if action.chunk_id in state.read_chunk_ids:
            return (
                "chunk_already_read",
                f"Chunk has already been READ: {action.chunk_id}",
            )
        return None

    def validate_selected_evidence(
        self,
        selected_evidence_refs: Sequence[EvidenceRef],
        state: ControllerState,
        scope_id: str,
    ) -> EvidenceSelectionValidation:
        """Resolve and validate Stage-1 evidence without mutating state."""

        self.substrate.require_scope(scope_id)
        resolved_refs: list[EvidenceRef] = []
        seen: set[tuple[str, str]] = set()
        for policy_ref in selected_evidence_refs:
            try:
                resolved_ref = resolve_evidence_ref(
                    policy_ref, state, self.substrate
                )
            except HandleResolutionError as exc:
                return EvidenceSelectionValidation.invalid(exc.code, str(exc))

            key = (resolved_ref.unit, resolved_ref.id)
            if key in seen:
                return EvidenceSelectionValidation.invalid(
                    "selected_evidence_duplicate",
                    "selected_evidence_refs must not contain duplicates",
                )
            seen.add(key)
            error = self.selected_evidence_error(
                resolved_ref,
                state,
                scope_id,
                display_id=policy_ref.id,
            )
            if error is not None:
                return EvidenceSelectionValidation.invalid(*error)
            resolved_refs.append(resolved_ref)
        return EvidenceSelectionValidation.valid(resolved_refs)

    def selected_evidence_error(
        self,
        ref: EvidenceRef,
        state: ControllerState,
        scope_id: str,
        *,
        display_id: str | None = None,
    ) -> tuple[str, str] | None:
        """Return the shared visibility/scope/eligibility error for one ref."""

        shown_id = display_id or ref.id
        if isinstance(ref, SentenceRef):
            if ref.id not in state.visible_sentence_ids:
                return (
                    "selected_evidence_not_visible",
                    f"Selected SentenceRef is not visible: {shown_id}",
                )
            if ref.id not in self.substrate.sentence_ids_by_scope[scope_id]:
                return (
                    "selected_evidence_out_of_scope",
                    f"Selected SentenceRef is outside scope {scope_id}: {shown_id}",
                )
            if ref.id not in state.eligible_sentence_ids:
                return (
                    "selected_evidence_not_eligible",
                    (
                        "Selected SentenceRef has not been shown as complete "
                        f"evidence: {shown_id}"
                    ),
                )
            return None

        if ref.id not in state.visible_chunk_ids:
            return (
                "selected_evidence_not_visible",
                f"Selected ChunkRef is not visible: {shown_id}",
            )
        if ref.id not in self.substrate.chunk_ids_by_scope[scope_id]:
            return (
                "selected_evidence_out_of_scope",
                f"Selected ChunkRef is outside scope {scope_id}: {shown_id}",
            )
        if ref.id not in state.read_chunk_ids:
            return (
                "selected_evidence_not_eligible",
                f"Selected ChunkRef must already be READ: {shown_id}",
            )
        return None

    def _search_options(self) -> list[dict[str, Any]]:
        return [
            {
                "method": method.value,
                "target": target.value,
                "top_k": 5,
            }
            for method, target in sorted(
                VALID_SEARCH_PAIRS,
                key=lambda pair: (pair[0].value, pair[1].value),
            )
        ]

    def _expand_options(
        self, state: ControllerState, scope_id: str
    ) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        for kind in sorted(self.enabled_expansions, key=lambda item: item.value):
            if kind in _ENTITY_SOURCE_KINDS:
                source_ids = state.visible_entity_ids
                source_type = "ENTITY"
            elif kind in _SENTENCE_SOURCE_KINDS:
                source_ids = state.visible_sentence_ids
                source_type = "SENTENCE"
            elif kind in _CHUNK_SOURCE_KINDS:
                source_ids = state.visible_chunk_ids
                source_type = "CHUNK"
            else:
                continue

            for source_id in sorted(source_ids):
                probe_directions: tuple[ExpansionDirection | None, ...]
                if kind is ExpansionKind.CHUNK_ADJACENT_CHUNK:
                    probe_directions = tuple(ExpansionDirection)
                else:
                    probe_directions = (None,)
                for direction in probe_directions:
                    probe = ExpandAction(
                        kind=kind,
                        source_id=source_id,
                        direction=direction,
                    )
                    if self.validate_expand(probe, state, scope_id) is not None:
                        continue
                    source = self._semantic_node(
                        source_id, source_type, state
                    )
                    if source is None:
                        continue
                    option: dict[str, Any] = {
                        "kind": kind.value,
                        "source_id": source["id"],
                        "source": source,
                        "direction": (
                            direction.value if direction is not None else None
                        ),
                        "top_k": 5,
                    }
                    options.append(option)
        return options

    def _read_options(
        self, state: ControllerState, scope_id: str
    ) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        for chunk_id in sorted(state.visible_chunk_ids):
            probe = ReadAction(chunk_id=chunk_id)
            if self.validate_read(probe, state, scope_id) is not None:
                continue
            if action_signature(probe) in state.action_signatures:
                continue
            chunk = self._semantic_node(chunk_id, "CHUNK", state)
            if chunk is None:
                continue
            options.append({"chunk_id": chunk["id"], "chunk": chunk})
        return options

    def _finish_options(
        self,
        selected_evidence_refs: Sequence[EvidenceRef],
        state: ControllerState,
        scope_id: str,
    ) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for policy_ref in selected_evidence_refs:
            try:
                resolved_ref = resolve_evidence_ref(
                    policy_ref, state, self.substrate
                )
            except HandleResolutionError:
                continue
            key = (resolved_ref.unit, resolved_ref.id)
            if key in seen:
                continue
            seen.add(key)
            if self.selected_evidence_error(
                resolved_ref, state, scope_id
            ) is not None:
                continue
            node_type = (
                "SENTENCE"
                if isinstance(resolved_ref, SentenceRef)
                else "CHUNK"
            )
            evidence = self._semantic_node(
                resolved_ref.id, node_type, state, include_text=True
            )
            if evidence is None:
                continue
            option_ref = {
                "unit": resolved_ref.unit,
                "id": evidence["id"],
            }
            options.append(
                {"evidence_ref": option_ref, "evidence": evidence}
            )
        return options

    def _semantic_node(
        self,
        stable_id: str,
        node_type: str,
        state: ControllerState,
        *,
        include_text: bool = False,
    ) -> dict[str, Any] | None:
        policy_id = state.handle_registry.handle_for(stable_id, node_type)
        if policy_id is None:
            return None
        stored = state.node_handles.get(stable_id)

        if node_type == "ENTITY":
            entity = self.substrate.entity_by_id.get(stable_id)
            if entity is None:
                return None
            return {
                "node_type": "ENTITY",
                "id": policy_id,
                "label": (
                    stored.label
                    if isinstance(stored, EntityHandle)
                    else entity.canonical_name
                ),
                "entity_type": (
                    stored.entity_type
                    if isinstance(stored, EntityHandle)
                    else entity.entity_type
                ),
            }

        if node_type == "SENTENCE":
            sentence = self.substrate.sentence_by_id.get(stable_id)
            if sentence is None:
                return None
            chunk = self.substrate.chunk_by_id[sentence.chunk_id]
            document = self.substrate.document_by_id[chunk.doc_id]
            parent_handle = state.handle_registry.handle_for(
                chunk.chunk_id, "CHUNK"
            )
            payload: dict[str, Any] = {
                "node_type": "SENTENCE",
                "id": policy_id,
                "text": (
                    stored.text
                    if isinstance(stored, SentenceHandle) and not include_text
                    else sentence.text
                ),
                "title": (
                    stored.title
                    if isinstance(stored, SentenceHandle)
                    else document.title
                ),
            }
            if parent_handle is not None:
                payload["parent_chunk_id"] = parent_handle
            return payload

        chunk = self.substrate.chunk_by_id.get(stable_id)
        if chunk is None:
            return None
        document = self.substrate.document_by_id[chunk.doc_id]
        payload = {
            "node_type": "CHUNK",
            "id": policy_id,
            "title": (
                stored.title
                if isinstance(stored, ChunkHandle)
                else document.title
            ),
            "has_been_read": stable_id in state.read_chunk_ids,
            "chunk_position": chunk.chunk_pos,
        }
        if include_text and stable_id in state.read_chunk_ids:
            payload["text"] = chunk.text
        previews = self._safe_previews(stored, state)
        if previews:
            payload["previews"] = previews
        return payload

    def _safe_previews(
        self, stored: object, state: ControllerState
    ) -> list[dict[str, str]]:
        if not isinstance(stored, ChunkHandle):
            return []
        previews: list[dict[str, str]] = []
        for preview in stored.previews:
            stable_sentence_id = state.handle_registry.stable_id_for(
                preview.sentence_id, "SENTENCE"
            )
            if stable_sentence_id is None:
                continue
            policy_id = state.handle_registry.handle_for(
                stable_sentence_id, "SENTENCE"
            )
            if policy_id is None:
                continue
            previews.append({"id": policy_id, "text": preview.text})
        return previews

    @staticmethod
    def _source_error(
        source_id: str,
        *,
        visible_ids: set[str],
        scope_ids: set[str],
        source_name: str,
        subject: str | None = None,
    ) -> tuple[str, str] | None:
        label = subject or f"{source_name} source"
        if source_id not in visible_ids:
            return (
                "source_not_visible",
                f"{label} is not visible to the agent: {source_id}",
            )
        if source_id not in scope_ids:
            return (
                "source_out_of_scope",
                f"{label} is outside the active scope: {source_id}",
            )
        return None

    @classmethod
    def _normalize_action_type(cls, action_type: object) -> str:
        raw = getattr(action_type, "value", action_type)
        normalized = str(raw).upper()
        if normalized not in cls._ACTION_TYPES:
            raise ValueError(f"Unsupported action type: {action_type}")
        return normalized


class DecisionValidator:
    """Validate policy choices without consulting gold evidence."""

    def __init__(
        self,
        substrate: Substrate,
        enabled_expansions: tuple[ExpansionKind, ...],
    ) -> None:
        self.substrate = substrate
        self.enabled_expansions = frozenset(enabled_expansions)
        self.legal_action_catalog = LegalActionCatalog(
            substrate, enabled_expansions
        )

    def legal_action_options(
        self,
        action_type: object,
        state: ControllerState,
        scope_id: str,
        *,
        selected_evidence_refs: Sequence[EvidenceRef] = (),
    ) -> list[dict[str, Any]]:
        """Return handle-safe recovery options for a V2-2 action family."""

        return self.legal_action_catalog.options(
            action_type,
            state,
            scope_id,
            selected_evidence_refs=selected_evidence_refs,
        )

    def validate_selected_evidence(
        self,
        selected_evidence_refs: Sequence[EvidenceRef],
        state: ControllerState,
        scope_id: str,
    ) -> EvidenceSelectionValidation:
        """Validate and resolve V2-2 Stage-1 evidence selection."""

        return self.legal_action_catalog.validate_selected_evidence(
            selected_evidence_refs, state, scope_id
        )

    def validate_action_selection(
        self,
        selection: ActionSelection,
        state: ControllerState,
        scope_id: str,
    ) -> EvidenceSelectionValidation:
        """Validate V2-2 intent semantics and pending evidence together."""

        handles = _referenced_policy_handles(
            selection.action_intent, state
        )
        if handles and not _intent_has_visible_semantics(
            selection.action_intent, state, self.substrate
        ):
            return EvidenceSelectionValidation.invalid(
                "action_intent_uses_handle",
                (
                    "action_intent must use the node's real name or meaning, "
                    f"not only the structural handle {handles[0]}"
                ),
            )
        return self.validate_selected_evidence(
            selection.selected_evidence_refs, state, scope_id
        )

    def validate(
        self,
        decision: PolicyDecision,
        state: ControllerState,
        scope_id: str,
        *,
        enforce_semantic_handles: bool = False,
    ) -> ValidationResult:
        self.substrate.require_scope(scope_id)
        if enforce_semantic_handles:
            semantic_error = self._validate_v22_semantic_fields(
                decision, state
            )
            if semantic_error is not None:
                return ValidationResult.invalid(*semantic_error)
        try:
            resolved_decision = resolve_policy_decision(
                decision, state, self.substrate
            )
        except HandleResolutionError as exc:
            return ValidationResult.invalid(exc.code, str(exc))

        action = resolved_decision.action
        signature = action_signature(action)

        if signature in state.action_signatures:
            if isinstance(action, ReadAction):
                read_error = self._validate_read(action, state, scope_id)
                if (
                    read_error is not None
                    and read_error[0] == "chunk_already_read"
                ):
                    return ValidationResult.invalid(
                        *read_error,
                        signature=signature,
                        resolved_decision=resolved_decision,
                    )
            return ValidationResult.invalid(
                "duplicate_action",
                "This exact action has already been attempted",
                signature=signature,
                resolved_decision=resolved_decision,
            )

        assessment_error = self._validate_assessment_refs(
            resolved_decision, state, scope_id
        )
        if assessment_error is not None:
            return ValidationResult.invalid(
                *assessment_error,
                signature=signature,
                resolved_decision=resolved_decision,
            )

        if isinstance(action, FinishAction):
            if (
                resolved_decision.assessment.status
                is not AssessmentStatus.SUFFICIENT
            ):
                return ValidationResult.invalid(
                    "finish_requires_sufficient",
                    "FINISH requires assessment.status=SUFFICIENT",
                    signature=signature,
                    resolved_decision=resolved_decision,
                )
            evidence_error = self._validate_finish_refs(
                resolved_decision, state, scope_id
            )
            if evidence_error is not None:
                return ValidationResult.invalid(
                    *evidence_error,
                    signature=signature,
                    resolved_decision=resolved_decision,
                )
        elif (
            resolved_decision.assessment.status
            is AssessmentStatus.SUFFICIENT
        ):
            return ValidationResult.invalid(
                "sufficient_requires_finish",
                "A SUFFICIENT assessment must choose FINISH",
                signature=signature,
                resolved_decision=resolved_decision,
            )
        elif isinstance(action, ExpandAction):
            expansion_error = self._validate_expand(action, state, scope_id)
            if expansion_error is not None:
                return ValidationResult.invalid(
                    *expansion_error,
                    signature=signature,
                    resolved_decision=resolved_decision,
                )
        elif isinstance(action, ReadAction):
            read_error = self._validate_read(action, state, scope_id)
            if read_error is not None:
                return ValidationResult.invalid(
                    *read_error,
                    signature=signature,
                    resolved_decision=resolved_decision,
                )
        elif not isinstance(action, SearchAction):
            return ValidationResult.invalid(
                "unsupported_action",
                f"Unsupported action type: {type(action).__name__}",
                signature=signature,
                resolved_decision=resolved_decision,
            )

        return ValidationResult.valid(signature, resolved_decision)

    @staticmethod
    def _validate_v22_semantic_fields(
        decision: PolicyDecision,
        state: ControllerState,
    ) -> tuple[str, str] | None:
        action = decision.action
        semantic_fields: list[tuple[str, str]] = []
        if isinstance(action, SearchAction):
            semantic_fields.append(("query", action.query))
        elif isinstance(action, ExpandAction) and action.query is not None:
            semantic_fields.append(("query", action.query))
        elif isinstance(action, FinishAction) and action.answer is not None:
            semantic_fields.append(("answer", action.answer))
        for field_name, text in semantic_fields:
            handles = _referenced_policy_handles(text, state)
            if handles:
                return (
                    "semantic_field_uses_handle",
                    (
                        f"{field_name} must use real names or meaning, not "
                        f"the structural handle {handles[0]}"
                    ),
                )
        return None

    def _validate_expand(
        self,
        action: ExpandAction,
        state: ControllerState,
        scope_id: str,
    ) -> tuple[str, str] | None:
        return self.legal_action_catalog.validate_expand(
            action, state, scope_id
        )

    def _validate_read(
        self,
        action: ReadAction,
        state: ControllerState,
        scope_id: str,
    ) -> tuple[str, str] | None:
        return self.legal_action_catalog.validate_read(action, state, scope_id)

    def _validate_assessment_refs(
        self,
        decision: PolicyDecision,
        state: ControllerState,
        scope_id: str,
    ) -> tuple[str, str] | None:
        for ref in decision.assessment.selected_evidence_refs:
            error = self.legal_action_catalog.selected_evidence_error(
                ref, state, scope_id
            )
            if error is not None:
                return error
        return None

    def _validate_finish_refs(
        self,
        decision: PolicyDecision,
        state: ControllerState,
        scope_id: str,
    ) -> tuple[str, str] | None:
        action = decision.action
        if not isinstance(action, FinishAction):
            raise TypeError("_validate_finish_refs requires a FINISH decision")
        selected = {
            (ref.unit, ref.id)
            for ref in decision.assessment.selected_evidence_refs
        }
        for ref in action.evidence_refs:
            if (ref.unit, ref.id) not in selected:
                return (
                    "finish_evidence_not_selected",
                    (
                        "Every FINISH evidence ref must occur in this turn's "
                        f"selected_evidence_refs: {ref.unit}:{ref.id}"
                    ),
                )
            if isinstance(ref, SentenceRef):
                if ref.id not in state.eligible_sentence_ids:
                    return (
                        "evidence_not_eligible",
                        f"Sentence has not been shown as complete evidence: {ref.id}",
                    )
                if ref.id not in self.substrate.sentence_ids_by_scope[scope_id]:
                    return (
                        "evidence_out_of_scope",
                        f"Sentence evidence is outside scope {scope_id}: {ref.id}",
                    )
            elif isinstance(ref, ChunkRef):
                if ref.id not in state.read_chunk_ids:
                    return (
                        "evidence_not_eligible",
                        f"Chunk must be READ before FINISH: {ref.id}",
                    )
                if ref.id not in self.substrate.chunk_ids_by_scope[scope_id]:
                    return (
                        "evidence_out_of_scope",
                        f"Chunk evidence is outside scope {scope_id}: {ref.id}",
                    )
        return None


def _referenced_policy_handles(
    text: str, state: ControllerState
) -> list[str]:
    """Return allocated handles used as natural-language tokens."""

    found: list[str] = []
    for handle in sorted(
        state.handle_registry.handle_to_stable_id,
        key=lambda value: (-len(value), value),
    ):
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(handle)}(?![A-Za-z0-9_])",
            text,
        ):
            found.append(handle)
    return found


_INTENT_GENERIC_WORDS = frozenset(
    {
        "action",
        "answer",
        "chunk",
        "current",
        "entity",
        "evidence",
        "expand",
        "fact",
        "find",
        "information",
        "more",
        "question",
        "read",
        "related",
        "search",
        "sentence",
        "visible",
    }
)


def _intent_has_visible_semantics(
    text: str,
    state: ControllerState,
    substrate: Substrate,
) -> bool:
    """Distinguish semantic grounding from an opaque S#/C#/E# shortcut."""

    intent_tokens = _semantic_tokens(text)
    for entity_id in state.visible_entity_ids:
        entity = substrate.entity_by_id.get(entity_id)
        if entity is None:
            continue
        label_tokens = _semantic_tokens(entity.canonical_name)
        if label_tokens and label_tokens.issubset(intent_tokens):
            return True
    for sentence_id in state.visible_sentence_ids:
        sentence = substrate.sentence_by_id.get(sentence_id)
        if sentence is None:
            continue
        overlap = intent_tokens & _semantic_tokens(sentence.text)
        if len(overlap) >= 2:
            return True
    for chunk_id in state.visible_chunk_ids:
        chunk = substrate.chunk_by_id.get(chunk_id)
        if chunk is None:
            continue
        document = substrate.document_by_id.get(chunk.doc_id)
        if document is None:
            continue
        title_tokens = _semantic_tokens(document.title or "")
        if len(intent_tokens & title_tokens) >= 2:
            return True
    return False


def _semantic_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.casefold())
        if token not in _INTENT_GENERIC_WORDS
    }
