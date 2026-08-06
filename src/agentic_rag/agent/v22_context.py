"""Progressively disclosed policy contexts for the V2-2 workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.models import (
    ActionSelection,
    ChunkHandle,
    ControllerState,
    EntityHandle,
    Message,
    PolicyView,
    SentenceHandle,
    StepRecord,
)
from agentic_rag.agent.skill import ProgressiveSkillBundle, SkillDocument


V22_SELECTION_PROTOCOL = """\
You are the high-level strategy stage of a two-stage Agentic RAG policy.
Return exactly one ActionSelection. Choose SEARCH, EXPAND, READ, or FINISH,
state the semantic reason in action_intent, and select any complete evidence
that should be retained transactionally if the resulting action succeeds.

Do not emit action parameters. In action_intent, use real entity names and
textual meaning, never E#/S#/C# as a substitute for meaning. Evidence refs are
the only handle-valued fields in this stage. Select only complete Sentence
evidence or already-read Chunk evidence shown in the frozen state.
"""


V22_ACTION_PROTOCOL = """\
You are the execution stage of a two-stage Agentic RAG policy. The high-level
action family is already fixed. Return only the parameters required by the
provided action-specific response schema; do not repeat the action type.

Follow the supplied action_intent. Copy E#/S#/C# handles only into structural
ID or evidence-reference fields. Natural-language query fields must use real
names and semantic text. visible_handles is observational memory, not a claim
that every displayed handle is legal for this action. The Validator decides
state-dependent legality after the draft.
"""


V22_REPAIR_PROTOCOL = """\
The preceding action draft was rejected. Return one corrected parameter object
for the same action family. Use only legal_action_options and the same frozen
visible_handles. Do not change the high-level action or its intent.
"""


_CAPABILITY_KEYS = frozenset(
    {
        "actionable_handles",
        "allowed_expansions",
        "available_handles",
        "can_expand",
        "can_read",
        "can_use_as_evidence",
        "legal_action_options",
    }
)


@dataclass(frozen=True, slots=True)
class V22BuiltContext:
    """Exact messages and disclosure metadata for one V2-2 provider call."""

    messages: list[Message]
    policy_view: PolicyView
    disclosed_documents: tuple[tuple[str, str], ...]


class V22ContextBuilder:
    """Build root, action-draft, and action-repair contexts."""

    def __init__(self, base: PolicyContextBuilder) -> None:
        self.base = base

    def build_selection(
        self,
        query: str,
        bundle: ProgressiveSkillBundle,
        state: ControllerState,
        trajectory: Sequence[StepRecord],
        *,
        scope_id: str,
        extra_instruction: str | None = None,
    ) -> V22BuiltContext:
        policy_view, state_payload = self._state_payload(
            state, trajectory, scope_id=scope_id
        )
        messages = [
            Message(role="system", content=V22_SELECTION_PROTOCOL),
            Message(
                role="system",
                content=self._render_documents(
                    bundle,
                    (bundle.root,),
                    heading="Root strategy skill",
                ),
            ),
            Message(role="user", content=f"Original question:\n{query}"),
            Message(
                role="user",
                content=self._json({"current_state": state_payload}),
            ),
        ]
        if extra_instruction:
            messages.append(Message(role="user", content=extra_instruction))
        return V22BuiltContext(
            messages=messages,
            policy_view=policy_view,
            disclosed_documents=self._document_metadata(
                bundle, (bundle.root,)
            ),
        )

    def build_action(
        self,
        query: str,
        bundle: ProgressiveSkillBundle,
        selection: ActionSelection,
        state: ControllerState,
        trajectory: Sequence[StepRecord],
        *,
        scope_id: str,
        recovery: bool = False,
        original_parameters: Mapping[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        legal_action_options: Sequence[Mapping[str, Any]] = (),
    ) -> V22BuiltContext:
        policy_view, state_payload = self._state_payload(
            state, trajectory, scope_id=scope_id
        )
        documents = bundle.stage_documents(
            selection.action_type, recovery=recovery
        )
        messages = [
            Message(
                role="system",
                content=(
                    V22_ACTION_PROTOCOL
                    + (f"\n\n{V22_REPAIR_PROTOCOL}" if recovery else "")
                ),
            ),
            Message(
                role="system",
                content=self._render_documents(
                    bundle,
                    documents,
                    heading=f"{selection.action_type.value} action skill",
                ),
            ),
            Message(role="user", content=f"Original question:\n{query}"),
        ]
        payload: dict[str, Any] = {
            "action_selection": selection.model_dump(mode="json"),
            "current_state": state_payload,
        }
        if recovery:
            payload["rejected_parameters"] = original_parameters
            payload["validation_error"] = {
                "code": error_code,
                "message": error_message,
            }
            payload["legal_action_options"] = list(legal_action_options)
        messages.append(Message(role="user", content=self._json(payload)))
        return V22BuiltContext(
            messages=messages,
            policy_view=policy_view,
            disclosed_documents=self._document_metadata(bundle, documents),
        )

    def _state_payload(
        self,
        state: ControllerState,
        trajectory: Sequence[StepRecord],
        *,
        scope_id: str,
    ) -> tuple[PolicyView, dict[str, Any]]:
        view = self.base.build_policy_view(
            state, trajectory, scope_id=scope_id
        )
        projected = view.policy_state
        payload = {
            "step": state.step,
            "policy_attempts": state.policy_attempts,
            "accumulated_evidence": [
                item.model_dump(mode="json")
                for item in projected.selected_evidence
            ],
            "visible_handles": self._visible_handles(state),
            "latest_observation": self._strip_capabilities(
                projected.latest_observation
            ),
            "attempted_actions": [
                item.model_dump(mode="json")
                for item in projected.attempted_actions
            ],
            "budget": projected.budget.model_dump(mode="json"),
        }
        return view, payload

    def _visible_handles(
        self, state: ControllerState
    ) -> list[dict[str, Any]]:
        substrate = (
            self.base.evidence_resolver.substrate
            if self.base.evidence_resolver is not None
            else None
        )
        visible: list[dict[str, Any]] = []
        for stable_id, handle in sorted(
            state.node_handles.items(),
            key=lambda item: (item[1].node_type, item[1].id),
        ):
            if isinstance(handle, EntityHandle):
                visible.append(
                    {
                        "node_type": "ENTITY",
                        "handle": handle.id,
                        "label": handle.label,
                        "entity_type": handle.entity_type,
                    }
                )
                continue
            if isinstance(handle, SentenceHandle):
                sentence_text = handle.text
                if handle.can_use_as_evidence and substrate is not None:
                    sentence = substrate.sentence_by_id.get(stable_id)
                    if sentence is not None:
                        # Complete evidence is disclosed verbatim. The legacy
                        # SentenceHandle summary remains appropriate only for
                        # navigation previews and older compact contexts.
                        sentence_text = sentence.text
                visible.append(
                    {
                        "node_type": "SENTENCE",
                        "handle": handle.id,
                        "text": sentence_text,
                        "title": handle.title,
                        "parent_chunk": handle.parent_chunk_id,
                        "completeness": (
                            "COMPLETE"
                            if handle.can_use_as_evidence
                            else "PREVIEW"
                        ),
                    }
                )
                continue
            if isinstance(handle, ChunkHandle):
                text = None
                if handle.has_been_read and substrate is not None:
                    chunk = substrate.chunk_by_id.get(stable_id)
                    text = chunk.text if chunk is not None else None
                visible.append(
                    {
                        "node_type": "CHUNK",
                        "handle": handle.id,
                        "title": handle.title,
                        "read_state": (
                            "READ" if handle.has_been_read else "UNREAD"
                        ),
                        "text": text,
                        "previews": [
                            {
                                "sentence_handle": preview.sentence_id,
                                "text": preview.text,
                            }
                            for preview in handle.previews
                        ],
                    }
                )
                continue
            raise TypeError(
                f"unsupported V2-2 node handle: {type(handle).__name__}"
            )
        return visible

    @classmethod
    def _strip_capabilities(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): cls._strip_capabilities(item)
                for key, item in value.items()
                if str(key) not in _CAPABILITY_KEYS
            }
        if isinstance(value, list):
            return [cls._strip_capabilities(item) for item in value]
        return value

    @staticmethod
    def _relative_path(
        bundle: ProgressiveSkillBundle,
        document: SkillDocument,
    ) -> str:
        if document.source_path is None:
            raise ValueError("bundle documents require source paths")
        source = Path(document.source_path).resolve()
        root = Path(bundle.source_root).resolve()
        return source.relative_to(root).as_posix()

    @classmethod
    def _document_metadata(
        cls,
        bundle: ProgressiveSkillBundle,
        documents: Sequence[SkillDocument],
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (cls._relative_path(bundle, document), document.sha256)
            for document in documents
        )

    @classmethod
    def _render_documents(
        cls,
        bundle: ProgressiveSkillBundle,
        documents: Sequence[SkillDocument],
        *,
        heading: str,
    ) -> str:
        sections = [heading]
        for document in documents:
            sections.append(
                f"\n--- {cls._relative_path(bundle, document)} ---\n"
                f"{document.content}"
            )
        return "\n".join(sections)

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
