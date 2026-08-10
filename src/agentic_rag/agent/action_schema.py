"""Build bounded-cache provider schemas from one AvailableActionSpace."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from agentic_rag.agent.models import (
    AgentModel,
    AvailableActionSpace,
    ExpansionKind,
)
from agentic_rag.agent.policy import (
    _WireAdjacentExpandAction,
    _WireAssessment,
    _WireFinishAction,
    _WireNonAdjacentExpandAction,
    _WireReadAction,
    _WireSearchAction,
)


class ActionSchemaBuilder:
    """Translate structural affordances into an OpenAI/Ollama response model."""

    def build(self, action_space: AvailableActionSpace) -> type[BaseModel]:
        if not action_space.has_actions:
            raise ValueError("cannot build a Policy schema without any legal action")
        return _state_conditioned_model(action_space.model_dump_json())


def decision_schema_sha256(model: type[BaseModel]) -> str:
    canonical = json.dumps(
        model.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@lru_cache(maxsize=256)
def _state_conditioned_model(serialized_action_space: str) -> type[BaseModel]:
    action_space = AvailableActionSpace.model_validate_json(serialized_action_space)
    digest = hashlib.sha256(serialized_action_space.encode("utf-8")).hexdigest()[:12]
    action_types: list[type[BaseModel]] = []

    for index, option in enumerate(action_space.search_options, start=1):
        method_type = _literal(option.method.value)
        target_type = _literal(option.target.value)
        action_types.append(
            create_model(
                f"StateSearchAction_{digest}_{index}",
                __base__=_WireSearchAction,
                method=(method_type, ...),
                target=(target_type, ...),
            )
        )

    for index, option in enumerate(action_space.expand_options, start=1):
        kind_type = _literal(option.kind.value)
        source_type = _literal(*(str(ref) for ref in option.source_refs))
        if option.kind is ExpansionKind.CHUNK_ADJACENT_CHUNK:
            direction_type = _literal(*(item.value for item in option.directions))
            action_types.append(
                create_model(
                    f"StateAdjacentExpandAction_{digest}_{index}",
                    __base__=_WireAdjacentExpandAction,
                    kind=(kind_type, ...),
                    source_ref=(source_type, ...),
                    direction=(direction_type, ...),
                )
            )
        else:
            action_types.append(
                create_model(
                    f"StateExpandAction_{digest}_{index}",
                    __base__=_WireNonAdjacentExpandAction,
                    kind=(kind_type, ...),
                    source_ref=(source_type, ...),
                )
            )

    if action_space.read_refs:
        ref_type = _literal(*(str(ref) for ref in action_space.read_refs))
        action_types.append(
            create_model(
                f"StateReadAction_{digest}",
                __base__=_WireReadAction,
                chunk_ref=(ref_type, ...),
            )
        )

    if action_space.finish_evidence_refs:
        evidence_type = _literal(
            *(str(ref) for ref in action_space.finish_evidence_refs)
        )
        evidence_list_type = list[evidence_type]
        action_types.append(
            create_model(
                f"StateFinishAction_{digest}",
                __base__=_WireFinishAction,
                evidence_refs=(
                    evidence_list_type,
                    Field(
                        min_length=1,
                        max_length=min(20, len(action_space.finish_evidence_refs)),
                        description="Visible complete S# or read C# refs",
                    ),
                ),
            )
        )

    if not action_types:
        raise ValueError("cannot build a Policy schema without any legal action")
    action_union: Any = action_types[0]
    for action_type in action_types[1:]:
        action_union = action_union | action_type
    return create_model(
        f"StateConditionedPolicyDecision_{digest}",
        __base__=AgentModel,
        assessment=(_WireAssessment, ...),
        action=(action_union, ...),
    )


def _literal(*values: str) -> Any:
    if not values:
        raise ValueError("Literal requires at least one value")
    return Literal.__getitem__(values)
