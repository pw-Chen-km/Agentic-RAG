"""Provider schemas for option selection and option policy turns."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from agentic_rag.agent.models import AgentModel, AvailableActionSpace, ExpansionKind
from agentic_rag.agent.policy import (
    _WireAdjacentExpandAction,
    _WireAssessment,
    _WireFinishAction,
    _WireNonAdjacentExpandAction,
    _WireReadAction,
    _WireSearchAction,
)


def selector_schema(option_ids: tuple[str, ...]) -> type[BaseModel]:
    if not option_ids:
        raise ValueError("option selector requires at least one option")
    return _selector_schema(tuple(option_ids))


@lru_cache(maxsize=128)
def _selector_schema(option_ids: tuple[str, ...]) -> type[BaseModel]:
    option_type = Literal.__getitem__(option_ids)
    return create_model(
        "OptionSelectorDecision_" + _digest(option_ids),
        __base__=AgentModel,
        option_id=(option_type, ...),
    )


def policy_schema(
    action_space: AvailableActionSpace,
    *,
    option_id: str,
    allowed_actions: tuple[str, ...],
) -> type[BaseModel]:
    """Build a schema from one frozen action space and one option's affordances."""

    action_types: list[type[BaseModel]] = []
    allowed = {item.upper() for item in allowed_actions}
    if "SEARCH" in allowed:
        for index, option in enumerate(action_space.search_options, start=1):
            action_types.append(
                create_model(
                    f"OptionSearchAction_{_digest((option_id, str(index), option.method.value, option.target.value))}",
                    __base__=_WireSearchAction,
                    method=(Literal.__getitem__((option.method.value,)), ...),
                    target=(Literal.__getitem__((option.target.value,)), ...),
                )
            )
    if "EXPAND" in allowed:
        for index, option in enumerate(action_space.expand_options, start=1):
            kind_type = Literal.__getitem__((option.kind.value,))
            ref_type = Literal.__getitem__(tuple(str(ref) for ref in option.source_refs))
            if option.kind is ExpansionKind.CHUNK_ADJACENT_CHUNK:
                direction_type = Literal.__getitem__(tuple(item.value for item in option.directions))
                action_types.append(
                    create_model(
                        f"OptionAdjacentExpandAction_{_digest((option_id, str(index), option.kind.value))}",
                        __base__=_WireAdjacentExpandAction,
                        kind=(kind_type, ...),
                        source_ref=(ref_type, ...),
                        direction=(direction_type, ...),
                    )
                )
            else:
                action_types.append(
                    create_model(
                        f"OptionExpandAction_{_digest((option_id, str(index), option.kind.value))}",
                        __base__=_WireNonAdjacentExpandAction,
                        kind=(kind_type, ...),
                        source_ref=(ref_type, ...),
                    )
                )
    if "READ" in allowed and action_space.read_refs:
        ref_type = Literal.__getitem__(tuple(str(ref) for ref in action_space.read_refs))
        action_types.append(
            create_model(
                f"OptionReadAction_{_digest((option_id, 'read'))}",
                __base__=_WireReadAction,
                chunk_ref=(ref_type, ...),
            )
        )
    if "FINISH" in allowed and action_space.finish_evidence_refs:
        evidence_type = Literal.__getitem__(tuple(str(ref) for ref in action_space.finish_evidence_refs))
        action_types.append(
            create_model(
                f"OptionFinishAction_{_digest((option_id, 'finish'))}",
                __base__=_WireFinishAction,
                evidence_refs=(
                    list[evidence_type],
                    Field(
                        min_length=1,
                        max_length=min(20, len(action_space.finish_evidence_refs)),
                    ),
                ),
            )
        )
    if not action_types:
        raise ValueError(f"option {option_id} has no legal primitive action")
    action_union: Any = action_types[0]
    for item in action_types[1:]:
        action_union = action_union | item
    status_type = Literal["CONTINUE", "COMPLETE", "BLOCKED"]
    return create_model(
        "OptionPolicyDecision_" + _digest((option_id, *allowed_actions, str(action_space.model_dump_json()))),
        __base__=AgentModel,
        assessment=(_WireAssessment, ...),
        option_status=(status_type, ...),
        # COMPLETE/BLOCKED are lifecycle-only responses and therefore omit
        # the primitive action.  A default is required here; an optional
        # annotation alone would still make the field mandatory in Pydantic.
        action=(action_union | None, None),
    )


def schema_sha256(model: type[BaseModel]) -> str:
    canonical = json.dumps(model.model_json_schema(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _digest(values: tuple[str, ...]) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()[:12]
