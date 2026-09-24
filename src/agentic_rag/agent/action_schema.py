"""Build bounded-cache provider schemas from one AvailableActionSpace."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from agentic_rag.agent.models import (
    AgentModel,
    Assessment,
    AvailableActionSpace,
    ExpandAction,
    ExpansionKind,
    FinishAction,
    PolicyDecision,
    ReadAction,
    SearchAction,
    SearchMethod,
    SearchTarget,
)
from agentic_rag.agent.interface_action_catalog import ACTION_CARDS, available_action_names
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


class InterfaceDecisionSchemaBuilder:
    """Build one constrained, plain-language action decision for study conditions.

    Unlike provider-native tool calling, this schema can represent exactly one
    action.  The provider constrains generation to the schema and the harness
    maps the selected action to the existing canonical action models.
    """

    def build(
        self,
        action_space: AvailableActionSpace,
        *,
        require_evidence_assessment: bool = True,
    ) -> type[BaseModel]:
        if not action_space.has_actions:
            raise ValueError("cannot build a Policy schema without any legal action")
        cache_key = json.dumps(
            {
                "action_space": action_space.model_dump(mode="json"),
                "require_evidence_assessment": require_evidence_assessment,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return _interface_decision_model(cache_key)


def policy_decision_from_constrained(value: BaseModel | dict[str, Any]) -> PolicyDecision:
    """Map a validated single-decision response to the canonical PolicyDecision."""

    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    raw_action = dict(payload["action"])
    name = raw_action.pop("name")
    assessment = None
    if "supported_facts" in payload or "missing_information" in payload:
        assessment = Assessment(
            supported_facts=payload.get("supported_facts", []),
            missing_information=payload.get("missing_information", []),
        )
    if name == "find_passages":
        action = SearchAction(
            query=raw_action["query"], method=SearchMethod.DENSE,
            target=SearchTarget.CHUNK,
        )
    elif name == "find_sentences":
        action = SearchAction(
            query=raw_action["query"], method=SearchMethod.DENSE,
            target=SearchTarget.SENTENCE,
        )
    elif name == "follow_entity_to_passages":
        action = ExpandAction(
            kind=ExpansionKind.ENTITY_MENTIONED_IN_CHUNK,
            source_ref=raw_action["entity_ref"], query=None,
        )
    elif name == "follow_entity_to_sentences":
        action = ExpandAction(
            kind=ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
            source_ref=raw_action["entity_ref"], query=None,
        )
    elif name == "read_passage":
        action = ReadAction(chunk_ref=raw_action["passage_ref"])
    elif name == "finish":
        action = FinishAction(
            answer=raw_action["answer"], evidence_refs=raw_action["evidence_refs"],
        )
    else:
        raise ValueError(f"unknown constrained action: {name}")
    return PolicyDecision(assessment=assessment, action=action)


def decision_schema_sha256(model: type[BaseModel]) -> str:
    canonical = json.dumps(
        model.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@lru_cache(maxsize=256)
def _interface_decision_model(cache_key: str) -> type[BaseModel]:
    config = json.loads(cache_key)
    action_space = AvailableActionSpace.model_validate(config["action_space"])
    require_assessment = bool(config["require_evidence_assessment"])
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:12]
    action_types: list[type[BaseModel]] = []

    for name in available_action_names(action_space):
        if name not in {"find_passages", "find_sentences"}:
            continue
        action_types.append(create_model(
            f"InterfaceSearch_{digest}_{name}", __base__=AgentModel,
            name=(Literal[name], Field(description=ACTION_CARDS[name].schema_description)),
            query=(str, Field(min_length=1, description="The text used to rank search results.")),
        ))

    for option in action_space.expand_options:
        if option.kind is ExpansionKind.ENTITY_MENTIONED_IN_CHUNK:
            name = "follow_entity_to_passages"
        elif option.kind is ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE:
            name = "follow_entity_to_sentences"
        else:
            raise ValueError("the study decision schema does not expose this navigation path")
        # Entity references are validated against the current observation by
        # the state validator.  Keeping the full E# list in the provider
        # schema makes the schema grow with every visible mention and does
        # not add semantic information beyond the entity card already shown
        # in the observation.
        ref_type = str
        fields: dict[str, Any] = {
            "name": (Literal[name], Field(description=ACTION_CARDS[name].schema_description)),
            "entity_ref": (ref_type, Field(description="A currently visible entity label and its displayed name.")),
        }
        action_types.append(create_model(
            f"InterfaceExpand_{digest}_{name}", __base__=AgentModel, **fields,
        ))

    if action_space.finish_available:
        if action_space.finish_evidence_refs:
            evidence_type = _literal(*(str(ref) for ref in action_space.finish_evidence_refs))
            evidence_list: Any = list[evidence_type]
            maximum = min(20, len(action_space.finish_evidence_refs))
        else:
            evidence_list, maximum = list[str], 0
        action_types.append(create_model(
            f"InterfaceFinish_{digest}", __base__=AgentModel,
            name=(Literal["finish"], Field(description=ACTION_CARDS["finish"].schema_description)),
            answer=(str, Field(min_length=1)),
            evidence_refs=(evidence_list, Field(min_length=0, max_length=maximum,
                                                description="Visible source labels supporting the answer.")),
        ))

    if not action_types:
        raise ValueError("cannot build a Policy schema without any legal action")
    action_union: Any = action_types[0]
    for action_type in action_types[1:]:
        action_union = action_union | action_type
    fields: dict[str, Any] = {
        "action": (action_union, Field(description="Choose exactly one currently available action.")),
    }
    if require_assessment:
        fields = {
            "supported_facts": (
                list[str], Field(max_length=5, description="Brief facts supported by source text already shown. Use an empty list when there are none."),
            ),
            "missing_information": (
                list[str], Field(max_length=3, description="Information still needed to answer. Use an empty list when nothing is missing."),
            ),
            **fields,
        }
    return create_model(
        f"InterfacePolicyDecision_{digest}", __base__=AgentModel, **fields,
    )


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

    if action_space.finish_available:
        if action_space.finish_evidence_refs:
            evidence_type = _literal(
                *(str(ref) for ref in action_space.finish_evidence_refs)
            )
            evidence_list_type = list[evidence_type]
            evidence_field = Field(
                min_length=0,
                max_length=min(20, len(action_space.finish_evidence_refs)),
                description="Visible complete S# or shown C# refs",
            )
        else:
            evidence_list_type = list[str]
            evidence_field = Field(
                min_length=0,
                max_length=0,
                description="No eligible evidence is currently visible",
            )
        action_types.append(
            create_model(
                f"StateFinishAction_{digest}",
                __base__=_WireFinishAction,
                evidence_refs=(
                    evidence_list_type,
                    evidence_field,
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
