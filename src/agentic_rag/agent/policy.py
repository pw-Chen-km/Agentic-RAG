"""Provider-neutral policy interface and OpenAI Responses implementation."""

from __future__ import annotations

import hashlib
import os
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from functools import lru_cache
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field, ValidationError, create_model

from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    ActionSelection,
    ActionType,
    AgentModel,
    AssessmentStatus,
    EvidenceAssessment,
    ExpandAction,
    ExpandParameters,
    ExpansionDirection,
    ExpansionKind,
    FinishAction,
    FinishParameters,
    Message,
    PolicyDecision,
    PolicyDecisionOutput,
    ReadAction,
    ReadParameters,
    SearchAction,
    SearchParameters,
    SearchMethod,
    SearchTarget,
    Usage,
    V31PolicyDecision,
    V3PolicyDecision,
)


class PolicyError(RuntimeError):
    """Base class for policy failures that terminate an episode cleanly."""


class PolicyConfigurationError(PolicyError):
    """The provider client cannot be configured."""


class PolicyResponseError(PolicyError):
    """The provider completed but did not return a valid decision."""


class PolicyTransportError(PolicyError):
    """The provider remained unavailable after bounded retries."""


StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


@runtime_checkable
class PolicyClient(Protocol):
    last_usage: Usage

    def generate_structured(
        self,
        messages: Sequence[Message | dict[str, str]],
        response_model: type[StructuredModelT],
    ) -> StructuredModelT:
        """Return one response validated as ``response_model``."""

    def decide(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        decision_format: type[BaseModel] | None = None,
    ) -> PolicyDecisionOutput:
        """Return exactly one assessment-and-action decision."""


class _WireSentenceRef(AgentModel):
    """OpenAI Structured Outputs representation of a Sentence reference.

    Provider-facing fields deliberately have no Python defaults. OpenAI's strict
    JSON Schema subset requires every object property to be required.
    """

    unit: Literal["SENTENCE"]
    id: str = Field(min_length=1)


class _WireChunkRef(AgentModel):
    unit: Literal["CHUNK"]
    id: str = Field(min_length=1)


_WireEvidenceRef = _WireSentenceRef | _WireChunkRef


class _WireEvidenceAssessment(AgentModel):
    status: AssessmentStatus
    supported_facts: list[str] = Field(max_length=5)
    missing_information: list[str] = Field(max_length=3)
    selected_evidence_refs: list[_WireEvidenceRef] = Field(max_length=20)


class _WireActionSelection(AgentModel):
    """OpenAI-strict form without discriminators/defaulted ref fields."""

    action_type: ActionType
    action_intent: str = Field(min_length=1)
    selected_evidence_refs: list[_WireEvidenceRef] = Field(max_length=20)


class _WireFinishParameters(AgentModel):
    """OpenAI-strict V2-2 FINISH parameter response."""

    answer: str = Field(min_length=1)
    evidence_refs: list[_WireEvidenceRef] = Field(
        min_length=1, max_length=20
    )


class _WireSearchAction(AgentModel):
    type: Literal["SEARCH"]
    query: str = Field(min_length=1)
    method: SearchMethod
    target: SearchTarget
    top_k: Literal[5]


class _WireExpandActionBase(AgentModel):
    """Shared required fields for provider-facing EXPAND variants."""

    type: Literal["EXPAND"]
    kind: str
    source_id: str = Field(min_length=1)
    query: str | None
    top_k: Literal[5]


class _WireNonAdjacentExpandAction(_WireExpandActionBase):
    """Non-adjacent expansions must explicitly emit JSON null."""

    direction: Literal[None]


class _WireAdjacentExpandAction(_WireExpandActionBase):
    """Chunk adjacency always requires a concrete traversal direction."""

    kind: Literal["CHUNK_ADJACENT_CHUNK"]
    direction: ExpansionDirection


class _WireReadAction(AgentModel):
    type: Literal["READ"]
    chunk_id: str = Field(min_length=1)


class _WireFinishAction(AgentModel):
    type: Literal["FINISH"]
    evidence_refs: list[_WireEvidenceRef] = Field(min_length=1, max_length=20)


class _WireDirectFinishAction(_WireFinishAction):
    """Single-agent terminal action containing the grounded final answer."""

    answer: str = Field(min_length=1)


class _WireV3EvidenceAssessment(AgentModel):
    status: AssessmentStatus
    supported_facts: list[str] = Field(max_length=5)
    missing_information: list[str] = Field(max_length=3)


class _WireV3ExpandActionBase(AgentModel):
    type: Literal["EXPAND"]
    kind: str
    source_context_index: int = Field(ge=1)
    query: str | None
    top_k: Literal[5]


class _WireV3NonAdjacentExpandAction(_WireV3ExpandActionBase):
    direction: Literal[None]


class _WireV3AdjacentExpandAction(_WireV3ExpandActionBase):
    kind: Literal["CHUNK_ADJACENT_CHUNK"]
    direction: ExpansionDirection


class _WireV3ReadAction(AgentModel):
    type: Literal["READ"]
    chunk_context_index: int = Field(ge=1)


class _WireV3FinishAction(AgentModel):
    type: Literal["FINISH"]
    answer: str = Field(min_length=1)
    citations: list[int] = Field(min_length=1, max_length=20)


class _WireV31ExpandActionBase(AgentModel):
    type: Literal["EXPAND"]
    kind: str
    source_ref: str = Field(min_length=2, description="Visible E#/S#/C# ref")
    query: str | None
    top_k: Literal[5]


class _WireV31NonAdjacentExpandAction(_WireV31ExpandActionBase):
    direction: Literal[None]


class _WireV31AdjacentExpandAction(_WireV31ExpandActionBase):
    kind: Literal["CHUNK_ADJACENT_CHUNK"]
    direction: ExpansionDirection


class _WireV31ReadAction(AgentModel):
    type: Literal["READ"]
    chunk_ref: str = Field(min_length=2, description="Visible unread C# ref")


class _WireV31FinishAction(AgentModel):
    type: Literal["FINISH"]
    answer: str = Field(min_length=1)
    evidence_refs: list[str] = Field(
        min_length=1,
        max_length=20,
        description="Visible S# or read C# refs",
    )


ScriptedDecision = (
    BaseModel
    | dict[str, Any]
    | Exception
    | Callable[[Sequence[Message | dict[str, str]]], BaseModel | dict[str, Any]]
)


class ScriptedPolicy:
    """Deterministic zero-cost policy used by unit and integration tests."""

    def __init__(self, decisions: Iterable[ScriptedDecision]) -> None:
        self._decisions = deque(decisions)
        self.calls: list[list[Message | dict[str, str]]] = []
        self.last_usage = Usage()

    def decide(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        decision_format: type[BaseModel] | None = None,
    ) -> PolicyDecisionOutput:
        if decision_format is not None:
            scripted: BaseModel | dict[str, Any] = self.generate_structured(
                messages,
                decision_format,
            )
        else:
            scripted = self._next_scripted(messages)
        if isinstance(
            scripted,
            (PolicyDecision, V3PolicyDecision, V31PolicyDecision),
        ):
            return scripted
        payload = (
            scripted.model_dump(mode="json")
            if isinstance(scripted, BaseModel)
            else scripted
        )
        preferred = (
            _decision_type_for_format(decision_format)
            if decision_format is not None
            else PolicyDecision
        )
        decision_types = tuple(
            dict.fromkeys(
                (
                    preferred,
                    PolicyDecision,
                    V3PolicyDecision,
                    V31PolicyDecision,
                )
            )
        )
        for decision_type in decision_types:
            try:
                return decision_type.model_validate(payload)
            except (ValidationError, TypeError, ValueError):
                continue
        raise PolicyResponseError(
            "scripted policy output failed PolicyDecision validation"
        )

    def generate_structured(
        self,
        messages: Sequence[Message | dict[str, str]],
        response_model: type[StructuredModelT],
    ) -> StructuredModelT:
        scripted = self._next_scripted(messages)
        payload = (
            scripted.model_dump(mode="json")
            if isinstance(scripted, BaseModel)
            else scripted
        )
        try:
            return response_model.model_validate(payload)
        except (ValidationError, TypeError, ValueError) as exc:
            raise PolicyResponseError(
                "scripted policy output failed structured response validation"
            ) from exc

    def _next_scripted(
        self,
        messages: Sequence[Message | dict[str, str]],
    ) -> BaseModel | dict[str, Any]:
        self.last_usage = Usage()
        self.calls.append(list(messages))
        self.last_usage = Usage(policy_calls=1)
        if not self._decisions:
            raise PolicyResponseError("scripted policy has no decisions remaining")
        scripted = self._decisions.popleft()
        if isinstance(scripted, Exception):
            raise scripted
        if callable(scripted):
            scripted = scripted(messages)
        if not isinstance(scripted, (BaseModel, dict)):
            raise PolicyResponseError(
                "scripted policy output must be a model or object"
            )
        return scripted


class OpenAIResponsesPolicy:
    """Structured-output PolicyClient backed by the OpenAI Responses API."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-terra",
        client: Any | None = None,
        enabled_expansions: Sequence[
            ExpansionKind | str
        ] = DEFAULT_ENABLED_EXPANSIONS,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        direct_answer: bool = False,
        semantic_memory_v3: bool = False,
        semantic_memory_v31: bool = False,
    ) -> None:
        if not model.strip():
            raise PolicyConfigurationError("model must not be blank")
        if max_retries < 0:
            raise PolicyConfigurationError("max_retries must be non-negative")
        if retry_backoff_seconds < 0:
            raise PolicyConfigurationError(
                "retry_backoff_seconds must be non-negative"
            )
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.enabled_expansions = _normalize_enabled_expansions(
            enabled_expansions
        )
        self.direct_answer = direct_answer
        self.semantic_memory_v3 = semantic_memory_v3
        self.semantic_memory_v31 = semantic_memory_v31
        self.decision_format = policy_decision_model(
            self.enabled_expansions,
            direct_answer=direct_answer,
            semantic_memory_v3=semantic_memory_v3,
            semantic_memory_v31=semantic_memory_v31,
        )
        self._client = client if client is not None else self._create_client()
        self.last_usage = Usage()

    @staticmethod
    def _create_client() -> Any:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise PolicyConfigurationError(
                "OPENAI_API_KEY is required for the OpenAI policy"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PolicyConfigurationError(
                "Install the 'openai' package to use OpenAIResponsesPolicy"
            ) from exc
        # Retry exactly at this adapter boundary so one configured retry count
        # cannot be multiplied by the SDK's own default retries.
        return OpenAI(api_key=api_key, max_retries=0)

    def decide(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        decision_format: type[BaseModel] | None = None,
    ) -> PolicyDecisionOutput:
        response_format = decision_format or self.decision_format
        parsed = self.generate_structured(messages, response_format)
        payload = parsed.model_dump(mode="json")
        try:
            decision_type = _decision_type_for_format(response_format)
            return decision_type.model_validate(payload)
        except Exception as exc:
            raise PolicyResponseError(
                "OpenAI response failed PolicyDecision validation"
            ) from exc

    def generate_structured(
        self,
        messages: Sequence[Message | dict[str, str]],
        response_model: type[StructuredModelT],
    ) -> StructuredModelT:
        """Generate and validate any Pydantic structured policy response."""

        # Never let usage from a previous provider call leak into a failed one.
        self.last_usage = Usage()
        provider_input = [
            (
                message.as_openai_input()
                if isinstance(message, Message)
                else dict(message)
            )
            for message in messages
        ]
        provider_response_model = _openai_wire_response_model(response_model)
        response = self._call_with_retries(
            lambda: self._client.responses.parse(
                model=self.model,
                input=provider_input,
                text_format=provider_response_model,
                reasoning={"context": "current_turn"},
                store=False,
            )
        )
        self.last_usage = _extract_usage(response, policy_calls=1)
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            refusal = _response_refusal(response)
            detail = f": {refusal}" if refusal else ""
            raise PolicyResponseError(
                "OpenAI response did not contain parsed structured output"
                f"{detail}"
            )
        try:
            payload = (
                parsed.model_dump(mode="json")
                if isinstance(parsed, BaseModel)
                else parsed
            )
            return response_model.model_validate(payload)
        except Exception as exc:
            raise PolicyResponseError(
                "OpenAI response failed structured response validation"
            ) from exc

    def _call_with_retries(self, operation: Callable[[], Any]) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                if isinstance(exc, (ValidationError, ValueError, TypeError)):
                    raise PolicyResponseError(
                        "OpenAI output failed structured response parsing"
                    ) from exc
                if attempt >= self.max_retries or not _is_transient(exc):
                    raise PolicyTransportError(
                        f"OpenAI policy request failed after {attempt + 1} attempt(s)"
                    ) from exc
                if self.retry_backoff_seconds:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise AssertionError("retry loop must return or raise")


def _openai_wire_response_model(
    response_model: type[StructuredModelT],
) -> type[BaseModel]:
    """Map V2-2 evidence unions to OpenAI's strict-schema subset."""

    if response_model is ActionSelection:
        return _WireActionSelection
    if response_model is FinishParameters:
        return _WireFinishParameters
    return response_model


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
    }:
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return isinstance(status, int) and (
        status in {408, 409, 429} or status >= 500
    )


def _decision_type_for_format(
    response_model: type[BaseModel],
) -> type[PolicyDecision] | type[V3PolicyDecision] | type[V31PolicyDecision]:
    """Identify the stable decision contract represented by a wire schema."""

    def contains_field(value: object, fields: frozenset[str]) -> bool:
        if isinstance(value, dict):
            if any(field in value for field in fields):
                return True
            return any(
                contains_field(child, fields) for child in value.values()
            )
        if isinstance(value, list):
            return any(contains_field(child, fields) for child in value)
        return False

    schema = response_model.model_json_schema()
    if contains_field(schema, frozenset({"source_ref", "chunk_ref"})):
        return V31PolicyDecision
    if contains_field(
        schema,
        frozenset(
            {"source_context_index", "chunk_context_index", "citations"}
        ),
    ):
        return V3PolicyDecision
    return PolicyDecision


def _extract_usage(
    response: Any, *, policy_calls: int = 0, answer_calls: int = 0
) -> Usage:
    raw = getattr(response, "usage", None)
    if raw is None:
        return Usage(policy_calls=policy_calls, answer_calls=answer_calls)

    def get(name: str, default: int = 0) -> int:
        if isinstance(raw, dict):
            value = raw.get(name, default)
        else:
            value = getattr(raw, name, default)
        return int(value or 0)

    reasoning_tokens = 0
    details = (
        raw.get("output_tokens_details")
        if isinstance(raw, dict)
        else getattr(raw, "output_tokens_details", None)
    )
    if details is not None:
        if isinstance(details, dict):
            reasoning_tokens = int(details.get("reasoning_tokens", 0) or 0)
        else:
            reasoning_tokens = int(
                getattr(details, "reasoning_tokens", 0) or 0
            )
    input_tokens = get("input_tokens")
    output_tokens = get("output_tokens")
    stage_usage: dict[str, int] = {}
    if policy_calls:
        stage_usage.update(
            policy_input_tokens=input_tokens,
            policy_output_tokens=output_tokens,
            policy_reasoning_tokens=reasoning_tokens,
        )
    if answer_calls:
        stage_usage.update(
            answer_input_tokens=input_tokens,
            answer_output_tokens=output_tokens,
            answer_reasoning_tokens=reasoning_tokens,
        )
    return Usage(
        policy_calls=policy_calls,
        answer_calls=answer_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=get("total_tokens"),
        **stage_usage,
    )


def _response_refusal(response: Any) -> str | None:
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            refusal = getattr(content, "refusal", None)
            if refusal:
                return str(refusal)
    return None


def policy_decision_model(
    enabled_expansions: Sequence[
        ExpansionKind | str
    ] = DEFAULT_ENABLED_EXPANSIONS,
    *,
    direct_answer: bool = False,
    semantic_memory_v3: bool = False,
    semantic_memory_v31: bool = False,
) -> type[BaseModel]:
    """Build the provider schema for exactly the enabled EXPAND kinds.

    The provider-specific instance is normalized back into the stable
    :class:`PolicyDecision` immediately after parsing.
    """

    normalized = _normalize_enabled_expansions(enabled_expansions)
    return _policy_decision_model(
        normalized,
        direct_answer,
        semantic_memory_v3,
        semantic_memory_v31,
    )


def action_parameters_model(
    enabled_expansions: Sequence[ExpansionKind | str] | ActionType | str = (
        DEFAULT_ENABLED_EXPANSIONS
    ),
    action_type: ActionType | str | None = None,
) -> type[BaseModel]:
    """Return the V2-2 stage-two schema for one selected action family.

    The two-argument form accepts ``(enabled_expansions, action_type)``.  For
    convenience, ``action_parameters_model(action_type)`` uses the default
    four expansion relations.
    """

    if action_type is None:
        if not isinstance(enabled_expansions, (ActionType, str)):
            raise TypeError("action_type is required")
        resolved_action_type = ActionType(enabled_expansions)
        normalized = DEFAULT_ENABLED_EXPANSIONS
    else:
        resolved_action_type = ActionType(action_type)
        if isinstance(enabled_expansions, (ActionType, str)):
            raise TypeError("enabled_expansions must be a sequence")
        normalized = _normalize_enabled_expansions(enabled_expansions)
    return _action_parameters_model(normalized, resolved_action_type)


@lru_cache(maxsize=None)
def _action_parameters_model(
    enabled_expansions: tuple[ExpansionKind, ...],
    action_type: ActionType,
) -> type[BaseModel]:
    if action_type is ActionType.SEARCH:
        return SearchParameters
    if action_type is ActionType.READ:
        return ReadParameters
    if action_type is ActionType.FINISH:
        return FinishParameters
    if not enabled_expansions:
        raise ValueError("EXPAND has no enabled expansion kinds")

    signature = ",".join(item.value for item in enabled_expansions)
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    enabled_enum = StrEnum(
        f"EnabledActionParameterExpansionKind_{digest}",
        {item.name: item.value for item in enabled_expansions},
        module=__name__,
    )
    return create_model(
        f"EnabledExpandParameters_{len(enabled_expansions)}_{digest}",
        __base__=ExpandParameters,
        kind=(enabled_enum, ...),
    )


def _normalize_enabled_expansions(
    enabled_expansions: Sequence[ExpansionKind | str],
) -> tuple[ExpansionKind, ...]:
    normalized = tuple(ExpansionKind(item) for item in enabled_expansions)
    if len(normalized) != len(set(normalized)):
        raise ValueError("enabled_expansions must not contain duplicates")
    return normalized


@lru_cache(maxsize=None)
def _policy_decision_model(
    enabled_expansions: tuple[ExpansionKind, ...],
    direct_answer: bool,
    semantic_memory_v3: bool,
    semantic_memory_v31: bool,
) -> type[BaseModel]:
    signature = ",".join(item.value for item in enabled_expansions)
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    suffix = (
        f"{len(enabled_expansions)}_{digest}"
        if enabled_expansions
        else "NO_EXPAND"
    )
    if direct_answer:
        suffix = f"{suffix}_DIRECT"
    if semantic_memory_v3 and semantic_memory_v31:
        raise ValueError("V3 index and V3.1 typed-ref schemas are exclusive")
    if semantic_memory_v31:
        suffix = f"{suffix}_V31"
        action_types: Any = _WireSearchAction
        non_adjacent = tuple(
            item
            for item in enabled_expansions
            if item is not ExpansionKind.CHUNK_ADJACENT_CHUNK
        )
        if non_adjacent:
            enabled_enum = StrEnum(
                f"EnabledV31NonAdjacentExpansionKind_{suffix}",
                {item.name: item.value for item in non_adjacent},
                module=__name__,
            )
            provider_expand = create_model(
                f"EnabledV31NonAdjacentExpandAction_{suffix}",
                __base__=_WireV31NonAdjacentExpandAction,
                kind=(enabled_enum, ...),
            )
            action_types = action_types | provider_expand
        if ExpansionKind.CHUNK_ADJACENT_CHUNK in enabled_expansions:
            action_types = action_types | _WireV31AdjacentExpandAction
        action_types = (
            action_types | _WireV31ReadAction | _WireV31FinishAction
        )
        return create_model(
            f"EnabledV31PolicyDecision_{suffix}",
            __base__=AgentModel,
            assessment=(_WireV3EvidenceAssessment, ...),
            action=(action_types, ...),
        )
    if semantic_memory_v3:
        suffix = f"{suffix}_V3"
        action_types: Any = _WireSearchAction
        non_adjacent = tuple(
            item
            for item in enabled_expansions
            if item is not ExpansionKind.CHUNK_ADJACENT_CHUNK
        )
        if non_adjacent:
            enabled_enum = StrEnum(
                f"EnabledV3NonAdjacentExpansionKind_{suffix}",
                {item.name: item.value for item in non_adjacent},
                module=__name__,
            )
            provider_expand = create_model(
                f"EnabledV3NonAdjacentExpandAction_{suffix}",
                __base__=_WireV3NonAdjacentExpandAction,
                kind=(enabled_enum, ...),
            )
            action_types = action_types | provider_expand
        if ExpansionKind.CHUNK_ADJACENT_CHUNK in enabled_expansions:
            action_types = action_types | _WireV3AdjacentExpandAction
        action_types = action_types | _WireV3ReadAction | _WireV3FinishAction
        return create_model(
            f"EnabledV3PolicyDecision_{suffix}",
            __base__=AgentModel,
            assessment=(_WireV3EvidenceAssessment, ...),
            action=(action_types, ...),
        )
    action_types: Any = _WireSearchAction
    non_adjacent = tuple(
        item
        for item in enabled_expansions
        if item is not ExpansionKind.CHUNK_ADJACENT_CHUNK
    )
    if non_adjacent:
        enabled_enum = StrEnum(
            f"EnabledNonAdjacentExpansionKind_{suffix}",
            {item.name: item.value for item in non_adjacent},
            module=__name__,
        )
        provider_expand = create_model(
            f"EnabledNonAdjacentExpandAction_{suffix}",
            __base__=_WireNonAdjacentExpandAction,
            kind=(enabled_enum, ...),
        )
        action_types = action_types | provider_expand
    if ExpansionKind.CHUNK_ADJACENT_CHUNK in enabled_expansions:
        action_types = action_types | _WireAdjacentExpandAction
    finish_type = (
        _WireDirectFinishAction if direct_answer else _WireFinishAction
    )
    action_types = action_types | _WireReadAction | finish_type

    return create_model(
        f"EnabledPolicyDecision_{suffix}",
        __base__=AgentModel,
        assessment=(_WireEvidenceAssessment, ...),
        action=(action_types, ...),
    )
