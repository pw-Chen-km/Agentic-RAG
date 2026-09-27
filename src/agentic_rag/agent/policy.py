"""Provider-neutral Policy interface and canonical structured-output schema."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from functools import lru_cache
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, Field, ValidationError, create_model

from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    AgentModel,
    ExpansionDirection,
    ExpansionKind,
    Message,
    PolicyDecision,
    SearchMethod,
    SearchTarget,
    Usage,
)


class PolicyError(RuntimeError):
    """Base class for provider failures that terminate an episode cleanly."""


class PolicyConfigurationError(PolicyError):
    pass


class PolicyResponseError(PolicyError):
    pass


class PolicyTransportError(PolicyError):
    pass


StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)


@runtime_checkable
class PolicyClient(Protocol):
    last_usage: Usage

    def decide(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        decision_format: type[BaseModel] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_models: Mapping[str, type[BaseModel]] | None = None,
        output_mode: str | None = None,
    ) -> PolicyDecision:
        """Return exactly one assessment-and-action decision.

        ``tools`` and ``tool_models`` are optional so providers that support
        native tool calling can opt into it without breaking the structured
        output contract used by existing policies.
        """


class _WireAssessment(AgentModel):
    supported_facts: list[str] = Field(max_length=5)
    missing_information: list[str] = Field(max_length=3)


class _WireSearchAction(AgentModel):
    type: Literal["SEARCH"]
    query: str = Field(min_length=1)
    method: SearchMethod
    target: SearchTarget
    top_k: Literal[5]


class _WireExpandActionBase(AgentModel):
    type: Literal["EXPAND"]
    kind: str
    source_ref: str = Field(min_length=2, description="Visible E#/S#/C# ref")
    query: str | None
    top_k: Literal[5]


class _WireNonAdjacentExpandAction(_WireExpandActionBase):
    direction: Literal[None]


class _WireAdjacentExpandAction(_WireExpandActionBase):
    kind: Literal["CHUNK_ADJACENT_CHUNK"]
    direction: ExpansionDirection


class _WireReadAction(AgentModel):
    type: Literal["READ"]
    chunk_ref: str = Field(min_length=2, description="Visible unread C# ref")


class _WireFinishAction(AgentModel):
    type: Literal["FINISH"]
    answer: str = Field(min_length=1)
    evidence_refs: list[str] = Field(
        min_length=1,
        max_length=20,
        description="Visible complete S# or read C# refs",
    )


ScriptedDecision = (
    BaseModel
    | dict[str, Any]
    | Exception
    | Callable[[Sequence[Message | dict[str, str]]], BaseModel | dict[str, Any]]
)


class ScriptedPolicy:
    """Deterministic zero-cost Policy used by tests and reproducible examples."""

    def __init__(self, decisions: Iterable[ScriptedDecision]) -> None:
        self._decisions = deque(decisions)
        self.calls: list[list[Message | dict[str, str]]] = []
        self.last_usage = Usage()

    def decide(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        decision_format: type[BaseModel] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_models: Mapping[str, type[BaseModel]] | None = None,
        output_mode: str | None = None,
    ) -> PolicyDecision:
        if output_mode == "native_tools":
            if not tools or not tool_models:
                raise PolicyConfigurationError(
                    "native_tools output mode requires tools and tool_models"
                )
            name, parsed = self.decide_tool(
                messages,
                tools=tools,
                tool_models=tool_models,
            )
            payload = parsed.model_dump(mode="json")
            try:
                return PolicyDecision.model_validate(payload)
            except (ValidationError, TypeError, ValueError) as exc:
                raise PolicyResponseError(
                    f"scripted native tool {name!r} failed PolicyDecision validation"
                ) from exc
        self.last_usage = Usage(policy_calls=1)
        self.calls.append(list(messages))
        if not self._decisions:
            raise PolicyResponseError("scripted policy has no decisions remaining")
        scripted = self._decisions.popleft()
        if isinstance(scripted, Exception):
            raise scripted
        if callable(scripted):
            scripted = scripted(messages)
        if not isinstance(scripted, (BaseModel, dict)):
            raise PolicyResponseError("scripted policy output must be a model or object")
        payload = scripted.model_dump(mode="json") if isinstance(scripted, BaseModel) else scripted
        try:
            if decision_format is not None:
                decision_format.model_validate(payload)
            return PolicyDecision.model_validate(payload)
        except (ValidationError, TypeError, ValueError) as exc:
            raise PolicyResponseError(
                "scripted policy output failed PolicyDecision validation"
            ) from exc

    def decide_tool(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        tools: Sequence[dict[str, Any]],
        tool_models: Mapping[str, type[BaseModel]],
    ) -> tuple[str, BaseModel]:
        """Deterministic native-tool test double for controller tests."""
        if not tools or not tool_models:
            raise PolicyConfigurationError("native tool call requires tools")
        self.last_usage = Usage(policy_calls=1)
        self.calls.append(list(messages))
        if not self._decisions:
            raise PolicyResponseError("scripted policy has no decisions remaining")
        scripted = self._decisions.popleft()
        if isinstance(scripted, Exception):
            raise scripted
        if callable(scripted):
            scripted = scripted(messages)
        if isinstance(scripted, BaseModel):
            payload: Any = scripted.model_dump(mode="json")
        elif isinstance(scripted, dict):
            payload = scripted
        else:
            raise PolicyResponseError("scripted tool output must be a model or object")
        if "tool_name" in payload:
            name = str(payload["tool_name"])
            arguments = payload.get("arguments", {})
        elif len(tool_models) == 1:
            name = next(iter(tool_models))
            arguments = payload
        else:
            action_type = payload.get("action", {}).get("type")
            name = str(action_type or "").casefold()
            arguments = payload
        model = tool_models.get(name)
        if model is None:
            raise PolicyResponseError(f"unknown scripted tool {name!r}")
        try:
            return name, model.model_validate(arguments)
        except (ValidationError, TypeError, ValueError) as exc:
            raise PolicyResponseError("scripted tool arguments failed validation") from exc


def policy_decision_model(
    enabled_expansions: Sequence[ExpansionKind | str] = DEFAULT_ENABLED_EXPANSIONS,
) -> type[BaseModel]:
    """Build an OpenAI/Ollama-compatible schema for the enabled relations."""

    normalized = tuple(ExpansionKind(item) for item in enabled_expansions)
    if len(normalized) != len(set(normalized)):
        raise ValueError("enabled_expansions must not contain duplicates")
    return _policy_decision_model(normalized)


@lru_cache(maxsize=None)
def _policy_decision_model(
    enabled_expansions: tuple[ExpansionKind, ...],
) -> type[BaseModel]:
    signature = ",".join(item.value for item in enabled_expansions)
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    suffix = f"{len(enabled_expansions)}_{digest}" if enabled_expansions else "NO_EXPAND"
    action_types: Any = _WireSearchAction
    non_adjacent = tuple(
        item for item in enabled_expansions if item is not ExpansionKind.CHUNK_ADJACENT_CHUNK
    )
    if non_adjacent:
        enabled_enum = StrEnum(
            f"EnabledExpansionKind_{suffix}",
            {item.name: item.value for item in non_adjacent},
            module=__name__,
        )
        provider_expand = create_model(
            f"EnabledExpandAction_{suffix}",
            __base__=_WireNonAdjacentExpandAction,
            kind=(enabled_enum, ...),
        )
        action_types = action_types | provider_expand
    if ExpansionKind.CHUNK_ADJACENT_CHUNK in enabled_expansions:
        action_types = action_types | _WireAdjacentExpandAction
    action_types = action_types | _WireReadAction | _WireFinishAction
    return create_model(
        f"EnabledPolicyDecision_{suffix}",
        __base__=AgentModel,
        assessment=(_WireAssessment, ...),
        action=(action_types, ...),
    )


def usage_from_openai_response(response: Any) -> Usage:
    raw = getattr(response, "usage", None)
    if raw is None:
        return Usage(policy_calls=1)

    def get(name: str) -> int:
        value = raw.get(name, 0) if isinstance(raw, dict) else getattr(raw, name, 0)
        return int(value or 0)

    details = (
        raw.get("output_tokens_details")
        if isinstance(raw, dict)
        else getattr(raw, "output_tokens_details", None)
    )
    reasoning = 0
    if details is not None:
        reasoning = int(
            (details.get("reasoning_tokens", 0) if isinstance(details, dict) else getattr(details, "reasoning_tokens", 0))
            or 0
        )
    return Usage(
        policy_calls=1,
        input_tokens=get("input_tokens"),
        output_tokens=get("output_tokens"),
        reasoning_tokens=reasoning,
        total_tokens=get("total_tokens"),
    )


def response_refusal(response: Any) -> str | None:
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            refusal = getattr(content, "refusal", None)
            if refusal:
                return str(refusal)
    return None


def is_transient_provider_error(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}:
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return isinstance(status, int) and (status in {408, 409, 429} or status >= 500)
