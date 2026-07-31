"""Provider-neutral policy interface and OpenAI Responses implementation."""

from __future__ import annotations

import hashlib
import os
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from functools import lru_cache
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError, create_model

from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    AgentModel,
    AssessmentStatus,
    EvidenceAssessment,
    ExpandAction,
    ExpansionDirection,
    ExpansionKind,
    FinishAction,
    Message,
    PolicyDecision,
    ReadAction,
    SearchAction,
    SearchMethod,
    SearchTarget,
    Usage,
)


class PolicyError(RuntimeError):
    """Base class for policy failures that terminate an episode cleanly."""


class PolicyConfigurationError(PolicyError):
    """The provider client cannot be configured."""


class PolicyResponseError(PolicyError):
    """The provider completed but did not return a valid decision."""


class PolicyTransportError(PolicyError):
    """The provider remained unavailable after bounded retries."""


@runtime_checkable
class PolicyClient(Protocol):
    last_usage: Usage

    def decide(
        self, messages: Sequence[Message | dict[str, str]]
    ) -> PolicyDecision:
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


ScriptedDecision = (
    PolicyDecision
    | dict[str, Any]
    | Exception
    | Callable[[Sequence[Message | dict[str, str]]], PolicyDecision]
)


class ScriptedPolicy:
    """Deterministic zero-cost policy used by unit and integration tests."""

    def __init__(self, decisions: Iterable[ScriptedDecision]) -> None:
        self._decisions = deque(decisions)
        self.calls: list[list[Message | dict[str, str]]] = []
        self.last_usage = Usage()

    def decide(
        self, messages: Sequence[Message | dict[str, str]]
    ) -> PolicyDecision:
        self.calls.append(list(messages))
        self.last_usage = Usage(policy_calls=1)
        if not self._decisions:
            raise PolicyResponseError("scripted policy has no decisions remaining")
        scripted = self._decisions.popleft()
        if isinstance(scripted, Exception):
            raise scripted
        if callable(scripted):
            scripted = scripted(messages)
        if isinstance(scripted, PolicyDecision):
            return scripted
        try:
            return PolicyDecision.model_validate(scripted)
        except (ValidationError, TypeError, ValueError) as exc:
            raise PolicyResponseError(
                "scripted policy output failed PolicyDecision validation"
            ) from exc


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
        self.decision_format = policy_decision_model(
            self.enabled_expansions
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
        self, messages: Sequence[Message | dict[str, str]]
    ) -> PolicyDecision:
        # Never let usage from a previous decision leak into a failed call.
        self.last_usage = Usage()
        provider_input = [
            (
                message.as_openai_input()
                if isinstance(message, Message)
                else dict(message)
            )
            for message in messages
        ]
        response = self._call_with_retries(
            lambda: self._client.responses.parse(
                model=self.model,
                input=provider_input,
                text_format=self.decision_format,
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
                f"OpenAI response did not contain a parsed PolicyDecision{detail}"
            )
        try:
            return PolicyDecision.model_validate(
                parsed.model_dump(mode="json")
                if isinstance(parsed, BaseModel)
                else parsed
            )
        except Exception as exc:
            raise PolicyResponseError(
                "OpenAI response failed PolicyDecision validation"
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
) -> type[BaseModel]:
    """Build the provider schema for exactly the enabled EXPAND kinds.

    The provider-specific instance is normalized back into the stable
    :class:`PolicyDecision` immediately after parsing.
    """

    normalized = _normalize_enabled_expansions(enabled_expansions)
    return _policy_decision_model(normalized)


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
) -> type[BaseModel]:
    signature = ",".join(item.value for item in enabled_expansions)
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    suffix = (
        f"{len(enabled_expansions)}_{digest}"
        if enabled_expansions
        else "NO_EXPAND"
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
    action_types = action_types | _WireReadAction | _WireFinishAction

    return create_model(
        f"EnabledPolicyDecision_{suffix}",
        __base__=AgentModel,
        assessment=(_WireEvidenceAssessment, ...),
        action=(action_types, ...),
    )
