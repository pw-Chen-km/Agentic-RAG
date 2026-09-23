"""OpenAI Responses API Policy provider."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    ExpansionKind,
    Message,
    PolicyDecision,
    Usage,
)
from agentic_rag.agent.policy import (
    PolicyConfigurationError,
    PolicyResponseError,
    PolicyTransportError,
    is_transient_provider_error,
    policy_decision_model,
    response_refusal,
    usage_from_openai_response,
)


class OpenAIResponsesPolicy:
    """Structured-output Policy backed by OpenAI Responses."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-luna",
        client: Any | None = None,
        enabled_expansions: Sequence[ExpansionKind | str] = DEFAULT_ENABLED_EXPANSIONS,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        if not model.strip():
            raise PolicyConfigurationError("model must not be blank")
        if max_retries < 0:
            raise PolicyConfigurationError("max_retries must be non-negative")
        if retry_backoff_seconds < 0:
            raise PolicyConfigurationError("retry_backoff_seconds must be non-negative")
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.enabled_expansions = tuple(ExpansionKind(item) for item in enabled_expansions)
        self.decision_format = policy_decision_model(self.enabled_expansions)
        self._client = client if client is not None else self._create_client()
        self.last_usage = Usage()

    @staticmethod
    def _create_client() -> Any:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise PolicyConfigurationError("OPENAI_API_KEY is required for OpenAI Policy")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PolicyConfigurationError(
                "Install the 'openai' package to use OpenAIResponsesPolicy"
            ) from exc
        return OpenAI(api_key=api_key, max_retries=0)

    def decide(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        decision_format: type[BaseModel] | None = None,
    ) -> BaseModel:
        response_model = decision_format or self.decision_format
        self.last_usage = Usage()
        provider_input = [
            message.as_openai_input() if isinstance(message, Message) else dict(message)
            for message in messages
        ]
        response = self._call_with_retries(
            lambda: self._client.responses.parse(
                model=self.model,
                input=provider_input,
                text_format=response_model,
                reasoning={"context": "current_turn"},
                store=False,
            )
        )
        self.last_usage = usage_from_openai_response(response)
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            refusal = response_refusal(response)
            detail = f": {refusal}" if refusal else ""
            raise PolicyResponseError(
                "OpenAI response did not contain parsed structured output" + detail
            )
        payload = parsed.model_dump(mode="json") if isinstance(parsed, BaseModel) else parsed
        try:
            if decision_format is not None:
                return decision_format.model_validate(payload)
            return PolicyDecision.model_validate(payload)
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
                if attempt >= self.max_retries or not is_transient_provider_error(exc):
                    raise PolicyTransportError(
                        f"OpenAI Policy request failed after {attempt + 1} attempt(s)"
                    ) from exc
                if self.retry_backoff_seconds:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise AssertionError("retry loop must return or raise")
