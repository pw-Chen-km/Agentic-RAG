"""Native Ollama chat adapters for policy decisions and grounded answers."""

from __future__ import annotations

import json
import math
import os
import shlex
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from agentic_rag.agent.answer import (
    AnswerGenerationError,
    AnswerGenerator,
    _AnswerResponse,
    answer_provider_input,
)
from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    ExpansionKind,
    Message,
    PolicyDecisionOutput,
    ResolvedEvidence,
    Usage,
)
from agentic_rag.agent.policy import (
    PolicyClient,
    PolicyConfigurationError,
    PolicyResponseError,
    PolicyTransportError,
    StructuredModelT,
    _decision_type_for_format,
    _is_transient,
    policy_decision_model,
)
from agentic_rag.benchmark_profiles import AnswerMode

OllamaThink = bool | Literal["low", "medium", "high"] | None


class OllamaChatPolicy(PolicyClient):
    """Structured-output policy backed by Ollama's native ``/api/chat``."""

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        host: str | None = None,
        enabled_expansions: Sequence[
            ExpansionKind | str
        ] = DEFAULT_ENABLED_EXPANSIONS,
        temperature: float = 0.0,
        num_ctx: int = 32_768,
        think: OllamaThink = False,
        keep_alive: float | str | None = None,
        timeout_seconds: float | None = 120.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        direct_answer: bool = False,
        semantic_memory_v3: bool = False,
        semantic_memory_v31: bool = False,
    ) -> None:
        normalized_model = _normalize_model(
            model, error_type=PolicyConfigurationError
        )
        resolved_host = _resolve_ollama_host(
            host, error_type=PolicyConfigurationError
        )
        _validate_common_options(
            model=normalized_model,
            host=resolved_host,
            temperature=temperature,
            num_ctx=num_ctx,
            think=think,
            keep_alive=keep_alive,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            error_type=PolicyConfigurationError,
        )
        self.model = normalized_model
        self.host = resolved_host
        self.temperature = float(temperature)
        self.num_ctx = num_ctx
        self.think = think
        self.keep_alive = keep_alive
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.semantic_memory_v3 = semantic_memory_v3
        self.semantic_memory_v31 = semantic_memory_v31
        try:
            self.decision_format = policy_decision_model(
                enabled_expansions,
                direct_answer=direct_answer,
                semantic_memory_v3=semantic_memory_v3,
                semantic_memory_v31=semantic_memory_v31,
            )
        except (TypeError, ValueError) as exc:
            raise PolicyConfigurationError(
                "enabled_expansions contains an invalid expansion kind"
            ) from exc
        self._client = client if client is not None else _create_client(
            host=resolved_host,
            timeout_seconds=timeout_seconds,
            error_type=PolicyConfigurationError,
        )
        self.last_usage = Usage()

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
        except (ValidationError, ValueError, TypeError) as exc:
            raise PolicyResponseError(
                "Ollama response failed PolicyDecision validation"
            ) from exc

    def generate_structured(
        self,
        messages: Sequence[Message | dict[str, str]],
        response_model: type[StructuredModelT],
    ) -> StructuredModelT:
        """Generate and validate any Pydantic structured policy response."""

        self.last_usage = Usage()
        request = _chat_request(
            model=self.model,
            messages=messages,
            schema=response_model.model_json_schema(),
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            think=self.think,
            keep_alive=self.keep_alive,
        )
        response = self._call_with_retries(
            lambda: self._client.chat(**request)
        )
        self.last_usage = _extract_ollama_usage(
            response, policy_calls=1
        )
        content = _response_content(response)
        if content is None or not content.strip():
            raise PolicyResponseError(
                "Ollama response did not contain structured output"
            )
        try:
            return response_model.model_validate_json(content)
        except (ValidationError, ValueError, TypeError) as exc:
            raise PolicyResponseError(
                "Ollama response failed structured response validation"
            ) from exc

    def _call_with_retries(self, operation: Callable[[], Any]) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except (ValidationError, ValueError, TypeError) as exc:
                raise PolicyResponseError(
                    "Ollama client failed to parse the chat response: "
                    f"{_provider_error_detail(exc)}"
                ) from exc
            except Exception as exc:
                if (
                    attempt >= self.max_retries
                    or not _is_ollama_transient(exc)
                ):
                    raise PolicyTransportError(
                        _transport_error_message(
                            operation="policy",
                            model=self.model,
                            exc=exc,
                            attempts=attempt + 1,
                        )
                    ) from exc
                if self.retry_backoff_seconds:
                    time.sleep(
                        self.retry_backoff_seconds * (2**attempt)
                    )
        raise AssertionError("retry loop must return or raise")


class OllamaChatAnswerGenerator(AnswerGenerator):
    """Evidence-only answer generator backed by native Ollama chat."""

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        host: str | None = None,
        temperature: float = 0.0,
        num_ctx: int = 32_768,
        think: OllamaThink = False,
        keep_alive: float | str | None = None,
        timeout_seconds: float | None = 120.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        answer_mode: AnswerMode = AnswerMode.SHORT,
    ) -> None:
        normalized_model = _normalize_model(
            model, error_type=AnswerGenerationError
        )
        resolved_host = _resolve_ollama_host(
            host, error_type=AnswerGenerationError
        )
        _validate_common_options(
            model=normalized_model,
            host=resolved_host,
            temperature=temperature,
            num_ctx=num_ctx,
            think=think,
            keep_alive=keep_alive,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            error_type=AnswerGenerationError,
        )
        self.model = normalized_model
        self.host = resolved_host
        self.temperature = float(temperature)
        self.num_ctx = num_ctx
        self.think = think
        self.keep_alive = keep_alive
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.answer_mode = answer_mode
        self._client = client if client is not None else _create_client(
            host=resolved_host,
            timeout_seconds=timeout_seconds,
            error_type=AnswerGenerationError,
        )
        self.last_usage = Usage()

    def generate(
        self, question: str, evidence: Sequence[ResolvedEvidence]
    ) -> str:
        self.last_usage = Usage()
        if not evidence:
            raise AnswerGenerationError(
                "answer generation requires at least one resolved evidence item"
            )
        messages = answer_provider_input(
            question,
            evidence,
            answer_mode=self.answer_mode,
        )
        request = _chat_request(
            model=self.model,
            messages=messages,
            schema=_AnswerResponse.model_json_schema(),
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            think=self.think,
            keep_alive=self.keep_alive,
        )
        response = self._call_with_retries(
            lambda: self._client.chat(**request)
        )
        self.last_usage = _extract_ollama_usage(
            response, answer_calls=1
        )
        content = _response_content(response)
        if content is None or not content.strip():
            raise AnswerGenerationError(
                "Ollama response did not contain a structured answer"
            )
        try:
            parsed = _AnswerResponse.model_validate_json(content)
        except (ValidationError, ValueError, TypeError) as exc:
            raise AnswerGenerationError(
                "Ollama response failed answer validation"
            ) from exc
        if not parsed.answer.strip():
            raise AnswerGenerationError("Ollama returned a blank answer")
        return parsed.answer

    def _call_with_retries(self, operation: Callable[[], Any]) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                if (
                    attempt >= self.max_retries
                    or not _is_ollama_transient(exc)
                ):
                    raise AnswerGenerationError(
                        _transport_error_message(
                            operation="answer",
                            model=self.model,
                            exc=exc,
                            attempts=attempt + 1,
                        )
                    ) from exc
                if self.retry_backoff_seconds:
                    time.sleep(
                        self.retry_backoff_seconds * (2**attempt)
                    )
        raise AssertionError("retry loop must return or raise")


def _chat_request(
    *,
    model: str,
    messages: Sequence[Message | dict[str, str]],
    schema: dict[str, Any],
    temperature: float,
    num_ctx: int,
    think: OllamaThink,
    keep_alive: float | str | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            (
                message.as_openai_input()
                if isinstance(message, Message)
                else dict(message)
            )
            for message in messages
        ],
        "stream": False,
        "format": schema,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    # ``False`` explicitly disables thinking. Only ``None`` delegates to the
    # model/server default and therefore omits the provider field.
    if think is not None:
        request["think"] = think
    if keep_alive is not None:
        request["keep_alive"] = keep_alive
    return request


def _response_content(response: Any) -> str | None:
    message = _get_value(response, "message")
    content = _get_value(message, "content")
    return content if isinstance(content, str) else None


def _extract_ollama_usage(
    response: Any, *, policy_calls: int = 0, answer_calls: int = 0
) -> Usage:
    input_tokens = _integer_field(response, "prompt_eval_count")
    output_tokens = _integer_field(response, "eval_count")
    stage_usage: dict[str, int] = {}
    if policy_calls:
        stage_usage.update(
            policy_input_tokens=input_tokens,
            policy_output_tokens=output_tokens,
            policy_reasoning_tokens=0,
        )
    if answer_calls:
        stage_usage.update(
            answer_input_tokens=input_tokens,
            answer_output_tokens=output_tokens,
            answer_reasoning_tokens=0,
        )
    return Usage(
        policy_calls=policy_calls,
        answer_calls=answer_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        # Ollama exposes a separate thinking string, but currently does not
        # expose a separate reasoning-token count.
        reasoning_tokens=0,
        total_tokens=input_tokens + output_tokens,
        **stage_usage,
    )


def _integer_field(value: Any, name: str) -> int:
    raw = _get_value(value, name)
    try:
        return max(int(raw or 0), 0)
    except (TypeError, ValueError):
        return 0


def _get_value(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _is_ollama_transient(exc: Exception) -> bool:
    if _is_transient(exc):
        return True
    # The official client translates connect failures to ConnectionError but
    # lets other httpx transport/time-out exceptions propagate.
    names = {cls.__name__ for cls in type(exc).__mro__}
    return type(exc).__module__.split(".", 1)[0] == "httpx" and bool(
        names
        & {
            "TimeoutException",
            "TransportError",
            "NetworkError",
            "RemoteProtocolError",
        }
    )


def _transport_error_message(
    *,
    operation: str,
    model: str,
    exc: Exception,
    attempts: int,
) -> str:
    detail = _provider_error_detail(exc)
    if _status_code(exc) == 404:
        command = f"ollama pull {shlex.quote(model)}"
        return (
            f"Ollama {operation} request failed: {detail}. "
            f"Model {model!r} may not be installed; run `{command}`."
        )
    return (
        f"Ollama {operation} request failed after {attempts} attempt(s): "
        f"{detail}"
    )


def _provider_error_detail(exc: Exception) -> str:
    raw = getattr(exc, "error", None)
    detail = str(raw if raw not in (None, "") else exc).strip()
    return " ".join(detail.split()) or type(exc).__name__


def _status_code(exc: Exception) -> int | None:
    raw = getattr(exc, "status_code", None)
    if raw is None:
        raw = getattr(getattr(exc, "response", None), "status_code", None)
    return raw if isinstance(raw, int) else None


def _normalize_model(
    model: str, *, error_type: type[Exception]
) -> str:
    if not isinstance(model, str) or not model.strip():
        raise error_type("model must not be blank")
    return model.strip()


def _resolve_ollama_host(
    host: str | None, *, error_type: type[Exception]
) -> str | None:
    raw = host if host is not None else os.environ.get("OLLAMA_HOST")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise error_type("host must be omitted rather than blank")

    trimmed = raw.strip().rstrip("/")
    candidate = (
        trimmed if "://" in trimmed else f"http://{trimmed}"
    )
    parsed = urlparse(candidate)
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        raise error_type("Ollama host scheme must be http or https")
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if not hostname:
        raise error_type("Ollama host must include a hostname")
    try:
        parsed.port
    except ValueError as exc:
        raise error_type("Ollama host contains an invalid port") from exc
    if (
        any(character.isspace() for character in trimmed)
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        raise error_type(
            "Ollama host must not contain whitespace, credentials, "
            "a query, or a fragment"
        )
    if hostname == "ollama.com" or hostname.endswith(".ollama.com"):
        raise error_type(
            "Ollama Cloud does not support the structured outputs "
            "required by Agentic RAG; configure a local Ollama host"
        )
    return parsed._replace(scheme=scheme).geturl().rstrip("/")


def _create_client(
    *,
    host: str | None,
    timeout_seconds: float | None,
    error_type: type[Exception],
) -> Any:
    try:
        from ollama import Client
    except ImportError as exc:
        raise error_type(
            "Install the 'ollama' package to use the Ollama provider"
        ) from exc

    kwargs: dict[str, Any] = {}
    if host is not None:
        kwargs["host"] = host
    if timeout_seconds is not None:
        kwargs["timeout"] = timeout_seconds
    try:
        return Client(**kwargs)
    except Exception as exc:
        raise error_type("Unable to configure the Ollama client") from exc


def _validate_common_options(
    *,
    model: str,
    host: str | None,
    temperature: float,
    num_ctx: int,
    think: OllamaThink,
    keep_alive: float | str | None,
    timeout_seconds: float | None,
    max_retries: int,
    retry_backoff_seconds: float,
    error_type: type[Exception],
) -> None:
    if not isinstance(model, str) or not model.strip():
        raise error_type("model must not be blank")
    if host is not None and (
        not isinstance(host, str) or not host.strip()
    ):
        raise error_type("host must be omitted rather than blank")
    if isinstance(temperature, bool):
        raise error_type("temperature must be a finite non-negative number")
    try:
        normalized_temperature = float(temperature)
    except (TypeError, ValueError) as exc:
        raise error_type(
            "temperature must be a finite non-negative number"
        ) from exc
    if (
        not math.isfinite(normalized_temperature)
        or normalized_temperature < 0
    ):
        raise error_type("temperature must be a finite non-negative number")
    if (
        isinstance(num_ctx, bool)
        or not isinstance(num_ctx, int)
        or num_ctx <= 0
    ):
        raise error_type("num_ctx must be a positive integer")
    if (
        think is not None
        and type(think) is not bool
        and think not in {"low", "medium", "high"}
    ):
        raise error_type(
            "think must be null, false, true, 'low', 'medium', or 'high'"
        )
    if isinstance(keep_alive, bool):
        raise error_type("keep_alive must not be a boolean")
    if isinstance(keep_alive, str) and not keep_alive.strip():
        raise error_type("keep_alive must be omitted rather than blank")
    if isinstance(keep_alive, (int, float)) and not math.isfinite(
        float(keep_alive)
    ):
        raise error_type("keep_alive numeric value must be finite")
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool):
            raise error_type("timeout_seconds must be positive or null")
        try:
            normalized_timeout = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise error_type(
                "timeout_seconds must be positive or null"
            ) from exc
        if not math.isfinite(normalized_timeout) or normalized_timeout <= 0:
            raise error_type("timeout_seconds must be positive or null")
    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise error_type("max_retries must be non-negative")
    if isinstance(retry_backoff_seconds, bool):
        raise error_type("retry_backoff_seconds must be non-negative")
    try:
        normalized_backoff = float(retry_backoff_seconds)
    except (TypeError, ValueError) as exc:
        raise error_type(
            "retry_backoff_seconds must be non-negative"
        ) from exc
    if not math.isfinite(normalized_backoff) or normalized_backoff < 0:
        raise error_type("retry_backoff_seconds must be non-negative")
