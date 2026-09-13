"""Native Ollama structured-output Policy provider."""

from __future__ import annotations

import math
import os
import shlex
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal
from urllib.parse import urlparse

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
)

OllamaThink = bool | Literal["low", "medium", "high"] | None


class OllamaChatPolicy:
    """Structured-output Policy backed by Ollama's native chat API."""

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        host: str | None = None,
        enabled_expansions: Sequence[ExpansionKind | str] = DEFAULT_ENABLED_EXPANSIONS,
        temperature: float = 0.0,
        num_ctx: int = 32_768,
        think: OllamaThink = False,
        keep_alive: float | str | None = None,
        timeout_seconds: float | None = 300.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        max_output_tokens: int = 2_048,
    ) -> None:
        self.model = _normalize_model(model)
        self.host = _resolve_host(host)
        _validate_options(
            temperature=temperature,
            num_ctx=num_ctx,
            think=think,
            keep_alive=keep_alive,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        self.temperature = float(temperature)
        self.num_ctx = num_ctx
        self.think = think
        self.keep_alive = keep_alive
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        if isinstance(max_output_tokens, bool) or max_output_tokens < 1:
            raise PolicyConfigurationError("max_output_tokens must be positive")
        self.max_output_tokens = int(max_output_tokens)
        try:
            self.enabled_expansions = tuple(ExpansionKind(item) for item in enabled_expansions)
            self.decision_format = policy_decision_model(self.enabled_expansions)
        except (TypeError, ValueError) as exc:
            raise PolicyConfigurationError(
                "enabled_expansions contains an invalid expansion kind"
            ) from exc
        self._client = client if client is not None else _create_client(
            host=self.host, timeout_seconds=timeout_seconds
        )
        self.last_usage = Usage()
        self.last_usage_metadata: dict[str, Any] = {}

    def decide(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        decision_format: type[BaseModel] | None = None,
    ) -> PolicyDecision:
        response_model = decision_format or self.decision_format
        self.last_usage = Usage()
        self.last_usage_metadata = {
            "provider": "ollama",
            "model": self.model,
            "host": self.host,
            "temperature": self.temperature,
            "think": self.think,
            "num_ctx": self.num_ctx,
            "max_output_tokens": self.max_output_tokens,
            "reasoning_tokens": "unavailable",
        }
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                message.as_openai_input() if isinstance(message, Message) else dict(message)
                for message in messages
            ],
            "stream": False,
            "format": response_model.model_json_schema(),
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_output_tokens,
            },
        }
        if self.think is not None:
            request["think"] = self.think
        if self.keep_alive is not None:
            request["keep_alive"] = self.keep_alive
        response = self._call_with_retries(lambda: self._client.chat(**request))
        self.last_usage = _usage(response)
        self.last_usage_metadata.update(
            {
                "provider_input_tokens": _optional_integer(response, "prompt_eval_count"),
                "provider_output_tokens": _optional_integer(response, "eval_count"),
                "provider_total_tokens": _optional_integer(response, "prompt_eval_count")
                + (_optional_integer(response, "eval_count") or 0)
                if _optional_integer(response, "prompt_eval_count") is not None
                and _optional_integer(response, "eval_count") is not None
                else None,
            }
        )
        content = _value(_value(response, "message"), "content")
        if not isinstance(content, str) or not content.strip():
            raise PolicyResponseError("Ollama response did not contain structured output")
        try:
            parsed = response_model.model_validate_json(content)
            return PolicyDecision.model_validate(parsed.model_dump(mode="json"))
        except (ValidationError, ValueError, TypeError) as exc:
            raise PolicyResponseError(
                "Ollama response failed PolicyDecision validation"
            ) from exc

    def _call_with_retries(self, operation: Callable[[], Any]) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except (ValidationError, ValueError, TypeError) as exc:
                raise PolicyResponseError(
                    f"Ollama client failed to parse the response: {_error_detail(exc)}"
                ) from exc
            except Exception as exc:
                if attempt >= self.max_retries or not _is_transient(exc):
                    raise PolicyTransportError(
                        _transport_error(self.model, exc, attempt + 1)
                    ) from exc
                if self.retry_backoff_seconds:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise AssertionError("retry loop must return or raise")


def _usage(response: Any) -> Usage:
    input_tokens = _integer(response, "prompt_eval_count")
    output_tokens = _integer(response, "eval_count")
    return Usage(
        policy_calls=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _integer(value: Any, name: str) -> int:
    try:
        return max(int(_value(value, name) or 0), 0)
    except (TypeError, ValueError):
        return 0


def _optional_integer(value: Any, name: str) -> int | None:
    raw = _value(value, name)
    if raw is None:
        return None
    try:
        return max(int(raw), 0)
    except (TypeError, ValueError):
        return None


def _value(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _is_transient(exc: Exception) -> bool:
    if is_transient_provider_error(exc):
        return True
    names = {cls.__name__ for cls in type(exc).__mro__}
    return type(exc).__module__.split(".", 1)[0] == "httpx" and bool(
        names & {"TimeoutException", "TransportError", "NetworkError", "RemoteProtocolError"}
    )


def _transport_error(model: str, exc: Exception, attempts: int) -> str:
    detail = _error_detail(exc)
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 404:
        return (
            f"Ollama Policy request failed: {detail}. Model {model!r} may not be "
            f"installed; run `ollama pull {shlex.quote(model)}`."
        )
    return f"Ollama Policy request failed after {attempts} attempt(s): {detail}"


def _error_detail(exc: Exception) -> str:
    raw = getattr(exc, "error", None)
    detail = str(raw if raw not in (None, "") else exc).strip()
    return " ".join(detail.split()) or type(exc).__name__


def _normalize_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise PolicyConfigurationError("model must not be blank")
    return model.strip()


def _resolve_host(host: str | None) -> str | None:
    raw = host if host is not None else os.environ.get("OLLAMA_HOST")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise PolicyConfigurationError("host must be omitted rather than blank")
    trimmed = raw.strip().rstrip("/")
    candidate = trimmed if "://" in trimmed else f"http://{trimmed}"
    parsed = urlparse(candidate)
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    try:
        parsed.port
    except ValueError as exc:
        raise PolicyConfigurationError("Ollama host contains an invalid port") from exc
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not hostname
        or any(character.isspace() for character in trimmed)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise PolicyConfigurationError("Ollama host must be a plain HTTP(S) URL")
    if hostname == "ollama.com" or hostname.endswith(".ollama.com"):
        raise PolicyConfigurationError(
            "Ollama Cloud does not support the required structured outputs"
        )
    return parsed._replace(scheme=parsed.scheme.casefold()).geturl().rstrip("/")


def _create_client(*, host: str | None, timeout_seconds: float | None) -> Any:
    try:
        from ollama import Client
    except ImportError as exc:
        raise PolicyConfigurationError("Install the 'ollama' package") from exc
    kwargs: dict[str, Any] = {}
    if host is not None:
        kwargs["host"] = host
    if timeout_seconds is not None:
        kwargs["timeout"] = timeout_seconds
    try:
        return Client(**kwargs)
    except Exception as exc:
        raise PolicyConfigurationError("Unable to configure the Ollama client") from exc


def _validate_options(
    *,
    temperature: float,
    num_ctx: int,
    think: OllamaThink,
    keep_alive: float | str | None,
    timeout_seconds: float | None,
    max_retries: int,
    retry_backoff_seconds: float,
) -> None:
    if isinstance(temperature, bool) or not math.isfinite(float(temperature)) or float(temperature) < 0:
        raise PolicyConfigurationError("temperature must be finite and non-negative")
    if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx <= 0:
        raise PolicyConfigurationError("num_ctx must be a positive integer")
    if think is not None and type(think) is not bool and think not in {"low", "medium", "high"}:
        raise PolicyConfigurationError("think has an unsupported value")
    if isinstance(keep_alive, bool) or (isinstance(keep_alive, str) and not keep_alive.strip()):
        raise PolicyConfigurationError("keep_alive has an unsupported value")
    if isinstance(keep_alive, (int, float)) and not math.isfinite(float(keep_alive)):
        raise PolicyConfigurationError("keep_alive must be finite")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise PolicyConfigurationError("timeout_seconds must be positive")
    if max_retries < 0 or retry_backoff_seconds < 0:
        raise PolicyConfigurationError("retry settings must be non-negative")
