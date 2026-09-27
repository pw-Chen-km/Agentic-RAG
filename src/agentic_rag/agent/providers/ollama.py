"""Native Ollama structured-output Policy provider."""

from __future__ import annotations

import json
import math
import os
import shlex
import time
from collections.abc import Callable, Mapping, Sequence
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
OllamaOutputMode = Literal["structured", "native_tools"]


class OllamaChatPolicy:
    """Structured-output Policy backed by Ollama's native chat API."""

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        host: str | None = None,
        output_mode: OllamaOutputMode = "structured",
        enabled_expansions: Sequence[ExpansionKind | str] = DEFAULT_ENABLED_EXPANSIONS,
        temperature: float = 0.0,
        num_ctx: int = 32_768,
        think: OllamaThink = False,
        keep_alive: float | str | None = None,
        timeout_seconds: float | None = 300.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
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
            output_mode=output_mode,
        )
        self.temperature = float(temperature)
        self.output_mode = output_mode
        self.num_ctx = num_ctx
        self.think = think
        self.keep_alive = keep_alive
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
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

    def decide(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        decision_format: type[BaseModel] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_models: Mapping[str, type[BaseModel]] | None = None,
        output_mode: OllamaOutputMode | None = None,
    ) -> PolicyDecision:
        mode = output_mode or self.output_mode
        if mode == "native_tools":
            return self._decide_with_native_tools(
                messages,
                decision_format=decision_format,
                tools=tools,
                tool_models=tool_models,
            )
        if mode != "structured":
            raise PolicyConfigurationError(
                "output_mode must be 'structured' or 'native_tools'"
            )
        response_model = decision_format or self.decision_format
        self.last_usage = Usage()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                message.as_openai_input() if isinstance(message, Message) else dict(message)
                for message in messages
            ],
            "stream": False,
            "format": response_model.model_json_schema(),
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if self.think is not None:
            request["think"] = self.think
        if self.keep_alive is not None:
            request["keep_alive"] = self.keep_alive
        response = self._call_with_retries(lambda: self._client.chat(**request))
        self.last_usage = _usage(response)
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

    def _decide_with_native_tools(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        decision_format: type[BaseModel] | None,
        tools: Sequence[dict[str, Any]] | None,
        tool_models: Mapping[str, type[BaseModel]] | None,
    ) -> PolicyDecision:
        if not tools:
            raise PolicyConfigurationError(
                "native_tools output mode requires at least one tool"
            )
        models = dict(tool_models or {})
        if not models and len(tools) == 1 and decision_format is not None:
            name = _tool_name(tools[0])
            if name is not None:
                models[name] = decision_format
        if not models:
            raise PolicyConfigurationError(
                "native_tools output mode requires tool_models"
            )
        self.last_usage = Usage()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                message.as_openai_input() if isinstance(message, Message) else dict(message)
                for message in messages
            ],
            "stream": False,
            "tools": list(tools),
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if self.think is not None:
            request["think"] = self.think
        if self.keep_alive is not None:
            request["keep_alive"] = self.keep_alive
        response = self._call_with_retries(lambda: self._client.chat(**request))
        self.last_usage = _usage(response)
        name, arguments = _extract_tool_call(response)
        model = models.get(name)
        if model is None:
            raise PolicyResponseError(f"Ollama returned an unknown tool {name!r}")
        try:
            parsed = model.model_validate(arguments)
            payload = parsed.model_dump(mode="json")
            if decision_format is not None:
                payload = decision_format.model_validate(payload).model_dump(mode="json")
            return PolicyDecision.model_validate(payload)
        except (ValidationError, ValueError, TypeError) as exc:
            raise PolicyResponseError(
                f"Ollama tool {name!r} arguments failed PolicyDecision validation"
            ) from exc

    def decide_tool(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        tools: Sequence[dict[str, Any]],
        tool_models: Mapping[str, type[BaseModel]],
    ) -> tuple[str, BaseModel]:
        """Call Ollama's native tool interface and validate one tool call."""
        if not tools or not tool_models:
            raise PolicyConfigurationError("native tool call requires tools")
        self.last_usage = Usage()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                message.as_openai_input() if isinstance(message, Message) else dict(message)
                for message in messages
            ],
            "stream": False,
            "tools": list(tools),
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if self.think is not None:
            request["think"] = self.think
        if self.keep_alive is not None:
            request["keep_alive"] = self.keep_alive
        response = self._call_with_retries(lambda: self._client.chat(**request))
        self.last_usage = _usage(response)
        name, arguments = _extract_tool_call(response)
        model = tool_models.get(name)
        if model is None:
            raise PolicyResponseError(f"Ollama returned an unknown tool {name!r}")
        try:
            return name, model.model_validate(arguments)
        except (ValidationError, ValueError, TypeError) as exc:
            raise PolicyResponseError(
                f"Ollama tool {name!r} arguments failed validation"
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


def _tool_name(tool: Any) -> str | None:
    function = _value(tool, "function")
    name = _value(function, "name")
    return name.strip() if isinstance(name, str) and name.strip() else None


def _extract_tool_call(response: Any) -> tuple[str, dict[str, Any]]:
    message = _value(response, "message")
    calls = _value(message, "tool_calls") or []
    if not calls:
        raise PolicyResponseError("Ollama response did not contain a native tool call")
    if len(calls) != 1:
        raise PolicyResponseError("Ollama response contained more than one native tool call")
    function = _value(calls[0], "function")
    name = _tool_name(calls[0])
    arguments = _value(function, "arguments")
    if name is None:
        raise PolicyResponseError("Ollama native tool call has no function name")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise PolicyResponseError("Ollama native tool arguments are not JSON") from exc
    if not isinstance(arguments, dict):
        raise PolicyResponseError("Ollama native tool arguments must be an object")
    return name, _decode_nested_json(arguments)


def _decode_nested_json(value: Any) -> Any:
    """Decode object/array strings emitted by some Qwen/Ollama models."""
    if isinstance(value, dict):
        return {key: _decode_nested_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_nested_json(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return _decode_nested_json(json.loads(stripped))
            except json.JSONDecodeError:
                return value
    return value


def _integer(value: Any, name: str) -> int:
    try:
        return max(int(_value(value, name) or 0), 0)
    except (TypeError, ValueError):
        return 0


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
    output_mode: OllamaOutputMode,
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
    if output_mode not in {"structured", "native_tools"}:
        raise PolicyConfigurationError(
            "output_mode must be 'structured' or 'native_tools'"
        )
