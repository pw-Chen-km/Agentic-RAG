"""OpenAI-compatible structured-output Policy provider for vLLM."""
from __future__ import annotations

import json
import re
import time
import urllib.request
from collections.abc import Sequence
from typing import Any
from pydantic import BaseModel, ValidationError
from agentic_rag.agent.action_schema import policy_decision_from_constrained
from agentic_rag.agent.models import DEFAULT_ENABLED_EXPANSIONS, ExpansionKind, Message, PolicyDecision, Usage
from agentic_rag.agent.policy import PolicyConfigurationError, PolicyResponseError, PolicyTransportError, policy_decision_model
from agentic_rag.agent.tool_calling import decision_from_tool_call, tool_schema_sha256

class OpenAICompatibleChatPolicy:
    def __init__(self, *, model: str, base_url: str, api_key: str = "EMPTY", enabled_expansions=DEFAULT_ENABLED_EXPANSIONS, temperature: float = 0.0, timeout_seconds: float = 600.0, max_retries: int = 2, num_ctx: int = 32768, max_output_tokens: int = 2048, seed: int | None = None) -> None:
        self.model, self.base_url, self.api_key = model.strip(), base_url.rstrip("/"), api_key
        self.temperature, self.timeout_seconds, self.max_retries = float(temperature), timeout_seconds, max_retries
        self.num_ctx, self.max_output_tokens, self.seed = num_ctx, max_output_tokens, seed
        try:
            self.enabled_expansions = tuple(ExpansionKind(x) for x in enabled_expansions)
            self.decision_format = policy_decision_model(self.enabled_expansions)
        except (TypeError, ValueError) as exc:
            raise PolicyConfigurationError("enabled_expansions contains an invalid expansion kind") from exc
        self.last_usage = Usage()
        self.last_usage_metadata: dict[str, Any] = {}

    def decide(self, messages: Sequence[Message | dict[str, Any]], *, decision_format: type[BaseModel] | None = None, tools: Sequence[dict[str, Any]] | None = None) -> PolicyDecision:
        response_model = decision_format or self.decision_format
        wire_messages = [m.as_openai_input() if isinstance(m, Message) else dict(m) for m in messages]
        payload = {"model": self.model, "messages": wire_messages, "temperature": self.temperature, "top_p": 1, "max_tokens": self.max_output_tokens, "stream": False, "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}}
        if self.seed is not None:
            payload["seed"] = self.seed
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "required"
            payload["parallel_tool_calls"] = False
        else:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "policy_decision", "schema": response_model.model_json_schema(), "strict": True}}
        raw = self._request(payload)
        usage = raw.get("usage") or {}
        prompt = _int_or_none(usage.get("prompt_tokens")); completion = _int_or_none(usage.get("completion_tokens")); total = _int_or_none(usage.get("total_tokens"))
        self.last_usage = Usage(policy_calls=1, input_tokens=prompt or 0, output_tokens=completion or 0, total_tokens=total if total is not None else (prompt or 0) + (completion or 0))
        self.last_usage_metadata = {"provider": "openai_compatible", "model": self.model, "base_url": self.base_url, "provider_input_tokens": prompt, "provider_output_tokens": completion, "provider_total_tokens": total, "reasoning_tokens": "unavailable", "native_tool_calling": bool(tools), "constrained_single_decision": not bool(tools), "tool_schema_sha256": tool_schema_sha256(tools or []), "tool_schema_token_estimate": _token_estimate(tools or []), "raw_provider_output": raw}
        try:
            if tools:
                message = raw["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                self.last_usage_metadata.update({"tool_call_count": len(calls), "raw_tool_calls": calls, "tool_call_names": [str((call.get("function") or {}).get("name") or "") for call in calls]})
                if len(calls) != 1:
                    raise PolicyResponseError(f"OpenAI-compatible native tool response must contain exactly one tool call; got {len(calls)}")
                decision = decision_from_tool_call(calls[0], tools)
                self.last_usage_metadata["selected_action"] = calls[0]["function"]["name"]
                return decision
            content = raw["choices"][0]["message"]["content"]
            parsed = response_model.model_validate_json(content)
            payload = parsed.model_dump(mode="json")
            self.last_usage_metadata.update({"decision_count": 1, "raw_structured_decision": payload,
                                             "selected_action": (payload.get("action") or {}).get("name")})
            if "name" in (payload.get("action") or {}):
                return policy_decision_from_constrained(parsed)
            return PolicyDecision.model_validate(payload)
        except (KeyError, IndexError, TypeError, ValidationError, ValueError) as exc:
            raise PolicyResponseError("OpenAI-compatible response failed PolicyDecision validation") from exc

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode()
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}, method="POST")
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                last = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
        raise PolicyTransportError(f"OpenAI-compatible Policy request failed: {last}") from last

def _int_or_none(value: Any) -> int | None:
    try:
        return max(int(value), 0) if value is not None else None
    except (TypeError, ValueError):
        return None


def _token_estimate(value: Any) -> int:
    return len(re.findall(r"(?u)\b\w+\b|[^\w\s]", str(value)))
