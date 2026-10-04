"""Optional source-grounded evidence Reader for observation experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_rag.agent.models import Message, Observation, Usage


class ReaderItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(min_length=2)
    role: Literal["direct_answer", "bridge", "context", "uncertain"]
    claim: str = Field(min_length=1, max_length=500)
    quote: str = Field(min_length=1, max_length=1200)
    confidence: Literal["high", "medium", "low"] = "medium"


class ReaderSufficiency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolved: list[str] = Field(default_factory=list, max_length=5)
    still_missing: list[str] = Field(default_factory=list, max_length=5)
    recommendation: Literal["continue", "finish", "uncertain"] = "uncertain"
    confidence: Literal["high", "medium", "low"] = "low"


class ReaderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ReaderItem] = Field(default_factory=list, max_length=12)
    sufficiency: ReaderSufficiency | None = None


class EvidenceReader(Protocol):
    last_usage: Usage

    def read(
        self,
        *,
        question: str,
        previous_context: str,
        observation: Observation,
        reader_assessment: bool,
    ) -> ReaderResult:
        ...


class NullReader:
    """Explicit no-op Reader used by raw/program modes and tests."""

    last_usage = Usage()

    def read(self, **_: Any) -> ReaderResult:
        return ReaderResult()


class ScriptedReader:
    """Small deterministic Reader double for unit tests and replay."""

    def __init__(self, results: Sequence[ReaderResult | Exception]) -> None:
        self._results = list(results)
        self.last_usage = Usage()

    def read(self, **_: Any) -> ReaderResult:
        self.last_usage = Usage(policy_calls=1)
        if not self._results:
            raise RuntimeError("scripted reader has no result")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class OllamaEvidenceReader:
    """Structured-output Reader backed by the same Ollama API as the Agent."""

    def __init__(
        self,
        *,
        model: str,
        host: str,
        client: Any | None = None,
        temperature: float = 0.0,
        num_ctx: int = 32_768,
        think: bool | str | None = False,
        timeout_seconds: float = 300.0,
        keep_alive: str | int | float | None = None,
    ) -> None:
        from ollama import Client

        self.model = model
        self.host = host
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.think = think
        self.keep_alive = keep_alive
        self.client = client or Client(host=host, timeout=timeout_seconds)
        self.last_usage = Usage()

    def read(
        self,
        *,
        question: str,
        previous_context: str,
        observation: Observation,
        reader_assessment: bool,
    ) -> ReaderResult:
        system = (
            "You are an evidence reader inside a retrieval agent. "
            "Summarize only text present in the new observation. Every item "
            "must quote its source and preserve the supplied source_ref. "
            "Do not answer the question and do not invent facts."
        )
        if reader_assessment:
            system += (
                " Also report which information is resolved and still missing, "
                "but mark uncertainty explicitly."
            )
        payload = {
            "question": question,
            "previous_context": previous_context,
            "new_observation": observation.model_dump(mode="json"),
        }
        schema = ReaderResult.model_json_schema()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "stream": False,
            "format": schema,
            "think": self.think,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if self.keep_alive is not None:
            request["keep_alive"] = self.keep_alive
        response = self.client.chat(**request)
        prompt_tokens = int(getattr(response, "prompt_eval_count", 0) or 0)
        output_tokens = int(getattr(response, "eval_count", 0) or 0)
        self.last_usage = Usage(
            policy_calls=1,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
        )
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Reader returned no structured content")
        try:
            result = ReaderResult.model_validate_json(content)
        except (ValidationError, ValueError, TypeError) as exc:
            raise ValueError("Reader returned invalid evidence JSON") from exc
        return result


def reader_items_as_dicts(result: ReaderResult) -> tuple[dict[str, Any], ...]:
    return tuple(item.model_dump(mode="json") for item in result.items)
