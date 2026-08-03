"""Evidence-only answer generation interfaces."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.agent.models import ResolvedEvidence, Usage
from agentic_rag.agent.policy import _extract_usage, _is_transient
from agentic_rag.benchmark_profiles import AnswerMode


class AnswerGenerationError(RuntimeError):
    """The fixed answer generator could not produce a grounded answer."""


@runtime_checkable
class AnswerGenerator(Protocol):
    last_usage: Usage

    def generate(
        self, question: str, evidence: Sequence[ResolvedEvidence]
    ) -> str:
        """Answer using only the explicitly resolved evidence."""


class _AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)


def answer_provider_input(
    question: str,
    evidence: Sequence[ResolvedEvidence],
    *,
    answer_mode: AnswerMode = AnswerMode.SHORT,
) -> list[dict[str, str]]:
    """Build the exact evidence-only provider messages.

    Keeping message construction pure makes live calls and persisted workflow
    traces use the same representation without recording HTTP headers or
    provider credentials.
    """

    evidence_payload = [item.model_dump(mode="json") for item in evidence]
    response_instruction = (
        "Provide a complete, well-supported answer. Include the explanation "
        "needed to satisfy a long-form question, but do not add unsupported "
        "claims."
        if answer_mode is AnswerMode.LONG
        else "Return a concise answer."
    )
    return [
        {
            "role": "system",
            "content": (
                "Answer the question using only the supplied evidence. "
                f"Do not use outside knowledge. {response_instruction}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"question": question, "evidence": evidence_payload},
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


class ScriptedAnswerGenerator:
    """Sequence-backed deterministic answer generator for tests."""

    def __init__(
        self,
        answers: str
        | Iterable[str]
        | Callable[[str, Sequence[ResolvedEvidence]], str],
    ) -> None:
        self._callable = answers if callable(answers) else None
        if self._callable is None:
            values = [answers] if isinstance(answers, str) else list(answers)
            self._answers: deque[str] = deque(values)
        else:
            self._answers = deque()
        self.calls: list[tuple[str, list[ResolvedEvidence]]] = []
        self.last_usage = Usage()

    def generate(
        self, question: str, evidence: Sequence[ResolvedEvidence]
    ) -> str:
        copied_evidence = list(evidence)
        self.calls.append((question, copied_evidence))
        self.last_usage = Usage(answer_calls=1)
        if self._callable is not None:
            answer = self._callable(question, evidence)
        else:
            if not self._answers:
                raise AnswerGenerationError(
                    "scripted answer generator has no answers remaining"
                )
            answer = self._answers.popleft()
        if not answer.strip():
            raise AnswerGenerationError("answer must not be blank")
        return answer


class FakeAnswerGenerator(ScriptedAnswerGenerator):
    """Named convenience fake with one deterministic answer."""

    def __init__(self, answer: str) -> None:
        super().__init__(answer)


class OpenAIResponsesAnswerGenerator:
    """Fixed evidence-only generator backed by OpenAI Responses."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-terra",
        client: Any | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        answer_mode: AnswerMode = AnswerMode.SHORT,
    ) -> None:
        if not model.strip():
            raise AnswerGenerationError("model must not be blank")
        if max_retries < 0:
            raise AnswerGenerationError("max_retries must be non-negative")
        if retry_backoff_seconds < 0:
            raise AnswerGenerationError(
                "retry_backoff_seconds must be non-negative"
            )
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.answer_mode = answer_mode
        self._client = client if client is not None else self._create_client()
        self.last_usage = Usage()

    @staticmethod
    def _create_client() -> Any:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise AnswerGenerationError(
                "OPENAI_API_KEY is required for OpenAI answer generation"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AnswerGenerationError(
                "Install the 'openai' package to use the OpenAI answer generator"
            ) from exc
        # Retry exactly at this adapter boundary; disable nested SDK retries.
        return OpenAI(api_key=api_key, max_retries=0)

    def generate(
        self, question: str, evidence: Sequence[ResolvedEvidence]
    ) -> str:
        # Never let usage from a previous generation leak into a failed call.
        self.last_usage = Usage()
        if not evidence:
            raise AnswerGenerationError(
                "answer generation requires at least one resolved evidence item"
            )
        provider_input = answer_provider_input(
            question,
            evidence,
            answer_mode=self.answer_mode,
        )
        response = self._call_with_retries(
            lambda: self._client.responses.parse(
                model=self.model,
                input=provider_input,
                text_format=_AnswerResponse,
                store=False,
            )
        )
        self.last_usage = _extract_usage(response, answer_calls=1)
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise AnswerGenerationError(
                "OpenAI response did not contain a parsed answer"
            )
        try:
            parsed_answer = (
                parsed
                if isinstance(parsed, _AnswerResponse)
                else _AnswerResponse.model_validate(parsed)
            )
        except Exception as exc:
            raise AnswerGenerationError(
                "OpenAI response failed answer validation"
            ) from exc
        return parsed_answer.answer

    def _call_with_retries(self, operation: Callable[[], Any]) -> Any:
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                if attempt >= self.max_retries or not _is_transient(exc):
                    raise AnswerGenerationError(
                        "OpenAI answer request failed after "
                        f"{attempt + 1} attempt(s)"
                    ) from exc
                if self.retry_backoff_seconds:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise AssertionError("retry loop must return or raise")
