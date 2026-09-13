"""Post-episode benchmark answer evaluation.

Gold answers enter the system only through :class:`EpisodeEvaluator`.  This
module deliberately has no dependency on the retrieval substrate, policy
context, or agent controller, which keeps gold labels outside target-agent
execution.
"""

from __future__ import annotations

import re
import string
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
)

from agentic_rag.evaluation.profiles import (
    DatasetProfile,
    EvaluationMetric,
    get_dataset_profile,
)


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_ASCII_PUNCTUATION = str.maketrans("", "", string.punctuation)


class EvaluationModel(BaseModel):
    """Strict base model for persisted evaluation records."""

    model_config = ConfigDict(extra="forbid")


class JudgeMessage(EvaluationModel):
    """One provider-neutral judge prompt message."""

    role: Literal["system", "user"]
    content: str

    def as_provider_input(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class JudgeUsage(EvaluationModel):
    """Exact provider-reported usage for one judge request."""

    calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    def __add__(self, other: object) -> "JudgeUsage":
        if not isinstance(other, JudgeUsage):
            return NotImplemented
        return JudgeUsage(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class JudgeResponse(EvaluationModel):
    """Normalized response returned by any judge provider."""

    correct: bool
    raw_output: JsonValue = None
    model: str | None = None
    usage: JudgeUsage = Field(default_factory=JudgeUsage)
    request_options: dict[str, JsonValue] = Field(default_factory=dict)


class EvaluationResult(EvaluationModel):
    """Complete JSON-ready payload written to ``evaluation.json``."""

    question: str
    predicted_answer: str
    gold_answer: str
    normalized_prediction: str
    normalized_gold: str
    status: Literal["answered", "failed"]
    llm_acc: int = Field(ge=0, le=1)
    contain_acc: int = Field(ge=0, le=1)
    hard: int = Field(ge=0, le=1)
    soft: float = Field(ge=0.0, le=1.0)
    dataset: str = "hotpotqa"
    reported_metrics: list[EvaluationMetric] = Field(
        default_factory=lambda: [
            EvaluationMetric.LLM_ACC,
            EvaluationMetric.CONTAIN_ACC,
        ]
    )
    hard_metric: EvaluationMetric = EvaluationMetric.LLM_ACC
    soft_metric: EvaluationMetric = EvaluationMetric.CONTAIN_ACC
    judge_messages: list[JudgeMessage] = Field(default_factory=list)
    raw_judge_output: JsonValue = None
    judge_model: str | None = None
    judge_usage: JudgeUsage = Field(default_factory=JudgeUsage)
    judge_request_options: dict[str, JsonValue] = Field(default_factory=dict)


class EvaluationError(RuntimeError):
    """Base class for typed post-episode evaluation failures."""

    code = "evaluation_error"


class EvaluationInputError(EvaluationError):
    """An evaluation label or question is invalid."""

    code = "evaluation_input_error"


class JudgeResponseError(EvaluationError):
    """The provider completed without a valid structured verdict."""

    code = "judge_response_error"


@runtime_checkable
class JudgeClient(Protocol):
    """Provider-neutral semantic answer judge."""

    def judge(self, messages: Sequence[JudgeMessage]) -> JudgeResponse:
        """Judge one answer using an already constructed evaluation prompt."""


ScriptedVerdict = (
    bool
    | JudgeResponse
    | Exception
    | Callable[[Sequence[JudgeMessage]], bool | JudgeResponse]
)


class ScriptedJudge:
    """Deterministic sequence-backed judge for CI and adapter tests."""

    def __init__(
        self,
        verdicts: bool | JudgeResponse | Iterable[ScriptedVerdict],
        *,
        model: str = "scripted-judge",
    ) -> None:
        if isinstance(verdicts, (bool, JudgeResponse)):
            values: list[ScriptedVerdict] = [verdicts]
        else:
            values = list(verdicts)
        self._verdicts = deque(values)
        self.model = model
        self.calls: list[list[JudgeMessage]] = []

    def judge(self, messages: Sequence[JudgeMessage]) -> JudgeResponse:
        copied = [message.model_copy(deep=True) for message in messages]
        self.calls.append(copied)
        if not self._verdicts:
            raise JudgeResponseError("scripted judge has no verdicts remaining")
        scripted = self._verdicts.popleft()
        if isinstance(scripted, Exception):
            raise scripted
        if callable(scripted):
            scripted = scripted(copied)
        if isinstance(scripted, JudgeResponse):
            return scripted
        if not isinstance(scripted, bool):
            raise JudgeResponseError("scripted judge verdict must be boolean")
        return JudgeResponse(
            correct=scripted,
            raw_output={"correct": scripted},
            model=self.model,
            usage=JudgeUsage(calls=1),
        )


class FakeJudge(ScriptedJudge):
    """Convenience fake returning one deterministic verdict."""

    def __init__(self, correct: bool) -> None:
        super().__init__(correct, model="fake-judge")


class EpisodeEvaluator:
    """Evaluate a completed episode without exposing gold to the target agent."""

    def __init__(
        self,
        judge: JudgeClient,
        *,
        profile: str | DatasetProfile = "hotpotqa",
    ) -> None:
        self.judge = judge
        self.profile = (
            profile
            if isinstance(profile, DatasetProfile)
            else get_dataset_profile(profile)
        )

    def evaluate(
        self,
        *,
        question: str,
        predicted_answer: str | None,
        gold_answer: str,
    ) -> EvaluationResult:
        if not question.strip():
            raise EvaluationInputError("question must not be blank")
        if not gold_answer.strip():
            raise EvaluationInputError("gold_answer must not be blank")

        prediction = predicted_answer or ""
        normalized_prediction = normalize_answer(prediction)
        normalized_gold = normalize_answer(gold_answer)
        if not prediction.strip():
            return EvaluationResult(
                question=question,
                predicted_answer=prediction,
                gold_answer=gold_answer,
                normalized_prediction=normalized_prediction,
                normalized_gold=normalized_gold,
                status="failed",
                llm_acc=0,
                contain_acc=0,
                hard=0,
                soft=0.0,
                dataset=self.profile.key,
                reported_metrics=list(self.profile.reported_metrics),
                hard_metric=self.profile.hard_metric,
                soft_metric=self.profile.soft_metric,
            )

        messages = judge_messages(
            predicted_answer=prediction,
            gold_answer=gold_answer,
        )
        response = self.judge.judge(messages)
        llm_acc = int(response.correct)
        contain_acc = contain_accuracy(prediction, gold_answer)
        scores = {
            EvaluationMetric.LLM_ACC: llm_acc,
            EvaluationMetric.CONTAIN_ACC: contain_acc,
        }
        return EvaluationResult(
            question=question,
            predicted_answer=prediction,
            gold_answer=gold_answer,
            normalized_prediction=normalized_prediction,
            normalized_gold=normalized_gold,
            status="answered",
            llm_acc=llm_acc,
            contain_acc=contain_acc,
            hard=scores[self.profile.hard_metric],
            soft=float(scores[self.profile.soft_metric]),
            dataset=self.profile.key,
            reported_metrics=list(self.profile.reported_metrics),
            hard_metric=self.profile.hard_metric,
            soft_metric=self.profile.soft_metric,
            judge_messages=messages,
            raw_judge_output=response.raw_output,
            judge_model=response.model,
            judge_usage=response.usage,
            judge_request_options=response.request_options,
        )


def normalize_answer(value: object | None) -> str:
    """Apply the benchmark's article/punctuation normalization."""

    if value is None:
        return ""
    text = str(value).lower().translate(_ASCII_PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def contain_accuracy(
    predicted_answer: object | None,
    gold_answer: object | None,
) -> int:
    """Return one when normalized gold occurs in normalized prediction."""

    if not predicted_answer or not gold_answer:
        return 0
    prediction = normalize_answer(predicted_answer)
    gold = normalize_answer(gold_answer)
    return int(bool(prediction and gold and gold in prediction))


def judge_messages(
    *, predicted_answer: str, gold_answer: str
) -> list[JudgeMessage]:
    """Construct the semantic-correctness judge prompt."""

    return [
        JudgeMessage(role="system", content="You are an expert evaluator."),
        JudgeMessage(
            role="user",
            content=(
                "Please evaluate if the generated answer is correct by "
                "comparing it with the gold answer.\n\n"
                f"Generated answer: {predicted_answer}\n"
                f"Gold answer: {gold_answer}\n\n"
                "The generated answer should be considered correct if it:\n"
                "1. Contains the key information from the gold answer\n"
                "2. Is factually accurate and consistent with the gold answer\n"
                "3. Does not contain any contradicting information\n\n"
                "Return the structured correctness verdict."
            ),
        ),
    ]


__all__ = [
    "EpisodeEvaluator",
    "EvaluationError",
    "EvaluationInputError",
    "EvaluationResult",
    "FakeJudge",
    "JudgeClient",
    "JudgeMessage",
    "JudgeResponse",
    "JudgeResponseError",
    "JudgeUsage",
    "ScriptedJudge",
    "judge_messages",
    "contain_accuracy",
    "normalize_answer",
]

from agentic_rag.evaluation.interface_study import aggregate, evaluate_episode, score_answer

__all__ += ["aggregate", "evaluate_episode", "score_answer"]
