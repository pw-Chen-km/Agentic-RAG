from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentic_rag.evaluation import (
    EpisodeEvaluator,
    EvaluationInputError,
    FakeJudge,
    JudgeConfigurationError,
    JudgeMessage,
    JudgeResponse,
    JudgeResponseError,
    JudgeTransportError,
    JudgeUsage,
    OpenAIResponsesJudge,
    ScriptedJudge,
    contain_accuracy,
    normalize_answer,
)


def test_arag_answer_normalization_is_deterministic() -> None:
    assert normalize_answer("  The U.S.A., an Answer!  ") == "usa answer"
    assert normalize_answer("A  Cat\tand THE dog") == "cat and dog"
    assert normalize_answer(None) == ""
    assert normalize_answer(42) == "42"


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        ("The answer is Warsaw, Poland.", "Warsaw", 1),
        ("A superhero role as the Marvel Comics hero.", "the Marvel Comics", 1),
        ("Paris", "Warsaw", 0),
        ("", "Warsaw", 0),
        ("Warsaw", "", 0),
        (None, "Warsaw", 0),
    ],
)
def test_contain_accuracy_matches_arag_semantics(
    prediction: object | None,
    gold: object | None,
    expected: int,
) -> None:
    assert contain_accuracy(prediction, gold) == expected


def test_episode_evaluator_maps_llm_and_contain_scores_for_skillopt() -> None:
    judge = FakeJudge(correct=True)
    result = EpisodeEvaluator(judge).evaluate(
        question="Where was Marie Curie born?",
        predicted_answer="Marie Curie was born in Warsaw, Poland.",
        gold_answer="Warsaw",
    )

    assert result.status == "answered"
    assert result.llm_acc == result.hard == 1
    assert result.contain_acc == 1
    assert result.soft == 1.0
    assert result.normalized_prediction == (
        "marie curie was born in warsaw poland"
    )
    assert result.normalized_gold == "warsaw"
    assert result.judge_model == "fake-judge"
    assert result.judge_usage.calls == 1
    assert len(judge.calls) == 1
    assert "Generated answer: Marie Curie" in judge.calls[0][1].content
    assert "Gold answer: Warsaw" in judge.calls[0][1].content

    # This is the complete payload the adapter may persist as evaluation.json.
    encoded = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    assert '"hard": 1' in encoded
    assert '"raw_judge_output": {"correct": true}' in encoded


def test_llm_and_contain_scores_remain_independent() -> None:
    result = EpisodeEvaluator(FakeJudge(correct=False)).evaluate(
        question="Where was Marie Curie born?",
        predicted_answer="Warsaw, but the answer is actually Paris.",
        gold_answer="Warsaw",
    )

    assert result.llm_acc == result.hard == 0
    assert result.contain_acc == 1
    assert result.soft == 1.0


def test_blank_prediction_scores_zero_without_calling_judge() -> None:
    judge = ScriptedJudge([])
    result = EpisodeEvaluator(judge).evaluate(
        question="Where was Marie Curie born?",
        predicted_answer=None,
        gold_answer="Warsaw",
    )

    assert result.status == "failed"
    assert result.hard == 0
    assert result.soft == 0.0
    assert result.judge_messages == []
    assert result.raw_judge_output is None
    assert result.judge_usage == JudgeUsage()
    assert judge.calls == []


def test_invalid_labels_are_typed_input_errors() -> None:
    evaluator = EpisodeEvaluator(FakeJudge(True))
    with pytest.raises(EvaluationInputError, match="question"):
        evaluator.evaluate(
            question=" ", predicted_answer="Warsaw", gold_answer="Warsaw"
        )
    with pytest.raises(EvaluationInputError, match="gold_answer"):
        evaluator.evaluate(
            question="Where?", predicted_answer="Warsaw", gold_answer=" "
        )


class _FakeResponses:
    def __init__(self, parsed: object, *, raw: str = '{"correct":true}') -> None:
        self.parsed = parsed
        self.raw = raw
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=self.parsed,
            output_text=self.raw,
            usage=SimpleNamespace(
                input_tokens=31,
                output_tokens=4,
                total_tokens=35,
                output_tokens_details=SimpleNamespace(reasoning_tokens=2),
            ),
        )


def test_openai_judge_uses_structured_responses_and_exact_usage() -> None:
    responses = _FakeResponses({"correct": True})
    judge = OpenAIResponsesJudge(
        client=SimpleNamespace(responses=responses),
        retry_backoff_seconds=0,
    )
    messages = EpisodeEvaluator(FakeJudge(True)).evaluate(
        question="Where?",
        predicted_answer="Warsaw",
        gold_answer="Warsaw",
    ).judge_messages

    result = judge.judge(messages)

    assert result == JudgeResponse(
        correct=True,
        raw_output='{"correct":true}',
        model="gpt-5.6-luna",
        usage=JudgeUsage(
            calls=1,
            input_tokens=31,
            output_tokens=4,
            reasoning_tokens=2,
            total_tokens=35,
        ),
        request_options={"temperature": 0.0, "store": False},
    )
    call = responses.calls[0]
    assert call["model"] == "gpt-5.6-luna"
    assert call["temperature"] == 0.0
    assert call["store"] is False
    assert call["text_format"].model_json_schema()["properties"] == {
        "correct": {"title": "Correct", "type": "boolean"}
    }
    assert call["input"] == [
        message.as_provider_input() for message in messages
    ]


def test_openai_judge_falls_back_when_temperature_is_unsupported() -> None:
    class UnsupportedTemperature(RuntimeError):
        status_code = 400
        body = {
            "error": {
                "message": "Unsupported parameter: temperature",
                "param": "temperature",
            }
        }

    class LunaResponses(_FakeResponses):
        def parse(self, **kwargs):
            self.calls.append(kwargs)
            if "temperature" in kwargs:
                raise UnsupportedTemperature()
            return SimpleNamespace(
                output_parsed={"correct": True},
                output_text='{"correct":true}',
                usage=None,
            )

    responses = LunaResponses({"correct": True})
    judge = OpenAIResponsesJudge(
        client=SimpleNamespace(responses=responses),
        max_retries=0,
    )
    result = judge.judge(
        [
            JudgeMessage(role="system", content="Evaluate."),
            JudgeMessage(role="user", content="Prediction and gold."),
        ]
    )

    assert result.correct is True
    assert len(responses.calls) == 2
    assert responses.calls[0]["temperature"] == 0.0
    assert "temperature" not in responses.calls[1]
    assert result.request_options == {
        "temperature": "provider_default",
        "store": False,
        "fallback_reason": "unsupported_temperature_parameter",
    }


class _StatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(str(status_code))
        self.status_code = status_code


class _RetryingResponses:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        if self.calls < 3:
            raise _StatusError(429)
        return SimpleNamespace(
            output_parsed={"correct": False},
            output_text='{"correct":false}',
            usage=None,
        )


def test_openai_judge_retries_transient_failures() -> None:
    responses = _RetryingResponses()
    judge = OpenAIResponsesJudge(
        client=SimpleNamespace(responses=responses),
        max_retries=2,
        retry_backoff_seconds=0,
    )
    result = judge.judge(
        [
            JudgeMessage(role="system", content="Evaluate."),
            JudgeMessage(role="user", content="Prediction and gold."),
        ]
    )

    assert result.correct is False
    assert result.usage.calls == 1
    assert responses.calls == 3


def test_openai_judge_transport_failure_is_not_scored_as_incorrect() -> None:
    class AlwaysUnavailable:
        def parse(self, **kwargs):
            raise ConnectionError("offline")

    judge = OpenAIResponsesJudge(
        client=SimpleNamespace(responses=AlwaysUnavailable()),
        max_retries=1,
        retry_backoff_seconds=0,
    )
    with pytest.raises(JudgeTransportError, match="2 attempt"):
        judge.judge(
            [
                JudgeMessage(role="system", content="Evaluate."),
                JudgeMessage(role="user", content="Prediction and gold."),
            ]
        )


def test_openai_judge_rejects_invalid_structured_boolean() -> None:
    judge = OpenAIResponsesJudge(
        client=SimpleNamespace(
            responses=_FakeResponses({"correct": "true"})
        ),
        retry_backoff_seconds=0,
    )
    with pytest.raises(JudgeResponseError, match="verdict validation"):
        judge.judge(
            [
                JudgeMessage(role="system", content="Evaluate."),
                JudgeMessage(role="user", content="Prediction and gold."),
            ]
        )


def test_openai_judge_requires_api_key_without_injected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(JudgeConfigurationError, match="OPENAI_API_KEY"):
        OpenAIResponsesJudge()
