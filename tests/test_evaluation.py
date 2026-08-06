from __future__ import annotations

from agentic_rag.evaluation import (
    EpisodeEvaluator,
    JudgeResponse,
    ScriptedJudge,
    contain_accuracy,
    normalize_answer,
)


def test_normalization_and_contain_accuracy() -> None:
    assert normalize_answer("The Warsaw!") == "warsaw"
    assert contain_accuracy("She was born in Warsaw, Poland.", "Warsaw") == 1
    assert contain_accuracy("Paris", "Warsaw") == 0


def test_gold_is_used_only_by_post_episode_evaluator() -> None:
    evaluator = EpisodeEvaluator(
        ScriptedJudge([JudgeResponse(correct=True, raw_output={"correct": True})])
    )
    result = evaluator.evaluate(
        question="Where was Marie Curie born?",
        predicted_answer="Warsaw",
        gold_answer="Warsaw",
    )
    assert result.llm_acc == 1
    assert result.contain_acc == 1
    assert result.hard == 1


def test_blank_answer_is_incorrect_without_judge_call() -> None:
    judge = ScriptedJudge([])
    result = EpisodeEvaluator(judge).evaluate(
        question="Question?", predicted_answer="", gold_answer="Answer"
    )
    assert result.status == "failed"
    assert result.llm_acc == 0
    assert judge.calls == []
