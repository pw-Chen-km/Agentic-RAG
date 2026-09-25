from dataclasses import replace

import pytest

from agentic_rag.evaluation.question_identity import prepare_question_rows, question_lookup
from agentic_rag.substrate.models import BenchmarkQuestion


def fixtures():
    questions = [BenchmarkQuestion(
        question_id=f"novel:benchmark_exact:q:{i:06d}", scope_id="novel:benchmark_exact:dev",
        source="novel", question=f"question {i}", answer=f"answer {i}",
        question_type="fact_retrieval", source_question_id=s, source_row_index=i,
    ) for i, s in enumerate(["duplicate", "unique", "duplicate"])]
    rows = [{"id": q.source_question_id, "question": q.question, "answer": q.answer,
             "question_type": "Fact Retrieval", "source": "original-document"} for q in questions]
    return rows, questions


def test_duplicate_rows_have_unique_stable_ids_and_correct_judge_answers():
    rows, questions = fixtures()
    result = prepare_question_rows(rows, questions, "novel")
    assert [r["id"] for r in result] == ["duplicate--row-000000", "unique", "duplicate--row-000002"]
    assert result == prepare_question_rows(rows, questions, "novel")
    assert rows[0]["id"] == rows[2]["id"] == "duplicate"
    lookup = question_lookup(questions)
    assert "duplicate" not in lookup
    assert [lookup[r["id"]].answer for r in result] == [r["answer"] for r in rows]
    assert len({f"{c}--{r['id']}" for r in result for c in ("C0", "C1", "C2", "C3", "C5", "C4", "A1")}) == 21
    assert result[0]["source"] == "original-document"
    assert result[2]["substrate_question_id"] == questions[2].question_id


@pytest.mark.parametrize("field", ["id", "question", "answer"])
def test_source_sidecar_mismatch_is_rejected(field):
    rows, questions = fixtures()
    rows[1][field] = "changed"
    with pytest.raises(ValueError, match="mismatch"):
        prepare_question_rows(rows, questions, "novel")


def test_missing_source_row_is_rejected():
    rows, questions = fixtures()
    with pytest.raises(ValueError, match="rows"):
        prepare_question_rows(rows, questions[:-1], "novel")


def test_suffix_collision_is_rejected():
    rows, questions = fixtures()
    questions[1] = replace(questions[1], source_question_id="duplicate--row-000000")
    rows[1]["id"] = questions[1].source_question_id
    with pytest.raises(ValueError, match="collision"):
        prepare_question_rows(rows, questions, "novel")


def test_hotpot_ids_remain_unchanged_and_duplicates_are_rejected():
    rows = [{"_id": "hotpot", "question": "q", "answer": "a"}]
    assert prepare_question_rows(rows, [], "hotpotqa") is rows
    with pytest.raises(ValueError, match="unique"):
        prepare_question_rows(rows * 2, [], "hotpotqa")
