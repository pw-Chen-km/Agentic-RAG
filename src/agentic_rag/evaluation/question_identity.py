"""Evaluation-only row identities shared by the runner and semantic backfill."""

from collections import Counter
from collections.abc import Sequence
from typing import Any

from agentic_rag.substrate.models import BenchmarkQuestion


IDENTITY_VERSION = "source-id-row-suffix-v1"


def question_lookup(questions: Sequence[BenchmarkQuestion]) -> dict[str, BenchmarkQuestion]:
    """Resolve canonical IDs and unambiguous episode IDs, never ambiguous aliases."""
    counts = Counter(q.source_question_id or q.question_id for q in questions)
    result: dict[str, BenchmarkQuestion] = {}
    for question in questions:
        source_id = question.source_question_id or question.question_id
        episode_id = source_id
        if counts[source_id] > 1:
            if question.source_row_index is None:
                raise ValueError(f"duplicate question ID lacks source row: {source_id}")
            episode_id = f"{source_id}--row-{question.source_row_index:06d}"
        for key in (question.question_id, episode_id):
            if key in result and result[key] != question:
                raise ValueError(f"question identity collision: {key}")
            result[key] = question
    return result


def prepare_question_rows(
    rows: list[dict[str, Any]],
    questions: Sequence[BenchmarkQuestion],
    dataset: str,
) -> list[dict[str, Any]]:
    """Validate source-to-sidecar alignment before creating any episode artifacts."""
    if dataset not in {"novel", "medical"}:
        ids = [str(row.get("_id") or row.get("id") or "") for row in rows]
        if any(not key for key in ids) or len(set(ids)) != len(ids):
            raise ValueError("question IDs must be nonempty and unique")
        return rows

    by_row = {q.source_row_index: q for q in questions}
    if len(by_row) != len(questions) or set(by_row) != set(range(len(rows))):
        raise ValueError("source question rows do not match substrate sidecar")
    counts = Counter(q.source_question_id for q in questions)
    normalized = []
    for index, row in enumerate(rows):
        question = by_row[index]
        source_id = str(row.get("_id") or row.get("id") or "")
        if (
            source_id != question.source_question_id
            or row.get("question") != question.question
            or row.get("answer") != question.answer
        ):
            raise ValueError(f"source/sidecar question mismatch at row {index}")
        episode_id = (
            f"{source_id}--row-{index:06d}" if counts[source_id] > 1 else source_id
        )
        value = dict(row)
        value.update(id=episode_id, source_question_id=source_id,
                     source_row_index=index, substrate_question_id=question.question_id)
        if "_id" in value:
            value["_id"] = episode_id
        normalized.append(value)
    lookup = question_lookup(questions)
    ids = [row["id"] for row in normalized]
    if len(set(ids)) != len(ids) or any(key not in lookup for key in ids):
        raise ValueError("question identity collision in episode schedule")
    return normalized
