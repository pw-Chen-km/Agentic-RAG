from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agentic_rag.skillopt.dataloader import AgenticRAGSkillOptDataLoader
from agentic_rag.skillopt.multidataset_prepare import (
    PREPARATION_SPLIT_SCHEMA_VERSION,
    PreparationError,
    build_reference_text,
    normalized_text,
    prepare_multidataset,
    split_rows,
    validate_prepared_split,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fake_mapper(dataset: str, substrate: Path, source: Path) -> SimpleNamespace:
    def for_question(row: dict, source_row: int) -> tuple[dict, list]:
        return ({"kind": "sentence" if dataset != "musique" else "paragraph",
                 "method": "source_provenance_v1", "status": "ready", "reason": None,
                 "facts": [{"fact_id": "fact:0000", "text": "Required evidence.",
                            "source": {"row": source_row}, "mappings": []}]}, [])
    return SimpleNamespace(for_question=for_question, metadata={"fake": True})


def dataset_fixture(tmp_path: Path, dataset: str = "medical", count: int = 20,
                    *, mutate=None, no_source_index=False) -> dict:
    root = tmp_path / dataset
    substrate = root / "substrate"
    evaluation = substrate / "evaluation"
    evaluation.mkdir(parents=True)
    task = {"medical": "fact_retrieval", "novel": "fact_retrieval",
            "hotpotqa": "bridge", "musique": "2_hop", "2wikimultihop": "compositional"}[dataset]
    source = [{"id": f"raw-{index}", "question": f"Question {index}?", "answer": f"Answer {index}",
               "question_type": task, "evidence": [f"Fact {index}.", f"Fact {index}."],
               "evidence_relations": [["KEEP_IN_ARCHIVE", index]],
               "context": [["Title", ["Required evidence."]]],
               "supporting_facts": [["Title", 0]]} for index in range(count)]
    if mutate:
        mutate(source)
    canonical = [{"question_id": f"{dataset}:benchmark_exact:q:{index:06d}",
                  "source_question_id": row["id"],
                  "source_row_index": None if no_source_index else index,
                  "question": row["question"], "answer": row["answer"],
                  "question_type": row["question_type"], "source": dataset,
                  "scope_id": f"{dataset}:benchmark_exact:dev"}
                 for index, row in enumerate(source)]
    questions = root / "questions.json"
    write_json(questions, source)
    pq.write_table(pa.Table.from_pylist(canonical), evaluation / "benchmark_questions.parquet")
    write_json(substrate / "manifest.json", {"dataset": dataset, "record_counts": {"chunks": 1},
               "source_artifacts": [{"role": "questions", "path": str(questions),
                                     "sha256": hashlib.sha256(questions.read_bytes()).hexdigest(),
                                     "size_bytes": questions.stat().st_size}]})
    (substrate / "index.bin").write_bytes(b"do-not-change-this-index")
    return {"dataset": dataset, "questions": str(questions), "substrate": str(substrate), "history": []}


def all_split_rows(output: Path, dataset: str) -> list[dict]:
    return [row for role in ("train", "validation", "test")
            for row in load_rows(output / dataset / f"{role}.jsonl")]


def test_medical_preparation_preserves_all_rows_and_runtime_index(tmp_path: Path) -> None:
    spec = dataset_fixture(tmp_path)
    output = tmp_path / "out"
    report = prepare_multidataset({"datasets": [spec]}, output)
    assert report["all_four_arms_ready"] is True
    manifest = load(output / "medical" / "split_manifest.json")
    assert manifest["schema_version"] == PREPARATION_SPLIT_SCHEMA_VERSION
    assert {role: entry["count"] for role, entry in manifest["splits"].items()} == {
        "train": 4, "validation": 4, "test": 12}
    assert len(all_split_rows(output, "medical")) == 20
    assert (Path(spec["substrate"]) / "index.bin").read_bytes() == b"do-not-change-this-index"
    for row in all_split_rows(output, "medical"):
        assert len(row["reference_evidence"]["facts"]) == 1
        assert "KEEP_IN_ARCHIVE" not in json.dumps(row)
        assert row["historical_analysis_status"] == "no_analysis_record"
        assert "historically_unexposed" not in row
    assert "KEEP_IN_ARCHIVE" in (output / "medical" / "hidden_reference_archive.jsonl").read_text()
    assert validate_prepared_split(output / "medical", spec["substrate"], True)["valid"]
    loader = AgenticRAGSkillOptDataLoader(output / "medical", dataset="medical")
    loader.setup({})
    assert len(loader.train_items) == 4 and len(loader.val_items) == 4 and len(loader.test_items) == 12


def test_reference_text_exact_normalized_dedup_retains_negation_and_indices() -> None:
    reference, issues = build_reference_text(
        {"evidence": [" Drug   works. ", "drug works.", "Drug does not work."]},
        question_id="q", source_record={"sha256": "a" * 64}, source_row_index=9)
    assert not issues
    assert [fact["text"] for fact in reference["facts"]] == [" Drug   works. ", "Drug does not work."]
    assert reference["facts"][0]["source"]["evidence_indices"] == [0, 1]
    assert reference["facts"][1]["source"]["source_row_index"] == 9


@pytest.mark.parametrize("evidence", [None, [], "not a list", [None], [3], [""], ["good", {"text": "bad"}]])
def test_malformed_reference_text_is_unavailable_not_silently_dropped(evidence: object) -> None:
    reference, issues = build_reference_text({"evidence": evidence}, question_id="q",
                                           source_record={"sha256": "a" * 64}, source_row_index=0)
    assert reference["status"] == "unavailable" and reference["facts"] == []
    assert issues


def test_malformed_medical_reference_retains_question_but_blocks_progress(tmp_path: Path) -> None:
    spec = dataset_fixture(tmp_path, mutate=lambda rows: rows[0].update(evidence=[None]))
    output = tmp_path / "out"
    report = prepare_multidataset({"datasets": [spec]}, output)
    assert report["datasets"]["medical"]["eligible_count"] == 20
    assert report["datasets"]["medical"]["split_status"] == "ready"
    assert report["all_four_arms_ready"] is False
    assert validate_prepared_split(output / "medical", spec["substrate"], False)["valid"]
    with pytest.raises(PreparationError, match="usable reference progress"):
        validate_prepared_split(output / "medical", spec["substrate"], True)


def test_novel_repeated_raw_id_does_not_join_wrong_evidence(tmp_path: Path) -> None:
    def mutate(rows: list[dict]) -> None:
        rows[1]["id"] = rows[0]["id"]
    spec = dataset_fixture(tmp_path, "novel", mutate=mutate)
    output = tmp_path / "out"
    prepare_multidataset({"datasets": [spec]}, output)
    rows = {row["source_row_index"]: row for row in all_split_rows(output, "novel")}
    assert rows[0]["source_question_id"] == rows[1]["source_question_id"]
    assert rows[0]["id"] != rows[1]["id"]
    assert rows[0]["reference_evidence"]["facts"][0]["text"] == "Fact 0."
    assert rows[1]["reference_evidence"]["facts"][0]["text"] == "Fact 1."


def test_duplicate_question_different_answers_stays_one_split(tmp_path: Path) -> None:
    def mutate(rows: list[dict]) -> None:
        rows[1]["question"] = "  QUESTION   0? "
    spec = dataset_fixture(tmp_path, mutate=mutate)
    output = tmp_path / "out"
    prepare_multidataset({"datasets": [spec]}, output)
    roles = {row["source_row_index"]: role for role in ("train", "validation", "test")
             for row in load_rows(output / "medical" / f"{role}.jsonl")}
    assert roles[0] == roles[1]
    duplicate = load(output / "medical" / "split_manifest.json")["selection"]["duplicate_groups"]
    assert duplicate[0]["answer_variant_count"] == 2
    assert sum(load(output / "medical" / "split_manifest.json")["selection"]["actual"].values()) == 20


def test_2wiki_bad_coordinate_excluded_unmapped_valid_question_retained(tmp_path: Path) -> None:
    def mutate(rows: list[dict]) -> None:
        rows[0]["supporting_facts"] = [["Title", 1]]
    spec = dataset_fixture(tmp_path, "2wikimultihop", mutate=mutate)

    def builder(*args) -> SimpleNamespace:
        def for_question(row: dict, index: int):
            return ({"kind": "sentence", "method": "source_provenance_v1", "status": "unavailable",
                     "reason": "unmapped", "facts": [{"fact_id": "sf:0", "text": "Required evidence.",
                                                      "source": {}, "mappings": []}]},
                    [{"code": "unmapped_gold_reference"}])
        return SimpleNamespace(for_question=for_question)

    output = tmp_path / "out"
    report = prepare_multidataset({"datasets": [spec]}, output, reference_builder=builder)
    item = report["datasets"]["2wikimultihop"]
    assert item["eligible_count"] == 19 and item["excluded_count"] == 1
    assert item["split_status"] == "ready" and item["all_four_arms_ready"] is False
    assert len(all_split_rows(output, "2wikimultihop")) == 19
    assert len(load_rows(output / "2wikimultihop" / "hidden_reference_archive.jsonl")) == 20
    assert load_rows(output / "2wikimultihop" / "exclusions.jsonl")[0]["reason"] == "invalid_gold_coordinate"


def test_hotpot_without_source_row_index_joins_verified_id_and_text(tmp_path: Path) -> None:
    spec = dataset_fixture(tmp_path, "hotpotqa", no_source_index=True)
    output = tmp_path / "out"
    report = prepare_multidataset({"datasets": [spec]}, output, reference_builder=fake_mapper)
    assert report["all_four_arms_ready"]
    assert {row["source_row_index"] for row in all_split_rows(output, "hotpotqa")} == set(range(20))


def test_canonical_source_row_mismatch_blocks_dataset(tmp_path: Path) -> None:
    spec = dataset_fixture(tmp_path)
    path = Path(spec["questions"])
    rows = load(path)
    rows[0]["question"] = "Changed source question"
    write_json(path, rows)
    report = prepare_multidataset({"datasets": [spec]}, tmp_path / "out")
    assert report["datasets"]["medical"]["error"]["code"] == "source_row_mismatch"


def test_history_roles_preserved_and_unknown_not_fresh(tmp_path: Path) -> None:
    spec = dataset_fixture(tmp_path)
    history = tmp_path / "history.json"
    write_json(history, [{"id": "old-canonical-id", "source_row_index": 0,
                          "question": "Question 0?", "answer": "Answer 0"}])
    spec["history"] = [{"path": str(history), "role": "train", "usage": "unknown"}]
    output = tmp_path / "out"
    prepare_multidataset({"datasets": [spec]}, output)
    rows = load_rows(output / "medical" / "train.jsonl")
    row = next(row for row in rows if row["source_row_index"] == 0)
    assert row["historical_analysis_status"] == "unknown" and row["history_usage"] == ["unknown"]


def test_conflicting_history_blocks_only_one_dataset_and_keeps_archive(tmp_path: Path) -> None:
    medical = dataset_fixture(tmp_path, mutate=lambda rows: rows[1].update(question=rows[0]["question"]))
    novel = dataset_fixture(tmp_path, "novel")
    history = tmp_path / "history.json"
    write_json(history, ["medical:benchmark_exact:q:000000"])
    medical["history"] = [{"path": str(history), "role": role, "usage": "prepared"}
                          for role in ("train", "test")]
    output = tmp_path / "out"
    report = prepare_multidataset({"datasets": [medical, novel]}, output)
    assert report["datasets"]["medical"]["split_status"] == "blocked"
    assert report["datasets"]["novel"]["split_status"] == "ready"
    assert not (output / "medical" / "train.jsonl").exists()
    assert len(load_rows(output / "medical" / "hidden_reference_archive.jsonl")) == 20
    assert load(output / "medical" / "history_report.json")["conflicts"]
    conflict = load(output / "medical" / "history_report.json")["conflicts"][0]
    assert conflict["questions"] == ["Question 0?"]
    assert len(conflict["ids"]) == 2 and len(conflict["events"]) == 2
    assert load(output / "medical" / "duplicate_groups.json")["groups"][0]["answer_variant_count"] == 2


def test_historical_count_over_target_retained_with_reported_deviation(tmp_path: Path) -> None:
    spec = dataset_fixture(tmp_path)
    history = tmp_path / "history.json"
    write_json(history, [f"medical:benchmark_exact:q:{index:06d}" for index in range(5)])
    spec["history"] = [{"path": str(history), "role": "train", "usage": "executed"}]
    output = tmp_path / "out"
    report = prepare_multidataset({"datasets": [spec]}, output)
    assert report["datasets"]["medical"]["split_status"] == "ready"
    rows = load_rows(output / "medical" / "train.jsonl")
    assert set(range(5)).issubset({row["source_row_index"] for row in rows})
    selection = load(output / "medical" / "split_manifest.json")["selection"]
    assert selection["quota_deviation"]["train"] >= 1
    assert selection["frozen_roles_over_nominal_quota"]["train"] == {"frozen_count": 5, "target": 4}
    assert all(row["historical_analysis_status"] == "no_analysis_record" for row in rows if row["source_row_index"] < 5)


def test_explicit_analysis_history_is_distinct_from_unknown_or_missing(tmp_path: Path) -> None:
    spec = dataset_fixture(tmp_path)
    history = tmp_path / "history.json"
    write_json(history, ["medical:benchmark_exact:q:000000"])
    spec["history"] = [{"path": str(history), "role": "test", "usage": "analyzed"}]
    output = tmp_path / "out"
    prepare_multidataset({"datasets": [spec]}, output)
    row = next(row for row in all_split_rows(output, "medical") if row["source_row_index"] == 0)
    assert row["historical_analysis_status"] == "analyzed"


def test_ambiguous_raw_id_history_does_not_select_first_novel_row(tmp_path: Path) -> None:
    spec = dataset_fixture(tmp_path, "novel", mutate=lambda rows: rows[1].update(id=rows[0]["id"]))
    path = tmp_path / "history.json"
    write_json(path, ["raw-0"])
    spec["history"] = [{"path": str(path), "role": "test", "usage": "analyzed"}]
    report = prepare_multidataset({"datasets": [spec]}, tmp_path / "out")
    assert report["datasets"]["novel"]["split_error"]["code"] == "history_conflict_or_unresolved"


def test_repeated_preparation_identical_files_and_no_overwrite(tmp_path: Path) -> None:
    spec = dataset_fixture(tmp_path)
    output = tmp_path / "out"
    first = prepare_multidataset({"datasets": [spec]}, output)
    files = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    second = prepare_multidataset({"datasets": [spec]}, output)
    assert first == second
    assert files == {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    with pytest.raises(PreparationError, match="Refusing to overwrite"):
        prepare_multidataset({"datasets": [spec], "seed": 43}, output)


@pytest.mark.parametrize("filename", ["index.bin", "manifest.json"])
def test_validator_rejects_changed_index_or_manifest(tmp_path: Path, filename: str) -> None:
    spec = dataset_fixture(tmp_path)
    output = tmp_path / "out"
    prepare_multidataset({"datasets": [spec]}, output)
    path = Path(spec["substrate"]) / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(PreparationError):
        validate_prepared_split(output / "medical", spec["substrate"], True)


def test_validator_rejects_changed_split(tmp_path: Path) -> None:
    spec = dataset_fixture(tmp_path)
    output = tmp_path / "out"
    prepare_multidataset({"datasets": [spec]}, output)
    path = output / "medical" / "train.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(PreparationError, match="Split file changed"):
        validate_prepared_split(output / "medical", spec["substrate"], False)


def test_normalization_keeps_number_negation_and_punctuation() -> None:
    assert normalized_text(" Ａ   B ") == "a b"
    assert normalized_text("Drug works.") != normalized_text("Drug does not work.")
    assert normalized_text("Year 2020?") != normalized_text("Year 2021?")
    assert normalized_text("AB?") != normalized_text("AB!")


def test_group_split_counts_floor_ratio_and_deterministic_order() -> None:
    rows = [{"id": str(index), "question_type": "a" if index % 2 else "b",
             "question_group_id": f"group:{index}", "answer": "x"} for index in range(21)]
    selected, report = split_rows(rows, dataset="medical")
    assert report["targets"] == {"train": 4, "validation": 4, "test": 13}
    assert report["actual"] == report["targets"]
    assert (selected, report) == split_rows(list(reversed(rows)), dataset="medical")
