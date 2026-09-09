"""Fresh assignments do not erase historical exposure or unseal transfer data."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rag.skillopt.multidataset_prepare import (
    PREPARATION_SPLIT_SCHEMA_VERSION,
    PREPARATION_VERSION,
    PreparationError,
    prepare_multidataset,
    split_rows,
    validate_prepared_split,
)
from test_skillopt_multidataset_prepare import (
    all_split_rows,
    dataset_fixture,
    fake_mapper,
    load,
    load_rows,
    write_json,
)


def _ids(output: Path, dataset: str) -> dict[str, list[str]]:
    return {split: [row["id"] for row in load_rows(output / dataset / f"{split}.jsonl")]
            for split in ("train", "validation", "test")}


def _unmapped_builder(*args):
    def for_question(row, index):
        return ({"kind": "paragraph" if row["source"] == "musique" else "sentence",
                 "method": "exact_source", "status": "unavailable", "reason": "unmapped",
                 "facts": [{"fact_id": "gold:0", "text": "Required evidence.",
                            "source": {}, "mappings": []}]}, [{"code": "unmapped_gold_reference"}])
    return SimpleNamespace(for_question=for_question, metadata={"fake": True})


def test_fresh_ignores_conflicting_roles_without_erasing_history(tmp_path):
    spec = dataset_fixture(tmp_path)
    history = tmp_path / "history.json"
    write_json(history, ["medical:benchmark_exact:q:000000"])
    clean_output, output = tmp_path / "clean", tmp_path / "fresh"
    prepare_multidataset({"history_policy": "fresh", "datasets": [spec]}, clean_output)
    spec["history"] = [{"path": str(history), "role": "train", "usage": "analyzed"},
                       {"path": str(history), "role": "test", "usage": "unknown"}]
    result = prepare_multidataset({"history_policy": "fresh", "datasets": [spec]}, output)
    assert result["all_training_sources_ready"] and result["all_inference_ready"]
    assert _ids(output, "medical") == _ids(clean_output, "medical")
    audit = load(output / "medical" / "history_report.json")
    assert audit["conflicts"] and not audit["historical_roles_enforced"]
    assert audit["fresh_split_is_not_unseen_data"] is True
    row = next(row for row in all_split_rows(output, "medical") if row["source_row_index"] == 0)
    assert row["history_usage"] == ["analyzed", "unknown"]
    assert row["historical_roles"] == ["test", "train"]
    assert row["historical_analysis_status"] == "analyzed"
    assert row["historical_exposure_status"] == "exposed"
    report = validate_prepared_split(output / "medical", spec["substrate"], True, for_training=True)
    assert report["history_policy"] == "fresh" and report["training_ready"]


def test_fresh_ignores_overfull_locks_and_keeps_exact_target_counts(tmp_path):
    spec = dataset_fixture(tmp_path)
    history = tmp_path / "all_train.json"
    write_json(history, [f"medical:benchmark_exact:q:{index:06d}" for index in range(20)])
    spec["history"] = [{"path": str(history), "role": "train", "usage": "executed"}]
    output = tmp_path / "fresh"
    prepare_multidataset({"history_policy": "fresh", "datasets": [spec]}, output)
    manifest = load(output / "medical" / "split_manifest.json")
    assert {role: len(ids) for role, ids in _ids(output, "medical").items()} == {"train": 4, "validation": 4, "test": 12}
    assert manifest["selection"]["historical_lock_count"] == 20
    assert manifest["selection"]["applied_historical_lock_count"] == 0
    assert manifest["selection"]["preserves_historical_roles"] is False
    assert all(row["historical_exposure_status"] == "exposed" for row in all_split_rows(output, "medical"))


def test_fresh_ambiguous_history_is_audited_and_possible_matches_are_unknown(tmp_path):
    spec = dataset_fixture(tmp_path, "novel", mutate=lambda rows: rows[1].update(id=rows[0]["id"]))
    history = tmp_path / "ambiguous.json"
    write_json(history, ["raw-0"])
    spec["history"] = [{"path": str(history), "role": "test", "usage": "analyzed"}]
    output = tmp_path / "fresh"
    report = prepare_multidataset({"history_policy": "fresh", "datasets": [spec]}, output)
    assert report["all_training_sources_ready"]
    audit = load(output / "novel" / "history_report.json")
    assert len(audit["unresolved_items"]) == 1
    rows = {row["source_row_index"]: row for row in all_split_rows(output, "novel")}
    for index in (0, 1):
        assert rows[index]["history_ambiguous_match"] is True
        assert rows[index]["historical_analysis_status"] == "unknown"
        assert rows[index]["historical_exposure_status"] == "unknown"
    assert rows[2]["historical_exposure_status"] == "no_record"
    assert validate_prepared_split(output / "novel", spec["substrate"], True)["valid"]


def test_fresh_unresolved_outside_pool_does_not_block_or_disappear(tmp_path):
    spec = dataset_fixture(tmp_path)
    history = tmp_path / "external.json"
    write_json(history, ["not-in-this-corpus"])
    spec["history"] = [{"path": str(history), "role": "validation", "usage": "unknown"}]
    output = tmp_path / "fresh"
    prepare_multidataset({"history_policy": "fresh", "datasets": [spec]}, output)
    assert load(output / "medical" / "history_report.json")["unresolved_items"][0]["id"] == "not-in-this-corpus"
    assert validate_prepared_split(output / "medical", spec["substrate"], True)["valid"]


def test_per_dataset_locked_override_still_blocks_conflict(tmp_path):
    medical, novel = dataset_fixture(tmp_path), dataset_fixture(tmp_path, "novel")
    history = tmp_path / "history.json"
    write_json(history, ["medical:benchmark_exact:q:000000"])
    medical.update(history_policy="locked", history=[{"path": str(history), "role": role, "usage": "executed"}
                                                    for role in ("train", "test")])
    output = tmp_path / "out"
    result = prepare_multidataset({"history_policy": "fresh", "datasets": [medical, novel]}, output)
    assert result["datasets"]["medical"]["split_status"] == "blocked"
    assert result["datasets"]["novel"]["history_policy"] == "fresh"
    assert result["datasets"]["novel"]["training_ready"]


def test_unknown_global_history_policy_rejected_before_writes(tmp_path):
    with pytest.raises(PreparationError, match="history policy"):
        prepare_multidataset({"history_policy": "silently-forget", "datasets": []}, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("dataset", ["musique", "2wikimultihop"])
def test_transfer_is_inference_ready_despite_unmapped_gt_and_unused_40_is_sealed(tmp_path, dataset):
    spec = {**dataset_fixture(tmp_path, dataset), "dataset_role": "transfer_only"}
    output = tmp_path / "fresh"
    result = prepare_multidataset({"history_policy": "fresh", "datasets": [spec]}, output,
                                 reference_builder=_unmapped_builder)
    readiness = result["datasets"][dataset]
    assert readiness["inference_ready"] is True
    assert readiness["reference_progress_ready"] is False
    assert readiness["training_ready"] is False
    assert readiness["all_four_arms_ready"] is False
    manifest = load(output / dataset / "split_manifest.json")
    assert manifest["dataset_role"] == "transfer_only"
    for role in ("train", "validation"):
        assert manifest["splits"][role]["usage"] == "sealed_unused"
        assert manifest["splits"][role]["sealed_unused"] is True
        assert manifest["splits"][role]["training_allowed"] is False
    assert manifest["splits"]["test"]["usage"] == "transfer_test"
    assert len(load_rows(output / dataset / "test.jsonl")) == 12
    assert sum(len(load_rows(output / dataset / f"{role}.jsonl")) for role in ("train", "validation")) == 8
    report = validate_prepared_split(output / dataset, spec["substrate"], False)
    assert report["inference_ready"] and not report["reference_progress_ready"]
    with pytest.raises(PreparationError, match="usable reference progress"):
        validate_prepared_split(output / dataset, spec["substrate"], True)
    with pytest.raises(PreparationError, match="sealed"):
        validate_prepared_split(output / dataset, spec["substrate"], False, for_training=True)


def test_transfer_remains_nontraining_even_with_complete_annotations(tmp_path):
    spec = {**dataset_fixture(tmp_path, "musique"), "dataset_role": "transfer_only"}
    output = tmp_path / "fresh"
    result = prepare_multidataset({"history_policy": "fresh", "datasets": [spec]}, output,
                                 reference_builder=fake_mapper)
    readiness = result["datasets"]["musique"]
    assert readiness["reference_progress_ready"] and not readiness["training_ready"]
    assert validate_prepared_split(output / "musique", spec["substrate"], True)["valid"]
    with pytest.raises(PreparationError, match="sealed"):
        validate_prepared_split(output / "musique", spec["substrate"], True, for_training=True)


def test_training_sources_readiness_does_not_depend_on_transfer_gold(tmp_path):
    medical = dataset_fixture(tmp_path)
    musique = {**dataset_fixture(tmp_path, "musique"), "dataset_role": "transfer_only"}
    result = prepare_multidataset({"history_policy": "fresh", "datasets": [medical, musique]}, tmp_path / "fresh",
                                 reference_builder=_unmapped_builder)
    assert result["all_training_sources_ready"] is True
    assert result["all_inference_ready"] is True
    assert result["all_four_arms_ready"] is False


def test_legacy_locked_manifest_remains_readable(tmp_path):
    spec = dataset_fixture(tmp_path)
    output = tmp_path / "out"
    prepare_multidataset({"datasets": [spec]}, output)
    manifest_path, readiness_path = output / "medical" / "split_manifest.json", output / "medical" / "readiness.json"
    manifest, readiness = load(manifest_path), load(readiness_path)
    manifest.update(schema_version="3.0", preparation_version="skillopt-multidataset-preparation-v1")
    for name in ("history_policy", "dataset_role", "history_report"):
        manifest.pop(name, None)
    readiness["version"] = "skillopt-multidataset-preparation-v1"
    for name in ("history_policy", "dataset_role", "inference_ready", "training_ready", "reference_progress_ready"):
        readiness.pop(name, None)
    write_json(manifest_path, manifest)
    write_json(readiness_path, readiness)
    result = validate_prepared_split(output / "medical", spec["substrate"], True)
    assert result["schema_version"] == "3.0" and result["history_policy"] == "locked"


def test_legacy_schema_cannot_be_used_to_bypass_fresh_policy_checks(tmp_path):
    spec = dataset_fixture(tmp_path)
    output = tmp_path / "out"
    prepare_multidataset({"history_policy": "fresh", "datasets": [spec]}, output)
    path = output / "medical" / "split_manifest.json"
    manifest = load(path)
    manifest.update(schema_version="3.0", preparation_version="skillopt-multidataset-preparation-v1")
    write_json(path, manifest)
    with pytest.raises(PreparationError, match="Legacy schema"):
        validate_prepared_split(output / "medical", spec["substrate"], False)


def test_validator_rejects_disagreeing_history_policy(tmp_path):
    spec = dataset_fixture(tmp_path)
    output = tmp_path / "out"
    prepare_multidataset({"history_policy": "fresh", "datasets": [spec]}, output)
    path = output / "medical" / "readiness.json"
    report = load(path)
    report["history_policy"] = "locked"
    write_json(path, report)
    with pytest.raises(PreparationError, match="differs across"):
        validate_prepared_split(output / "medical", spec["substrate"], False)


def test_validator_rejects_erased_history_after_fresh_preparation(tmp_path):
    spec = dataset_fixture(tmp_path)
    output = tmp_path / "out"
    prepare_multidataset({"history_policy": "fresh", "datasets": [spec]}, output)
    path = output / "medical" / "history_report.json"
    report = load(path)
    report["usage_counts"] = {"analyzed": 123}
    write_json(path, report)
    with pytest.raises(PreparationError, match="audit report changed"):
        validate_prepared_split(output / "medical", spec["substrate"], False)


def test_validator_rejects_unsealed_transfer_partition(tmp_path):
    spec = {**dataset_fixture(tmp_path, "musique"), "dataset_role": "transfer_only"}
    output = tmp_path / "out"
    prepare_multidataset({"history_policy": "fresh", "datasets": [spec]}, output, reference_builder=_unmapped_builder)
    path = output / "musique" / "split_manifest.json"
    manifest = load(path)
    manifest["splits"]["train"]["training_allowed"] = True
    write_json(path, manifest)
    with pytest.raises(PreparationError, match="sealed transfer"):
        validate_prepared_split(output / "musique", spec["substrate"], False)


@pytest.mark.parametrize("dataset,count,expected", [
    ("hotpotqa", 1000, (200, 200, 600)),
    ("medical", 2062, (412, 412, 1238)),
    ("novel", 2010, (402, 402, 1206)),
    ("musique", 1000, (200, 200, 600)),
    ("2wikimultihop", 991, (198, 198, 595)),
])
def test_full_pool_fresh_counts_and_duplicate_groups(dataset, count, expected):
    rows = [{"id": str(index), "question_type": f"type{index % 4}",
             "question_group_id": f"group:{index}", "answer": "x"} for index in range(count)]
    for index in range(0, 12, 2):
        rows[index + 1]["question_group_id"] = rows[index]["question_group_id"]
    locks = {row["question_group_id"]: "train" for row in rows}
    selected, report = split_rows(rows, dataset=dataset, seed=42, locks=locks, history_policy="fresh")
    assert tuple(len(selected[role]) for role in ("train", "validation", "test")) == expected
    assert report["applied_historical_lock_count"] == 0
    groups = [{row["question_group_id"] for row in selected[role]} for role in ("train", "validation", "test")]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    reversed_result = split_rows(list(reversed(rows)), dataset=dataset, seed=42, locks=locks, history_policy="fresh")
    assert (selected, report) == reversed_result


def test_fresh_2wiki_keeps_exactly_nine_coordinate_exclusions(tmp_path):
    def mutate(rows):
        for index in range(9):
            rows[index]["supporting_facts"] = [["Title", 99]]
    spec = {**dataset_fixture(tmp_path, "2wikimultihop", count=1000, mutate=mutate), "dataset_role": "transfer_only"}
    output = tmp_path / "fresh"
    result = prepare_multidataset({"history_policy": "fresh", "datasets": [spec]}, output,
                                 reference_builder=_unmapped_builder)
    readiness = result["datasets"]["2wikimultihop"]
    assert readiness["eligible_count"] == 991 and readiness["excluded_count"] == 9
    assert readiness["inference_ready"] and not readiness["reference_progress_ready"]
    assert {role: len(ids) for role, ids in _ids(output, "2wikimultihop").items()} == {"train": 198, "validation": 198, "test": 595}
    assert len(load_rows(output / "2wikimultihop" / "mapping_gaps.jsonl")) == 991
    assert validate_prepared_split(output / "2wikimultihop", spec["substrate"], False)["total_question_count"] == 991


def test_fresh_preparation_version_and_input_index_bytes_are_pinned(tmp_path):
    spec = dataset_fixture(tmp_path)
    substrate = Path(spec["substrate"])
    original = {str(path.relative_to(substrate)): path.read_bytes() for path in substrate.rglob("*") if path.is_file()}
    output = tmp_path / "fresh"
    prepare_multidataset({"history_policy": "fresh", "datasets": [spec]}, output)
    assert original == {str(path.relative_to(substrate)): path.read_bytes() for path in substrate.rglob("*") if path.is_file()}
    manifest = load(output / "medical" / "split_manifest.json")
    assert manifest["schema_version"] == PREPARATION_SPLIT_SCHEMA_VERSION == "3.1"
    assert manifest["preparation_version"] == PREPARATION_VERSION
    (substrate / "index.bin").write_bytes(b"changed")
    with pytest.raises(PreparationError):
        validate_prepared_split(output / "medical", substrate, False)
