from __future__ import annotations

import hashlib
import json

import pytest

from agentic_rag.skillopt.offline_examples import write_offline_examples


def test_synthetic_examples_are_complete_deterministic_and_hash_verifiable(tmp_path):
    output = tmp_path / "examples"
    summary = write_offline_examples(output)
    assert summary["synthetic"]
    assert summary["case_count"] == 5
    assert summary["arm_count"] == 4
    assert summary["model_calls"] == summary["saved_real_trajectories_used"] == 0
    assert summary == json.loads((output / "summary.json").read_text())
    hashes_before = {}
    mtimes_before = {}
    for relative, expected in summary["artifact_sha256"].items():
        path = output / relative
        hashes_before[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        mtimes_before[relative] = path.stat().st_mtime_ns
        assert hashes_before[relative] == expected
    assert write_offline_examples(output) == summary
    assert {name: (output / name).stat().st_mtime_ns for name in mtimes_before} == mtimes_before
    report = (output / "example_report.md").read_text()
    assert report.count("（SYNTHETIC）") == 5
    assert "不是實驗成績" in report


def test_examples_have_expected_numbers_and_direct_only_bridge_attribution(tmp_path):
    cases = {case["case_id"]: case for case in write_offline_examples(tmp_path)["cases"]}
    lexical = cases["lexical_repeat"]["steps"]
    assert lexical[0]["visible_score"] == pytest.approx(3 / 7)
    assert lexical[1]["visible_delta"] == 0
    assert lexical[2]["visible_score"] == pytest.approx(13 / 14)
    preview = cases["preview_then_read"]["steps"]
    assert preview[0]["visible_score"] == 1
    assert preview[0]["eligible_score"] == 0
    assert preview[1]["visible_delta"] == 0
    assert preview[1]["eligible_delta"] == 1
    bridge = cases["expand_then_read"]["bridge_attribution"]
    assert bridge["reference_progress"]["visible_delta"] == 0
    assert bridge["later_reads_of_direct_children"][0]["reference_progress"]["visible_delta"] == 1
    assert cases["word_order_limitation"]["steps"][0]["visible_score"] == 1
    paragraph = cases["paragraph_exposure"]["steps"]
    assert paragraph[0]["visible_score"] == pytest.approx(0.1)
    assert paragraph[1]["visible_score"] == 0.5


def test_actual_four_arm_native_inputs_share_gt_without_fourth_retrieval_text(tmp_path):
    summary = write_offline_examples(tmp_path)
    for case in summary["cases"]:
        case_id = case["case_id"]
        canary = f"RETRIEVED_ONLY_CANARY_{case_id}"
        prefix = tmp_path / "cases" / case_id
        for arm in ("raw", "organized", "organized_support_labels", "progress_abstracted"):
            native = (prefix / arm / "optimizer_input.json").read_text()
            assert (canary in native) == (arm != "progress_abstracted")
        labels = json.loads((prefix / "organized_support_labels" / "reflection_input.json").read_text())
        abstract = json.loads((prefix / "progress_abstracted" / "reflection_input.json").read_text())
        assert labels["hidden_reference"] == abstract["hidden_reference"]
        assert "mappings" not in json.dumps(abstract["hidden_reference"])
        assert [step["reference_progress"] for step in labels["trajectory"]["step_diagnostics"]] == [
            step["reference_progress"] for step in abstract["trajectory"]["abstract_steps"]]


def test_nonidentical_existing_output_is_preserved_and_rejected_before_writes(tmp_path):
    output = tmp_path / "examples"
    output.mkdir()
    report = output / "example_report.md"
    report.write_text("Existing user report")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_offline_examples(output)
    assert report.read_text() == "Existing user report"
    assert not (output / "cases").exists()
    assert not (output / "summary.json").exists()
