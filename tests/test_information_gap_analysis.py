"""Raw artifact diagnostics must work independently of runtime Assessment models."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def analysis_module():
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_information_gap_smoke.py"
    spec = importlib.util.spec_from_file_location("information_gap_analysis_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step(gaps, *, error=None, args=None, missing_usage=False):
    arguments = args if args is not None else {"assessment": {"missing_information": gaps}, "query": "question"}
    provider = {"raw_tool_calls": [{"function": {"name": "find_passages", "arguments": arguments}}]}
    if error:
        provider["failure_category"] = "state_invalid" if error == "duplicate_action" else "protocol_invalid"
    if not missing_usage:
        provider.update(provider_input_tokens=10, provider_output_tokens=5, provider_total_tokens=15)
    return {"provider_metadata": provider, "context_audit": {"assessment_requested": True},
            "decision": {"assessment": {"missing_information": gaps}, "action": {"type": "SEARCH", "target": "CHUNK"}},
            "validation_status": "invalid" if error else "valid", "assessment_status": "provided",
            "observation": {"status": "duplicate_action" if error == "duplicate_action" else "invalid_action" if error else "ok",
                            "error_code": error},
            "messages": [{"role": "user", "content": "NEW SOURCE TEXT\nEvidence\n\nVisible entity references:\nE1 — Marie Curie\nE2 — Warsaw\n\nEARLIER ACTION HISTORY\nNone."}]}


def write_episode(root, episode, *, dataset="novel"):
    owner = root / dataset
    owner.mkdir(parents=True, exist_ok=True)
    (owner / "run_manifest.json").write_text(json.dumps({"dataset": dataset, "require_evidence_assessment": True}), encoding="utf-8")
    path = owner / "episodes" / episode["episode_id"] / "episode.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(episode), encoding="utf-8")
    return path


def test_aggregates_gaps_duplicates_tokens_and_manual_review(tmp_path, analysis_module):
    episode = {"episode_id": "C3--q1", "query": "Where did the director study?", "termination_reason": "finished",
               "usage": {"policy_calls": 3}, "trajectory": [step(["need more information"]),
               step(["the director's university"], error="duplicate_action"),
               step(["the director's university"], error="duplicate_action")]}
    write_episode(tmp_path, episode)
    result = analysis_module.analyze(tmp_path)
    aggregate = result["dataset_condition_aggregates"][0]
    assert (aggregate["dataset"], aggregate["condition"]) == ("novel", "C3")
    assert aggregate["terminal_artifacts"] == 1
    assert aggregate["policy_calls"] == 3
    assert aggregate["attempted_actions"] == {"find_passages": 3}
    assert aggregate["executed_actions"] == {"find_passages": 1}
    assert aggregate["provider_tokens"] == {"input": 30, "output": 15, "total": 45}
    assert aggregate["failure_categories"] == {"state_invalid": 2, "duplicate_action": 2}
    assert aggregate["max_consecutive_duplicates"] == 2
    assert aggregate["gap_changes"] == 1
    assert aggregate["gap_comparisons"] == 2
    assert aggregate["generic_gap_heuristic_count"] == 1
    assert aggregate["mean_entity_cards"] == 2
    assert aggregate["rejected_turns_with_available_assessment"] == 2
    assert all(sample["status"] == "needs_review" and sample["semantic_alignment_score"] is None
               for sample in result["manual_action_gap_review"])


def test_raw_malformed_assessment_preserved_without_runtime_parser(tmp_path, analysis_module):
    invalid = step([], error="bad_arguments", args={"assessment": {"supported_facts": [],
                      "missing_information": ["", "more evidence", "a", "b"]}, "query": "question"})
    invalid["decision"] = None
    malformed = step([], args='{"assessment":{"missing_information":[],"missing_information":["x"]}}')
    malformed["decision"] = None
    missing = step([], args={"query": "question"})
    missing["decision"] = None
    write_episode(tmp_path, {"episode_id": "C0--legacy", "termination_reason": "error", "trajectory": [invalid, malformed, missing]})
    result = analysis_module.analyze(tmp_path)
    row = result["episodes"][0]
    assert row["raw_assessment_parse_failures"] == 3
    assert row["raw_assessment_statuses"] == {"invalid_assessment": 1, "invalid_json": 1, "missing_assessment": 1}
    assert row["turns"][0]["missing_information"] == ["", "more evidence", "a", "b"]
    assert row["turns"][0]["blank_gap_count"] == 1
    assert row["turns"][0]["assessment_schema_valid"] is False
    assert result["manual_action_gap_review"] == []


def test_unknown_provider_usage_stays_null_and_zero_remains_known(tmp_path, analysis_module):
    first = step(["specific fact"], missing_usage=True)
    second = step([])
    second["provider_metadata"].update(provider_input_tokens=0, provider_output_tokens=0, provider_total_tokens=0)
    write_episode(tmp_path, {"episode_id": "C1--q1", "termination_reason": "finished", "trajectory": [first, second]})
    result = analysis_module.analyze(tmp_path)
    assert result["episodes"][0]["provider_tokens"] == {"input": None, "output": None, "total": None}
    aggregate = result["dataset_condition_aggregates"][0]
    assert aggregate["token_availability"] == {"input": 0, "output": 0, "total": 0}
    assert result["episodes"][0]["turns"][1]["provider_tokens"]["total"] == 0


def test_assessment_off_missing_raw_call_and_broken_artifact_are_explicit(tmp_path, analysis_module):
    off = step([])
    off["context_audit"]["assessment_requested"] = False
    off["decision"]["assessment"] = None
    off["provider_metadata"]["raw_tool_calls"][0]["function"]["arguments"] = {"query": "question"}
    unavailable = step([])
    unavailable["provider_metadata"]["raw_tool_calls"] = []
    unavailable["decision"] = None
    write_episode(tmp_path, {"episode_id": "A1--q1", "trajectory": [off, unavailable]})
    broken = tmp_path / "novel/episodes/C0--broken/episode.json"
    broken.parent.mkdir(parents=True)
    broken.write_text("{broken", encoding="utf-8")
    result = analysis_module.analyze(tmp_path)
    assert result["status"] == "completed_with_read_errors"
    assert result["terminal_artifacts"] == 0
    assert result["episodes"][0]["raw_assessment_statuses"] == {"not_requested": 1}
    assert result["episodes"][0]["raw_assessment_parse_failures"] == 0
    assert len(result["artifact_read_errors"]) == 1


def test_cli_reads_artifacts_and_only_writes_selected_output(tmp_path, analysis_module):
    run = tmp_path / "run"
    artifact = write_episode(run, {"episode_id": "C2--q1", "termination_reason": "finished", "trajectory": [step([])]})
    original = artifact.read_bytes()
    output = tmp_path / "analysis/result.json"
    process = subprocess.run([sys.executable, analysis_module.__file__, "--run", str(run), "--output", str(output)],
                             check=True, capture_output=True, text=True)
    assert json.loads(process.stdout)["terminal_artifacts"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["analysis_version"] == analysis_module.VERSION
    assert artifact.read_bytes() == original
