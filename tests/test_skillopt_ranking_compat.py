from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agentic_rag.skillopt.ranking_compat import (
    build_safe_ranker,
    use_skillopt_ranking_compatibility,
)


def _fake_native(responses):
    calls = []
    pending = iter(responses)

    def chat(**kwargs):
        calls.append(kwargs)
        value = next(pending)
        if isinstance(value, Exception):
            raise value
        return value, {}

    native = SimpleNamespace(
        rank_and_select=lambda *args, **kwargs: {"native_mode": kwargs["update_mode"]},
        normalize_update_mode=lambda mode: mode,
        get_payload_items=lambda patch, mode: patch.get("edits", []),
        describe_item=lambda edit, mode: json.dumps(edit),
        format_meta_skill_context=lambda value: value,
        load_prompt=lambda name: "Existing ranking prompt allows selecting none.",
        chat_optimizer=chat,
        extract_json=json.loads,
    )
    return native, calls


def _patch(count=2):
    return {"reasoning": "shared reasoning", "edits": [
        {"op": "append", "content": f"rule {index}"} for index in range(count)
    ]}


@pytest.mark.parametrize("count", [1, 2, 4])
def test_explicit_empty_selection_is_not_replaced_by_first_edit(count):
    native, calls = _fake_native(['{"selected_indices": [], "reasoning": "not useful"}'])
    original = _patch(count)
    output = build_safe_ranker(native)("Skill", original, 1)
    assert output["edits"] == []
    assert output["ranking_details"]["status"] == "abstained"
    assert output["ranking_details"]["reasoning"] == "not useful"
    assert len(calls) == 1
    assert calls[0]["retries"] == 1
    assert "Select at most 1" in calls[0]["user"]
    assert "Select none" in calls[0]["user"]
    assert original == _patch(count)


def test_no_candidates_and_zero_budget_do_not_call_model():
    native, calls = _fake_native([])
    rank = build_safe_ranker(native)
    assert rank("Skill", _patch(0), 1)["ranking_details"]["status"] == "no_candidates"
    assert rank("Skill", _patch(), 0)["ranking_details"]["status"] == "no_budget"
    assert calls == []


@pytest.mark.parametrize("indices", [[0], [1, 0], [2]])
def test_valid_indices_preserve_model_order(indices):
    native, calls = _fake_native([json.dumps({"selected_indices": indices})])
    patch = _patch(3)
    output = build_safe_ranker(native)("Skill", patch, 2)
    assert output["edits"] == [patch["edits"][index] for index in indices]
    assert output["ranking_details"]["status"] == "selected"
    assert len(calls) == 1


@pytest.mark.parametrize("response", [
    "invalid JSON", "null", "[]", "{}", '{"selected_indices": null}',
    '{"selected_indices": "0"}', '{"selected_indices": [true]}',
    '{"selected_indices": [-1]}', '{"selected_indices": [2]}',
    '{"selected_indices": [0.0]}', '{"selected_indices": [0, 0]}',
    '{"selected_indices": [0, 1]}', '{"selected_indices": [0, 99]}',
])
def test_bad_response_retries_three_times_then_returns_no_edit(response):
    native, calls = _fake_native([response] * 3)
    output = build_safe_ranker(native)("Skill", _patch(), 1)
    assert len(calls) == 3
    assert all(call["retries"] == 1 for call in calls)
    assert output["edits"] == []
    assert output["ranking_details"]["status"] == "ranking_error"
    assert len(output["ranking_details"]["attempts"]) == 3
    assert all(attempt["error"] for attempt in output["ranking_details"]["attempts"])


def test_service_error_and_parse_error_can_recover_without_applying_a_fallback():
    native, calls = _fake_native([
        TimeoutError("offline test"), "invalid JSON", '{"selected_indices": [1]}',
    ])
    output = build_safe_ranker(native)("Skill", _patch(), 1)
    assert len(calls) == 3
    assert output["edits"] == _patch()["edits"][1:]
    assert [attempt["status"] for attempt in output["ranking_details"]["attempts"]] == [
        "error", "error", "valid",
    ]


def test_service_errors_never_choose_first_edit():
    native, calls = _fake_native([TimeoutError("offline test")] * 3)
    output = build_safe_ranker(native)("Skill", _patch(), 1)
    assert len(calls) == 3
    assert output["edits"] == []
    assert output["ranking_details"]["status"] == "ranking_error"


def test_non_patch_modes_keep_native_behavior():
    native, calls = _fake_native([])
    assert build_safe_ranker(native)("Skill", {}, 1, update_mode="rewrite_from_suggestions") == {
        "native_mode": "rewrite_from_suggestions",
    }
    assert calls == []


def test_scoped_patch_restores_both_references_even_after_error():
    pytest.importorskip("skillopt")
    import skillopt.engine.trainer as trainer
    import skillopt.optimizer.clip as clip

    original_clip = clip.rank_and_select
    original_trainer = trainer.rank_and_select
    with pytest.raises(RuntimeError, match="test interruption"):
        with use_skillopt_ranking_compatibility():
            assert clip.rank_and_select is trainer.rank_and_select
            assert clip.rank_and_select is not original_clip
            raise RuntimeError("test interruption")
    assert clip.rank_and_select is original_clip
    assert trainer.rank_and_select is original_trainer


@pytest.mark.parametrize("score, accepted", [(0.5, False), (0.8, False), (0.9, True)])
def test_selected_edit_still_needs_native_validation_gate(score, accepted):
    pytest.importorskip("skillopt")
    from skillopt.evaluation.gate import evaluate_gate
    from skillopt.optimizer.skill import apply_patch_with_report

    native, _ = _fake_native(['{"selected_indices": [0]}'])
    patch = build_safe_ranker(native)("Original Skill", _patch(1), 1)
    candidate, _ = apply_patch_with_report("Original Skill", patch)
    assert candidate != "Original Skill"
    gate = evaluate_gate(
        candidate_skill=candidate, cand_hard=score,
        current_skill="Original Skill", current_score=0.8,
        best_skill="Original Skill", best_score=0.8, best_step=0, global_step=1,
    )
    assert (gate.current_skill == candidate) is accepted


def test_integration_contract_records_hashes_and_rejects_old_or_changed_runs(tmp_path):
    from agentic_rag.skillopt.trainer import _ensure_integration_contract

    old = tmp_path / "old"
    old.mkdir()
    (old / "config.json").write_text("original")
    with pytest.raises(FileExistsError, match="new output directory"):
        _ensure_integration_contract(old)
    assert (old / "config.json").read_text() == "original"
    assert not (old / "skillopt_integration.json").exists()
    fresh = tmp_path / "fresh"
    _ensure_integration_contract(fresh)
    path = fresh / "skillopt_integration.json"
    original = path.read_text()
    metadata = json.loads(original)
    assert metadata["reflection_schema_version"].endswith("v3")
    assert len(metadata["prompt_sha256"]) == 6
    assert all(len(value) == 64 for value in metadata["code_sha256"].values())
    _ensure_integration_contract(fresh)
    assert path.read_text() == original
    metadata["reflection_schema_version"] = "old"
    path.write_text(json.dumps(metadata))
    with pytest.raises(FileExistsError):
        _ensure_integration_contract(fresh)
    assert json.loads(path.read_text())["reflection_schema_version"] == "old"
