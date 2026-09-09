from __future__ import annotations

import copy
import json

import pytest

from agentic_rag.skillopt.adapter import _native_reflection_conversation
from agentic_rag.skillopt.reflection_format import (
    VERSION, compact_reflection_input, expand_reflection_input,
)
from agentic_rag.skillopt.trajectory import REFLECTION_SCHEMA_VERSION
from test_skillopt_trajectory import _build, _steps_for_view


@pytest.mark.parametrize("arm", ["raw", "organized", "organized_support_labels", "progress_abstracted"])
def test_each_actual_view_is_reversible_and_does_not_mutate_audit(arm):
    rendered, manifest = _build(arm)
    before = copy.deepcopy(rendered)
    compact = compact_reflection_input(rendered)
    assert rendered == before
    assert expand_reflection_input(compact) == rendered
    assert manifest["optimizer_input_format"] == VERSION
    assert compact_reflection_input(rendered) == compact
    for step in _steps_for_view(compact):
        assert "available_action_space" not in step["interface_context"]
        assert "available_action_space" in step["decision_context"]
        assert "remaining_budget" not in step
        assert "remaining_budget" in step["decision_context"]
    assert VERSION in _native_reflection_conversation(rendered)[0]["content"]


def test_full_roundtrip_preserves_repeats_scores_paths_and_visibility_in_all_arms():
    views = [_build(arm)[0] for arm in ("raw", "organized", "organized_support_labels", "progress_abstracted")]
    restored = [expand_reflection_input(compact_reflection_input(view)) for view in views]
    contexts = [[s["decision_context"] for s in _steps_for_view(view)] for view in restored]
    assert all(context == contexts[0] for context in contexts)
    # Direct comparisons cover the original duplicate READ attempt and every
    # acquired unit, including its full list of acquisition paths.
    assert restored == views
    labeled = restored[2]["trajectory"]["step_diagnostics"]
    abstract = restored[3]["trajectory"]["abstract_steps"]
    for label, row in zip(labeled, abstract, strict=True):
        assert label["information_progress"] == row["information_progress"]
        assert label["supporting_fact_progress"] == row["supporting_fact_progress"]


def test_compaction_is_actually_in_native_messages_without_text_rewriting():
    rendered, _ = _build("raw")
    step = rendered["trajectory"]["raw_steps"][0]
    step["decision_context"]["missing_information"] = ["entity:bridge", "U1"]
    step["submitted_action"]["query"] = "entity:bridge"
    step["tool_raw"]["results"][0]["text"] = "entity:bridge"
    rendered["episode_outcome"]["final_answer"] = "entity:bridge"
    rendered["hidden_reference"] = {"answer": "entity:bridge", "text": "U1"}
    compact = compact_reflection_input(rendered)
    row = compact["trajectory"]["raw_steps"][0]
    assert row["decision_context"]["missing_information"] == ["entity:bridge", "U1"]
    assert row["submitted_action"]["query"] == "entity:bridge"
    results = row["tool_raw"]["results"]
    assert results["rows"][0][results["columns"].index("text")] == "entity:bridge"
    assert compact["episode_outcome"]["final_answer"] == "entity:bridge"
    assert compact["hidden_reference"] == rendered["hidden_reference"]
    messages = _native_reflection_conversation(rendered)
    payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
    assert payload == compact["trajectory"]
    assert "id_lookup" in messages[0]["content"]


def test_collision_mixed_columns_and_mismatched_duplicates_do_not_lose_information():
    rendered, _ = _build("organized")
    step = rendered["trajectory"]["action_ledger"][2]
    step["interface_context"]["available_action_space"] = {"different": True}
    step["remaining_budget"] = {"different": True}
    step["decision_context"]["visible_references"].append({"ref": "C1", "stable_id": "C1"})
    rendered["trajectory"]["retrieved_units"].append({"stable_id": "U1", "canonical_text": "U1"})
    encoded = compact_reflection_input(rendered)
    aliases = encoded["input_format"]["id_lookup"]
    assert len({row[0] for row in aliases}) == len(aliases)
    assert len({row[1] for row in aliases}) == len(aliases)
    assert expand_reflection_input(encoded) == rendered
    assert encoded["trajectory"]["action_ledger"][2]["remaining_budget"] == {"different": True}


def test_many_visible_units_save_space_without_removing_rows_or_changing_order():
    rendered, _ = _build("organized")
    refs = [{"ref": f"C{i}", "stable_id": "dataset:benchmark_exact:document:long-source-identifier:chunk:" + str(i),
             "node_type": "CHUNK", "has_been_read": False, "can_read": True, "can_use_as_evidence": False}
            for i in range(1, 201)]
    for step in rendered["trajectory"]["action_ledger"]:
        step["decision_context"]["visible_references"] = refs
    encoded = compact_reflection_input(rendered)
    assert len(json.dumps(encoded)) < len(json.dumps(rendered)) * .6
    assert expand_reflection_input(encoded) == rendered


def test_old_audit_can_be_read_without_silently_changing_its_prompt_format():
    rendered, _ = _build("organized")
    assert rendered["schema_version"] == REFLECTION_SCHEMA_VERSION
    rendered["schema_version"] = "agentic-rag-skillopt-reflection-v4"
    messages = _native_reflection_conversation(rendered)
    assert VERSION not in messages[0]["content"]
    assert json.loads(messages[-1]["content"].split("\n", 1)[1]) == rendered["trajectory"]


def test_mixed_record_tables_keep_absent_null_order_and_repeated_text():
    rendered, _ = _build("raw")
    rows = [{"node_type": "CHUNK", "text": "repeated text", "preview": None},
            {"node_type": "ENTITY", "name": "Bridge"}] * 20
    rendered["trajectory"]["raw_steps"][0]["tool_raw"]["results"] = rows
    compact = compact_reflection_input(rendered)
    table = compact["trajectory"]["raw_steps"][0]["tool_raw"]["results"]
    assert "column_sets" in table
    assert len(table["rows"]) == 40
    assert json.dumps(table).count("repeated text") == 20
    assert expand_reflection_input(compact) == rendered
