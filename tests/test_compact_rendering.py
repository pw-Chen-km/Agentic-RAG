"""Compression must preserve evidence, temporal visibility and repeated actions."""

import copy

import pytest
from phase1_fixtures import chunks_example, raw_example, source_example
from test_phase1 import rules

from arag_skillopt.compact_rendering import DISPLAY_VERSION, CompactDisplay
from arag_skillopt.meta_comparison import (
    comparison_views,
    render_comparisons_with_audit,
)
from arag_skillopt.support import build_support_map
from arag_skillopt.trajectory import build_view, render_view_with_audit


def view():
    entry = build_support_map(source_example(), chunks_example())["questions"]["q1"]
    return build_view(raw_example(), {"correct": True, "status": "ok"}, entry)[0]


def recover(audit):
    """Use typed audit segments, never guess references inside untrusted prose."""
    bodies = {}
    for ref in audit["text_references"]:
        text = "".join(
            part["text"]
            if "text" in part
            else bodies[part["ref"]][slice(*part["source_range"])]
            for part in ref["display_segments"]
        )
        assert len(text) == ref["original_characters"]
        bodies[ref["ref"]] = text
    return {loc: bodies[ref] for loc, ref in audit["location_references"].items()}


def comparison(a, b):
    entry = build_support_map(source_example(), chunks_example())["questions"]["q1"]
    branches = [
        {"raw": raw, "evaluation": {"correct": True, "status": "ok"}} for raw in (a, b)
    ]
    return {
        "question_id": "q1",
        "question": a["question"],
        "rule_differences": [
            {
                "rule_id": "R01",
                "before": rules()["retrieval_policy"][0],
                "after": {"title": "Different"},
            }
        ],
        "first_difference": None,
        "views": comparison_views(*branches, entry),
    }


def test_all_result_bodies_roundtrip_and_view_unchanged():
    v = view()
    before = copy.deepcopy(v)
    text, audit = render_view_with_audit(v)
    expected = CompactDisplay()
    expected.prepare(v)
    assert recover(audit) == expected.bodies
    assert v == before
    assert audit["renderer_version"] == DISPLAY_VERSION
    assert "Repeated text:" not in text
    assert text.count("Model response:\nYork") == 1
    assert "final_answer" not in text
    assert "GT progress after delivery: seen 1->2/2; READ 0->2/2" in text
    assert "read_chunk; success -> D3" in text
    assert "input/output/total=400/100/500" in text


def test_repeated_result_and_calls_keep_decision_cost_and_later_delivery():
    v = view()
    duplicate = copy.deepcopy(v["decisions"][1])
    duplicate.update(
        decision=4, model_usage={"input_tokens": 9999, "output_tokens": 77}
    )
    duplicate["results"][0]["results_visible_at_decision"] = 5
    v["decisions"].insert(2, duplicate)
    text, audit = render_view_with_audit(v)
    assert "Decision 4" in text
    assert "Calls selected together: identical to D2" in text
    assert "read_chunk; success -> D5; identical to D2.R1" in text
    assert "Cost: model input=9999; output=77" in text
    assert audit["identical_results"][-1]["visible_at_decision"] == 5
    expected = CompactDisplay()
    expected.prepare(v)
    assert recover(audit) == expected.bodies


def test_tiny_overlap_is_literal_not_more_expensive_reference():
    display = CompactDisplay()
    tiny = "abcdefghijklmnopqrst"
    display.bodies = {"first": tiny, "second": tiny + " plus new content"}
    display.body(
        {"chunk_id": "17", "kind": "snippet", "content": [{"text": tiny}]}, "first", ""
    )
    rendered = display.body(
        {
            "chunk_id": "17",
            "kind": "full",
            "content": [
                {"same_as": "first", "source_range": [0, len(tiny)]},
                {"text": " plus new content"},
            ],
        },
        "second",
        "",
    )
    assert tiny + " plus new content" in rendered
    assert "@" not in rendered


def test_preview_to_full_is_not_collapsed_into_same_result():
    v = view()
    text, audit = render_view_with_audit(v)
    assert "Chunk 17 / snippet" in text and "Chunk 17 / full" in text
    assert "first full reads=2" in text
    assert "It was published in spring." in text
    assert recover(audit)["Decision 2, result 1, item 1"].endswith(
        "It was published in spring."
    )


def test_same_text_under_other_source_id_is_not_merged():
    v = view()
    v["decisions"] = v["decisions"][:1]
    item = copy.deepcopy(v["decisions"][0]["results"][0]["items"][0])
    item["chunk_id"] = "another-source"
    v["decisions"][0]["results"][0]["items"].append(item)
    text, _ = render_view_with_audit(v)
    assert text.count("The River was written by Alice.") == 2
    assert "Chunk another-source / snippet" in text


def test_error_empty_unknown_and_undelivered_are_not_zeroed():
    v = view()
    d = v["decisions"][0]
    d["results"][0].update(
        outcome="tool_error",
        items=[],
        raw_result="Error: backend unavailable",
        results_visible_at_decision=None,
    )
    d["retrieval_statistics"]["status"] = "unavailable_or_partial"
    d["supporting_fact_progress_evaluator_only"].update(
        evaluable=False, not_evaluable_reason="execution_unavailable"
    )
    # The later view references the changed result; isolate this error decision.
    v["decisions"] = [d]
    d["results"] = d["results"][:1]
    text, _ = render_view_with_audit(v)
    assert "tool_error -> NOT confirmed delivered" in text
    assert "Error: backend unavailable" in text
    assert "Sources: N/A (unavailable_or_partial)" in text
    assert "GT progress: N/A (execution_unavailable" in text


def test_crop_forced_answer_and_answer_mismatch_remain_visible():
    v = view()
    v["decisions"][-1].update(
        forced_answer=True,
        visible_before_this_decision="Some earlier tool results were removed or changed in this request: call1-0.",
    )
    v["episode_evaluation"]["final_answer"] = "A DIFFERENT stored answer"
    text, _ = render_view_with_audit(v)
    assert "removed or changed" in text and "Runtime forced an answer" in text
    assert "York" in text and "A DIFFERENT stored answer" in text
    assert "different from displayed responses" in text


def test_invalid_reference_does_not_silently_drop_text():
    v = view()
    v["decisions"][0]["results"][0]["items"][0]["content"] = [
        {"same_as": "missing", "source_range": [0, 10]}
    ]
    with pytest.raises(ValueError, match="unresolved"):
        render_view_with_audit(v)


def test_meta_common_prefix_once_but_costs_and_states_are_independent():
    a, b = raw_example(), raw_example()
    b["decision_audit"][1]["response"]["input_tokens"] = 4567
    c = comparison(a, b)
    before = copy.deepcopy(c)
    text, audit = render_comparisons_with_audit([c])
    assert c == before
    assert audit["cases"][0]["common_prefix_decisions"] == 3
    assert "A2 cost: model input=100; output=25" in text
    assert "B2 cost: model input=4567; output=25" in text
    assert text.count("Model response:\nYork") == 1
    assert text.count("correct=True") == 2
    expected = CompactDisplay()
    for label in ("A", "B"):
        expected.prepare(c["views"][label], f"Branch {label} / ")
    assert recover(audit["cases"][0]) == expected.bodies


def test_same_tool_but_different_query_is_not_a_common_decision():
    c = comparison(raw_example(), raw_example())
    c["views"]["B"]["decisions"][0]["calls_selected_together"][0]["arguments"][
        "keywords"
    ] = ["DIFFERENT TARGET"]
    text, audit = render_comparisons_with_audit([c])
    assert audit["cases"][0]["common_prefix_decisions"] == 0
    assert "DIFFERENT TARGET" in text
    assert "Branch A" in text and "Branch B" in text
    assert "A1)" in text and "B1)" in text
    expected = CompactDisplay()
    for label in ("A", "B"):
        expected.prepare(c["views"][label], f"Branch {label} / ")
    assert recover(audit["cases"][0]) == expected.bodies


def test_meta_result_order_changes_are_preserved():
    c = comparison(raw_example(), raw_example())
    display = CompactDisplay()
    for label in ("A", "B"):
        display.prepare(c["views"][label], f"Branch {label} / ")
    items = c["views"]["B"]["decisions"][0]["results"][0]["items"]
    # Replace the original bodies with literals before reversing their order.
    for ui, item in enumerate(items, 1):
        item["content"] = [
            {"text": display.bodies[f"Branch B / Decision 1, result 1, item {ui}"]}
        ]
    items.reverse()
    _, audit = render_comparisons_with_audit([c])
    assert audit["cases"][0]["common_prefix_decisions"] == 0


def test_common_prefix_cannot_hide_raw_metadata_difference():
    c = comparison(raw_example(), raw_example())
    c["first_difference"] = {"decision": 2}
    _, audit = render_comparisons_with_audit([c])
    assert audit["cases"][0]["common_prefix_decisions"] == 1


def test_renderer_version_changes_contract_and_blocks_old_resume(tmp_path):
    from test_phase1 import make_engine, questions

    from arag_skillopt.checkpoint import load_checkpoint, save_checkpoint

    engine = make_engine(tmp_path, [])
    assert engine.contract["optimizer_renderer_version"] == DISPLAY_VERSION
    engine.run(questions("t", 2), questions("v", 2))
    path = tmp_path / "contract.json"
    old = load_checkpoint(path)
    old.pop("optimizer_renderer_version")
    save_checkpoint(path, old)
    with pytest.raises(ValueError, match="contract changed"):
        make_engine(tmp_path, []).run(questions("t", 2), questions("v", 2))
