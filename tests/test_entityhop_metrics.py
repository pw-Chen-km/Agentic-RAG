from __future__ import annotations

from arag_skillopt.entityhop_metrics import (
    aggregate_metrics,
    episode_metrics,
    markdown_report,
    normalized_call_key,
    paired_comparison,
)


def _snippet(chunk_id: str) -> str:
    return f"Chunk ID: {chunk_id}, Matched entity in chunk: ... Alice appears ..."


def _full(chunk_id: str, text: str = "Alice appears in the complete chunk.") -> str:
    return f"\n[Chunk {chunk_id}]\n{'-' * 80}\n{text}\n{'=' * 80}"


def _raw():
    return {
        "trajectory": [
            {"loop": 1, "tool_name": "keyword_search", "arguments": {"keywords": ["Alice"]}, "tool_result": "Chunk ID: 1, Matched keywords in chunk: ... Alice ...", "retrieved_tokens": 4},
            {"loop": 1, "tool_name": "entity_hop", "arguments": {"entity": "Alice", "top_k": 5}, "tool_result": _snippet("2"), "retrieved_tokens": 5, "status": "ok", "hop_backend": "direct_bm25", "chunks_found": 1},
            {"loop": 2, "tool_name": "read_chunk", "arguments": {"chunk_ids": ["2"]}, "tool_result": _full("2"), "retrieved_tokens": 8},
            {"loop": 3, "tool_name": "entity_hop", "arguments": {"entity": " alice ", "top_k": 5}, "tool_result": _snippet("2"), "retrieved_tokens": 2, "status": "ok", "hop_backend": "direct_bm25", "chunks_found": 1},
            {"loop": 4, "tool_name": "read_chunk", "arguments": {"chunk_ids": ["2", "3"]}, "tool_result": _full("2") + _full("3", "A new full chunk.")},
        ],
        "decision_audit": [{"decision": i} for i in range(1, 6)],
        "usage": {"complete": True, "input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        "elapsed_seconds": 1.5,
    }


def test_episode_metrics_track_visibility_and_hop_credit():
    metrics = episode_metrics(_raw(), {"correct": True})
    assert metrics["tool_calls"] == 5
    assert metrics["read_calls"] == 2
    assert metrics["unique_full_chunks_read"] == 2
    assert metrics["unique_chunks_seen"] == 3
    assert metrics["entity_hop_calls"] == 2
    assert metrics["hop_successes"] == 2
    assert metrics["repeated_hop_calls"] == 1
    assert metrics["hop_read_after_count"] == 2
    # One read call omits its tool token count, so the episode total is
    # unknown rather than an imputed partial sum.
    assert metrics["retrieved_tokens"] is None
    assert metrics["total_tokens"] == 120
    assert metrics["input_tokens"] == 100
    assert metrics["evidence_recall"] is None


def test_same_decision_read_is_not_credit_for_hop_and_unknown_usage_stays_null():
    raw = {
        "trajectory": [
            {"loop": 1, "tool_name": "entity_hop", "arguments": {"entity": "A"}, "tool_result": _snippet("1"), "status": "ok"},
            {"loop": 1, "tool_name": "read_chunk", "arguments": {"chunk_ids": ["1"]}, "tool_result": _full("1")},
        ],
        "usage": {"complete": False, "input_tokens": 99, "output_tokens": 20},
    }
    metrics = episode_metrics(raw)
    assert metrics["hop_read_after_count"] == 0
    assert metrics["hop_read_after_ratio"] == 0
    assert metrics["input_tokens"] is None
    assert metrics["output_tokens"] is None
    assert metrics["total_tokens"] is None


def test_error_and_no_match_are_preserved():
    raw = {
        "trajectory": [
            {"loop": 1, "tool_name": "entity_hop", "arguments": {"entity": "Missing"}, "tool_result": "No entity-linked chunks found for entity: Missing", "status": "no_entity_match"},
            {"loop": 2, "tool_name": "keyword_search", "arguments": {"keywords": ["x"]}, "tool_result": "Error executing tool: unavailable", "error": "unavailable"},
        ],
        "execution_errors": ["service unavailable"],
    }
    metrics = episode_metrics(raw)
    assert metrics["no_entity_match_count"] == 1
    assert metrics["service_error_count"] == 1
    assert metrics["tool_error_count"] == 1


def test_aggregation_and_paired_comparison_exclude_unknowns_without_imputation():
    first = {"question_id": "q1", "metrics": episode_metrics(_raw(), {"correct": True})}
    second_raw = _raw()
    second_raw["usage"] = {"complete": False}
    second = {"question_id": "q1", "metrics": episode_metrics(second_raw, {"correct": None})}
    summary = aggregate_metrics([first, second])
    assert summary["episodes"] == 2
    assert summary["known_evaluations"] == 1
    assert summary["accuracy"] == 1
    assert summary["input_tokens"]["missing_episodes"] == 1
    assert summary["input_tokens"]["complete_sum"] is None
    paired = paired_comparison([first], [second])
    assert paired["accuracy_pair_count"] == 0
    assert paired["costs"]["input_tokens"]["pair_count"] == 0


def test_call_key_and_report_are_deterministic():
    assert normalized_call_key("entity_hop", {"entity": " Alice ", "top_k": 5}) == normalized_call_key("entity_hop", {"top_k": 5, "entity": "alice"})
    report = markdown_report({"initial": [{"question_id": "q1", "metrics": episode_metrics(_raw(), {"correct": True})}]})
    assert "Evidence recall: N/A" in report
    assert "initial" in report

