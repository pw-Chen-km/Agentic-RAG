from arag_skillopt.retrieval_diagnostics import build_retrieval_diagnostics


def _snippet(chunk_id="c1"):
    return f"Chunk ID: {chunk_id} (LinearHop score: 0.9)\nBridge: Alice -> s1 -> York\nMatched: Alice was born in York."


def _full(chunk_id="c1"):
    return f"[Chunk {chunk_id}]\n" + "-" * 80 + "\nAlice was born in York.\n" + "=" * 80


def _raw(*steps):
    return {
        "question_id": "q1",
        "decision_audit": [{"decision": 1, "response": {"message": {}}}],
        "trajectory": list(steps),
        "execution_errors": [],
    }


def test_hop_status_bridge_and_adoption_are_recorded():
    raw = _raw(
        {
            "loop": 1,
            "tool_name": "linear_hop",
            "arguments": {"entity": "Alice"},
            "tool_result": _snippet(),
            "status": "ok",
            "activated_entities": ["York"],
            "hop_trace": [
                {
                    "hop": 1,
                    "sentence_id": "s1",
                    "chunk_id": "c1",
                    "source_entity_ids": ["Alice"],
                    "activated_entity_ids": ["York"],
                    "similarity": 0.9,
                }
            ],
            "hop_backend": "linear_graph",
            "max_hops": 2,
            "top_k": 5,
        },
        {
            "loop": 2,
            "tool_name": "read_chunk",
            "arguments": {"chunk_ids": ["c1"]},
            "tool_result": _full(),
            "new_chunks_read": ["c1"],
        },
    )
    event = build_retrieval_diagnostics(raw)["events"][0]
    assert event["status"] == "ok"
    assert event["bridge"][0]["chunk_id"] == "c1"
    assert event["activated_entities"] == ["York"]
    assert event["later_full_reads"] == ["c1"]
    assert event["adoption"] is True
    assert event["adoption_status"] == "observed"


def test_hop_without_later_use_is_not_observed_when_trace_is_complete():
    raw = _raw(
        {
            "loop": 1,
            "tool_name": "entity_hop",
            "arguments": {"entity": "Alice"},
            "tool_result": "Chunk ID: c2, Matched entity in chunk: Alice",
            "status": "ok",
            "chunk_ids": ["c2"],
        }
    )
    event = build_retrieval_diagnostics(raw)["events"][0]
    assert event["adoption"] is False
    assert event["adoption_status"] == "not_observed"


def test_no_match_is_distinguished_from_unused_success():
    raw = _raw(
        {
            "loop": 1,
            "tool_name": "linear_hop",
            "arguments": {"entity": "Unknown"},
            "tool_result": "No graph entities found for entity: Unknown",
            "status": "no_entity_match",
            "chunk_ids": [],
        }
    )
    event = build_retrieval_diagnostics(raw)["events"][0]
    assert event["status"] == "no_entity_match"
    assert event["adoption"] is None
    assert event["adoption_status"] == "not_applicable"
    assert event["bridge"] == []
