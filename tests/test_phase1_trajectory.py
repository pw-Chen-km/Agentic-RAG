import copy
import json

from phase1_fixtures import chunks_example, full, raw_example, source_example

from arag_skillopt.observations import result_units
from arag_skillopt.runtime_audit import AuditClient, attach_audit
from arag_skillopt.support import SupportTracker, build_support_map, contains_sentence
from arag_skillopt.trajectory import build_view, render_view


def entry():
    return build_support_map(source_example(), chunks_example())["questions"]["q1"]


def test_decisions_multi_tool_novelty_read_recall_and_raw_unchanged():
    raw = raw_example()
    before = copy.deepcopy(raw)
    view, lineage = build_view(raw, {"correct": True, "status": "ok"}, entry())
    assert raw == before
    d1, d2, d3 = view["decisions"]
    assert len(d1["calls_selected_together"]) == 2
    assert d1["retrieval_statistics"]["returned_chunks"] == 2  # not 3 tool hits
    assert d1["retrieval_statistics"]["new_source_ratio"] == 1
    assert d2["retrieval_statistics"]["new_source_ratio"] == 0
    assert d2["retrieval_statistics"]["first_full_reads"] == 2
    assert d2["retrieval_statistics"]["old_chunks_with_new_text"] == ["17", "31"]
    assert (
        d1["supporting_fact_progress_evaluator_only"]["before"]["visible_recall"] == 0
    )
    assert (
        d1["supporting_fact_progress_evaluator_only"]["after_results_visible"][
            "visible_recall"
        ]
        == 0.5
    )
    assert (
        d2["supporting_fact_progress_evaluator_only"]["after_results_visible"][
            "read_recall"
        ]
        == 1
    )
    assert (
        d3["supporting_fact_progress_evaluator_only"]["before"]["visible_recall"] == 1
    )
    assert "York" not in json.dumps(d1)
    assert d1["results"][0]["results_visible_at_decision"] == 2
    assert "same_as" in d1["results"][1]["items"][0]["content"][0]
    assert view["episode_evaluation"]["tool_calls"] == 3
    assert view["episode_evaluation"]["model_usage"]["total_tokens"] == 500
    assert view["episode_evaluation"]["retrieved_text_tokens"] == 60
    assert "hash" not in render_view(view)
    assert "Observation texts" not in render_view(view)
    assert len(lineage["lineage"]) > 3


def test_dedup_is_lossless_by_resolving_ranges_against_raw():
    raw = raw_example()
    view, audit = build_view(raw, {"correct": True}, entry())
    bodies = {p["location"]: p["raw_body"] for p in audit["lineage"] if "location" in p}
    for d in view["decisions"]:
        for i, result in enumerate(d["results"], 1):
            for j, item in enumerate(result["items"], 1):
                if "content" not in item:
                    continue
                text = "".join(
                    s["text"]
                    if "text" in s
                    else bodies[s["same_as"]][slice(*s["source_range"])]
                    for s in item["content"]
                )
                assert text == bodies[f"Decision {d['decision']}, result {i}, item {j}"]


def test_never_count_undelivered_last_tool_results():
    raw = raw_example()
    raw["decision_audit"].pop()  # READ was returned but no later model input
    view, _ = build_view(raw, {"correct": None}, entry())
    assert (
        view["episode_evaluation"]["supporting_fact_recall_at_end"]["read_recall"] == 0
    )
    assert (
        view["decisions"][1]["supporting_fact_progress_evaluator_only"]["evaluable"]
        is False
    )
    assert view["decisions"][1]["results"][0]["results_visible_at_decision"] is None


def test_legacy_visibility_is_unknown_not_invented():
    raw = raw_example()
    raw.pop("decision_audit")
    view, _ = build_view(raw, {"correct": True}, entry())
    assert view["decisions"][0]["retrieval_statistics"]["new_chunks"] is None
    assert not view["decisions"][0]["supporting_fact_progress_evaluator_only"][
        "evaluable"
    ]


def test_context_removed_is_reported_not_claimed_still_visible():
    raw = raw_example()
    raw["decision_audit"][-1]["request"]["messages"] = [
        m
        for m in raw["decision_audit"][-1]["request"]["messages"]
        if m.get("tool_call_id") != "call1-0"
    ]
    view, _ = build_view(raw, {"correct": True}, entry())
    assert "removed" in view["decisions"][-1]["visible_before_this_decision"]


def test_mapping_raw_coordinates_recover_gold_text():
    from arag_skillopt.support import chunk_records, normalize

    chunks = {c["id"]: c["text"] for c in chunk_records(chunks_example())}
    for fact in entry()["facts"]:
        for match in fact["matches"]:
            extracted = chunks[match["chunk_id"]][
                match["raw_chunk_start"] : match["raw_chunk_end"]
            ]
            assert normalize(extracted) == normalize(fact["text"])


def test_combining_unicode_offsets_do_not_drop_entire_chunk():
    from arag_skillopt.support import normalize, raw_offsets

    text = "Earlier: 한. Alice cafe\u0301."
    positions = raw_offsets(text)
    assert positions is not None
    start = normalize(text).index("alicecafé.")
    end = start + len("alicecafé.")
    assert normalize(text[positions[0][start] : positions[1][end - 1]]) == "alicecafé."


def test_actual_upstream_arag_runtime_tools_and_audit(tmp_path):
    from arag.agent.base import BaseAgent
    from arag.tools.keyword_search import KeywordSearchTool
    from arag.tools.read_chunk import ReadChunkTool
    from arag.tools.registry import ToolRegistry

    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(chunks_example()))
    tools = ToolRegistry()
    tools.register(KeywordSearchTool(str(path)))
    tools.register(ReadChunkTool(str(path)))

    class ScriptedLLM:
        n = 0

        def chat(self, **kwargs):
            self.n += 1
            msg = {"role": "assistant", "content": None}
            if self.n == 1:
                msg["tool_calls"] = [
                    {
                        "id": "call1",
                        "function": {
                            "name": "keyword_search",
                            "arguments": json.dumps({"keywords": ["Alice"]}),
                        },
                    }
                ]
            elif self.n == 2:
                msg["tool_calls"] = [
                    {
                        "id": "call2",
                        "function": {
                            "name": "read_chunk",
                            "arguments": json.dumps({"chunk_ids": ["17", "31"]}),
                        },
                    }
                ]
            else:
                msg["content"] = "York"
            return {"message": msg, "input_tokens": 100, "output_tokens": 20, "cost": 0}

    audit = AuditClient(ScriptedLLM())
    raw = attach_audit(
        BaseAgent(audit, tools, max_loops=5).run(
            "Where was the author of The River born?"
        ),
        audit,
    )
    raw.update(question_id="q1", question="Where was the author of The River born?")
    view, _ = build_view(raw, {"correct": True}, entry())
    assert len(view["decisions"]) == 3
    assert (
        view["episode_evaluation"]["supporting_fact_recall_at_end"]["read_recall"] == 1
    )
    assert raw["usage"]["total_tokens"] == 360


def test_empty_vs_tool_error():
    for body, expected in [
        ("No results found for keywords: [x]", 0),
        ("Error executing tool: offline", None),
    ]:
        raw = raw_example()
        raw["trajectory"][0]["tool_result"] = body
        raw["trajectory"][1]["tool_result"] = body
        for audit in raw["decision_audit"][1:]:
            for message in audit["request"]["messages"]:
                if message.get("tool_call_id") in {"call1-0", "call1-1"}:
                    message["content"] = body
        view, _ = build_view(raw, {"correct": True}, entry())
        assert (
            view["decisions"][0]["retrieval_statistics"]["returned_chunks"] == expected
        )
        assert view["decisions"][0]["retrieval_statistics"]["new_source_ratio"] is None


def test_mapping_missing_ambiguous_and_original_denominator():
    mapping = build_support_map(source_example(), chunks_example()[:1])
    snapshot = SupportTracker(mapping["questions"]["q1"]).snapshot()
    assert not snapshot["evaluable"] and snapshot["gold_fact_count"] == 2
    assert snapshot["visible_recall"] is None
    source = source_example()
    source[0]["context"].append(["Other", ["The River was written by Alice."]])
    assert (
        build_support_map(source, chunks_example())["questions"]["q1"]["facts"][0][
            "status"
        ]
        == "ambiguous_source_text"
    )


def test_no_substring_word_or_omission_false_hits_and_terminal_punctuation():
    assert not contains_sentence("Yorkshire", "York.", True)
    assert contains_sentence("Alice was born in York", "Alice was born in York.", True)
    tracker = SupportTracker(entry())
    tracker.observe(
        [{"kind": "snippet", "chunk_id": "31", "text": "Alice was ... born in York."}]
    )
    assert tracker.snapshot()["visible_recall"] == 0


def test_repeated_read_and_nonexistent_chunk_are_not_new_full_text():
    units, _ = result_units(
        "read_chunk",
        "\n[Chunk 17]\n(This chunk has been read before)\n"
        + "=" * 80
        + "\n[Chunk 99] - Not found",
    )
    assert [u["kind"] for u in units] == ["already_read", "not_found"]
    assert not units[1]["available"]
    units, _ = result_units("read_chunk", full("17", "abc"))
    assert units[0]["text"] == "abc"


def test_audit_records_real_requests_not_mutable_aliases():
    class Client:
        def chat(self, **kwargs):
            return {
                "message": {"content": "ok"},
                "input_tokens": 11,
                "output_tokens": 7,
            }

    client = AuditClient(Client())
    messages = [{"role": "user", "content": "question"}]
    client.chat(messages=messages)
    messages[0]["content"] = "changed"
    raw = attach_audit({"trajectory": []}, client)
    assert raw["decision_audit"][0]["request"]["messages"][0]["content"] == "question"
    assert raw["usage"]["total_tokens"] == 18
