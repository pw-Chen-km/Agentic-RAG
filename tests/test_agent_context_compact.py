import json

from arag.agent.base import BaseAgent
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.tools.registry import ToolRegistry
from arag_skillopt.runtime_audit import AuditClient, attach_audit


def _tools(tmp_path):
    chunks = [
        {"id": "17", "text": "The River was written by Alice."},
        {"id": "31", "text": "Alice was born in York."},
    ]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(chunks), encoding="utf-8")
    registry = ToolRegistry()
    registry.register(KeywordSearchTool(str(path)))
    registry.register(ReadChunkTool(str(path)))
    return registry


def test_compact_context_keeps_latest_protocol_pair_and_deduplicates_state(tmp_path):
    class Scripted:
        def __init__(self):
            self.requests = []
            self.index = 0

        def chat(self, **kwargs):
            self.requests.append(kwargs)
            self.index += 1
            if self.index == 1:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call-search",
                        "function": {"name": "keyword_search", "arguments": json.dumps({"keywords": ["Alice"]})},
                    }],
                }
            elif self.index == 2:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call-read",
                        "function": {"name": "read_chunk", "arguments": json.dumps({"chunk_ids": ["17", "31"]})},
                    }],
                }
            else:
                message = {"role": "assistant", "content": "York"}
            return {"message": message, "input_tokens": 10, "output_tokens": 2, "cost": 0}

    client = Scripted()
    audit = AuditClient(client)
    result = BaseAgent(audit, _tools(tmp_path), max_loops=5).run(
        "Where was the author of The River born?"
    )
    assert result["answer"] == "York"
    assert len(client.requests) == 3

    first, second, third = client.requests
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "tool"]
    assert [m["role"] for m in third["messages"]] == ["system", "user", "assistant", "tool"]
    assert "[AGENT CONTEXT / arag-agent-context-compact-v1]" in second["messages"][1]["content"]
    assert "call-search" not in second["messages"][1]["content"]
    assert "call-search" not in third["messages"][1]["content"]
    assert third["messages"][2]["tool_calls"][0]["id"] == "call-read"
    assert all(message.get("tool_call_id") != "call-search" for message in third["messages"])

    # The full read result is kept only in the current protocol pair; older
    # search text is not replayed as a second full tool result.
    state = third["messages"][1]["content"]
    assert state.count("Chunk 17") == 0
    assert "Alice was born in York." not in state
    assert result["runtime_context"]["evidence_cards"]


def test_runtime_context_records_actions_without_retaining_full_prompts(tmp_path):
    class OneShot:
        def chat(self, **kwargs):
            return {
                "message": {"role": "assistant", "content": "done"},
                "input_tokens": 4,
                "output_tokens": 1,
                "cost": 0,
            }

    audit = AuditClient(OneShot())
    result = attach_audit(
        BaseAgent(audit, _tools(tmp_path), system_prompt="private system").run("q"),
        audit,
    )
    assert result["answer"] == "done"
    assert "runtime_context" in result
    assert "private system" not in json.dumps(result["runtime_context"])
