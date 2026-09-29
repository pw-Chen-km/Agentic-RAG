from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agentic_rag.skillopt.workflow.optimizer_native import (
    build_optimizer_tools,
    native_optimizer_call,
)


CATALOG = {
    "editable_sections": ["retrieval_policy", "recovery_policy"],
    "rules": [{"rule_id": "R03"}, {"rule_id": "R05"}],
}


def _tool(name, arguments):
    return SimpleNamespace(message=SimpleNamespace(tool_calls=[
        SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))
    ]))


def _rule():
    return {"title": "Recover after no progress", "when": "No new information is returned",
            "action_sequence": ["Use another legal path"],
            "stop_or_recovery": "Stop when evidence is sufficient", "exceptions": []}


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_tools_are_flat_and_catalog_ids_are_dynamic():
    tools = build_optimizer_tools("retrieval", CATALOG)
    assert [tool["function"]["name"] for tool in tools] == [
        "add_rule", "replace_rule", "delete_rule", "no_change"
    ]
    replace = next(t for t in tools if t["function"]["name"] == "replace_rule")
    assert replace["function"]["parameters"]["properties"]["rule_id"] == {
        "type": "string", "enum": ["R03", "R05"]
    }
    add = next(t for t in tools if t["function"]["name"] == "add_rule")
    assert "section" in add["function"]["parameters"]["required"]
    assert "rule_id" in add["function"]["parameters"]["required"]
    assert "rule" in add["function"]["parameters"]["required"]


def test_native_add_rule_is_converted_to_runner_edit_and_audited(tmp_path: Path):
    response = _tool("add_rule", {"section": "retrieval_policy", "rule_id": "R08",
                                    "rule": _rule(), "reason": "Repeated pattern",
                                    "supporting_case_ids": ["case-1"]})
    client = FakeClient([response])
    result = native_optimizer_call(client=client, model="qwen", messages=[],
                                   stage="retrieval", payload={"rule_catalog": CATALOG},
                                   output=tmp_path / "audit.json", think=True)
    assert result["no_change"] is False
    assert result["edits"][0]["operation"] == "add"
    assert result["edits"][0]["rule_id"] == "R08"
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["attempts"][0]["status"] == "valid"
    assert audit["attempts"][0]["tool_calls"][0]["name"] == "add_rule"
    assert "tools" in audit and client.calls[0]["tools"]
    assert "format" not in client.calls[0]


def test_two_edits_are_allowed_but_no_change_cannot_mix(tmp_path: Path):
    first = _tool("replace_rule", {"rule_id": "R03", "rule": _rule(),
                                    "reason": "A", "supporting_case_ids": []})
    second = _tool("delete_rule", {"rule_id": "R05", "reason": "B",
                                    "supporting_case_ids": []})
    # Fake one response with two calls to verify the two-edit limit.
    response = SimpleNamespace(message=SimpleNamespace(tool_calls=[
        first.message.tool_calls[0], second.message.tool_calls[0]
    ]))
    result = native_optimizer_call(client=FakeClient([response]), model="qwen", messages=[],
                                   stage="retrieval", payload={"rule_catalog": CATALOG},
                                   output=tmp_path / "audit.json")
    assert [edit["operation"] for edit in result["edits"]] == ["replace", "delete"]

    mixed = SimpleNamespace(message=SimpleNamespace(tool_calls=[
        first.message.tool_calls[0],
        SimpleNamespace(function=SimpleNamespace(name="no_change", arguments={"reason": "uncertain"})),
    ]))
    result = native_optimizer_call(client=FakeClient([mixed, mixed, mixed]), model="qwen", messages=[],
                                   stage="retrieval", payload={"rule_catalog": CATALOG},
                                   output=tmp_path / "mixed.json")
    assert result == {"edits": [], "no_change": True,
                      "reason": "native_optimizer_failed_after_retries"}
    audit = json.loads((tmp_path / "mixed.json").read_text())
    assert len(audit["attempts"]) == 3
    assert all(item["status"] == "error" for item in audit["attempts"])


def test_three_failures_return_no_change_without_selecting_candidate(tmp_path: Path):
    responses = [ValueError("offline"), ValueError("offline"), ValueError("offline")]
    client = FakeClient(responses)
    result = native_optimizer_call(client=client, model="qwen", messages=[],
                                   stage="retrieval", payload={"rule_catalog": CATALOG},
                                   output=tmp_path / "failed.json")
    assert result["no_change"] is True and result["edits"] == []
    assert len(client.calls) == 3
    audit = json.loads((tmp_path / "failed.json").read_text())
    assert audit["attempts"][-1]["error"] == "offline"
    assert audit["response"]["no_change"] is True


def test_add_existing_rule_id_is_rejected_and_retried(tmp_path: Path):
    response = _tool("add_rule", {"section": "retrieval_policy", "rule_id": "R03",
                                    "rule": _rule(), "reason": "duplicate",
                                    "supporting_case_ids": ["case-1"]})
    client = FakeClient([response, response, response])
    result = native_optimizer_call(client=client, model="qwen", messages=[],
                                   stage="retrieval", payload={"rule_catalog": CATALOG},
                                   output=tmp_path / "duplicate.json")
    assert result["no_change"] is True
    audit = json.loads((tmp_path / "duplicate.json").read_text())
    assert all("existing rule_id" in item["error"] for item in audit["attempts"])
