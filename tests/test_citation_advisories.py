"""Regression: model classification must not masquerade as missing provenance."""

import copy
import json

import pytest
from test_phase1 import diagnostic_call, edits_call, make_engine, questions, rules

from arag_skillopt.diagnostics import citation_advisories, support_ids
from arag_skillopt.optimizer_tools import tool_schemas


def proposal(ids, layer="policy", basis="observed_pattern"):
    report = diagnostic_call(ids)["function"]["arguments"]
    report["diagnoses"][0].update(failure_layer=layer, basis=basis)
    return {
        "proposal_id": "group-" + ids[0],
        "analysis_kind": "success",
        "case_ids": list(ids),
        "status": "ok",
        **report,
    }


class Reply:
    def __init__(self, ids):
        self.ids = ids
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        ids = self.ids[min(len(self.requests) - 1, len(self.ids) - 1)]
        return {"message": {"tool_calls": [edits_call(ids=ids)]}}


@pytest.mark.parametrize("layer", ["policy", "interface", "answer", "backend", "evaluation", "uncertain"])
@pytest.mark.parametrize("basis", ["observed_pattern", "policy_hypothesis", "uncertain"])
def test_layer_and_basis_never_remove_documented_support(tmp_path, layer, basis):
    reports = [proposal(["q1", "q2"], layer, basis)]
    assert support_ids(reports) == {"q1", "q2"}
    engine = make_engine(tmp_path, [])
    engine.client = Reply([["q1", "q2"]])
    result = engine.merge(rules(), reports, tmp_path)
    assert result["status"] == "ok" and len(result["edits"]) == 1
    expected = 0 if layer == "policy" and basis != "uncertain" else 2
    assert len(result["citation_advisories"]) == expected
    assert result["audit"][0]["citation_advisories"] == result["citation_advisories"]


def test_six_policy_plus_two_interface_cases_are_not_whole_proposal_rejected(tmp_path):
    ids = [f"q{i}" for i in range(8)]
    reports = [proposal(ids[:5]), proposal(ids[5:6]), proposal(ids[6:], "interface")]
    engine = make_engine(tmp_path, [])
    engine.client = Reply([ids])
    result = engine.merge(rules(), reports, tmp_path)
    assert result["status"] == "ok" and len(engine.client.requests) == 1
    assert result["edits"][0]["supporting_case_ids"] == ids
    assert {n["case_id"] for n in result["citation_advisories"]} == set(ids[6:])
    request = engine.client.requests[0]
    assert "Eligible supporting_case_ids" in request["messages"][1]["content"]
    schema = request["tools"][0]["function"]["parameters"]["properties"]["supporting_case_ids"]
    assert set(schema["items"]["enum"]) == set(ids)
    assert schema["minItems"] == 2 and schema["uniqueItems"]


@pytest.mark.parametrize(
    "bad_ids, expected",
    [
        (["q1"], "at least two DIFFERENT"),
        (["q1", "q1"], "received 1"),
        (["q1", "validation-1"], "outside the supplied train cases"),
        (["q1", "test-1"], "outside the supplied train cases"),
        (["q1", "q3"], "without a documented SUPPORT observation"),
    ],
)
def test_retry_names_actual_problem_and_then_accepts_explicit_correction(tmp_path, bad_ids, expected):
    reports = [proposal(["q1", "q2", "q3"])]
    reports[0]["diagnoses"][0]["observations"][2]["interpretation"] = "neutral"
    engine = make_engine(tmp_path, [])
    engine.client = Reply([bad_ids, ["q1", "q2"]])
    result = engine.merge(rules(), reports, tmp_path)
    assert result["status"] == "ok" and len(result["audit"]) == 2
    assert expected in result["audit"][0]["validation_error"]
    assert expected in engine.client.requests[1]["messages"][-1]["content"]
    assert "['q1', 'q2']" in engine.client.requests[1]["messages"][-1]["content"]
    assert result["edits"][0]["supporting_case_ids"] == ["q1", "q2"]


def test_non_support_cannot_pad_support_even_with_interface_label():
    reports = [proposal(["q1", "q2", "q3"], "interface", "uncertain")]
    obs = reports[0]["diagnoses"][0]["observations"]
    obs[1]["interpretation"] = "neutral"
    obs[2]["interpretation"] = "counterexample"
    assert support_ids(reports) == {"q1"}
    assert [a["case_id"] for a in citation_advisories(reports)] == ["q1"]


def test_real40_interface_diagnoses_still_go_through_validation_gate(tmp_path):
    events = []
    engine = make_engine(tmp_path, events, accept=False)
    base = engine.client

    class InterfaceDiagnoses:
        def chat(self, **kwargs):
            reply = base.chat(**kwargs)
            for c in reply["message"]["tool_calls"]:
                if c["function"]["name"] == "report_diagnoses":
                    for d in c["function"]["arguments"]["diagnoses"]:
                        d["failure_layer"] = "interface"
            return reply

    engine.client = InterfaceDiagnoses()
    summary = engine.run(questions("t", 40), questions("v", 2))
    assert summary["accepted_updates"] == 0  # advisory is not auto-adoption
    receipt = json.loads((tmp_path / "batches/001/completed.json").read_text())["value"]
    assert len(receipt["edits"]) == 3
    assert not receipt["gate"]["accepted"] and receipt["gate"]["reason"] != "no_change"
    assert len([e for e in events if isinstance(e, tuple) and e[0] == "rollout" and e[1].startswith("v")]) == 4
    merged = json.loads((tmp_path / "batches/001/merge.json").read_text())["value"]
    assert merged["citation_advisories"]


def test_native_edit_schema_is_optional_for_legacy_users_but_bounded_when_supplied():
    old = tool_schemas(["R01"])[0]["function"]["parameters"]["properties"]["supporting_case_ids"]
    new = tool_schemas(["R01"], ["q2", "q1", "q1"])[0]["function"]["parameters"]["properties"]["supporting_case_ids"]
    assert old["items"] == {"type": "string"}
    assert new["items"]["enum"] == ["q1", "q2"]


def test_meta_also_treats_interface_diagnosis_as_advisory_and_keeps_gate(tmp_path):
    from test_meta_training import engine as make_workflow

    events = []
    trainer = make_workflow(tmp_path, events)
    base = trainer.phase1.client

    class InterfaceDiagnoses:
        def chat(self, **kwargs):
            reply = base.chat(**kwargs)
            for c in reply["message"]["tool_calls"]:
                if c["function"]["name"] == "report_diagnoses":
                    for d in c["function"]["arguments"]["diagnoses"]:
                        d["failure_layer"] = "interface"
            return reply

    trainer.phase1.client = InterfaceDiagnoses()
    summary = trainer.run(questions("t", 40), questions("v", 4))
    assert summary["phase1"]["accepted_updates"] == 1
    assert summary["meta"]["comparison_count"] == 40
    assert summary["meta"]["accepted_updates"] == 1
    merged = json.loads((tmp_path / "meta/batches/001/merge.json").read_text())["value"]
    assert merged["status"] == "ok" and merged["citation_advisories"]
    assert (tmp_path / "phase1/batches/001/validation.json").exists()
