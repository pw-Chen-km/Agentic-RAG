import json

import pytest
from test_phase1 import diagnostic_call, edits_call, make_engine, questions, rules

from arag_skillopt.diagnostics import diagnostic_tools, parse_diagnoses, support_ids
from arag_skillopt.phase1 import Phase1Config


def report(call, ids):
    return parse_diagnoses(call, ids, ["R01", "R02"], {i: ["1"] for i in ids})


def test_singleton_diagnosis_is_legal_but_not_an_edit():
    parsed = report(diagnostic_call(["q1"]), ["q1"])
    assert len(parsed["diagnoses"]) == 1
    assert "edits" not in parsed
    assert [t["function"]["name"] for t in diagnostic_tools()] == ["report_diagnoses"]


@pytest.mark.parametrize(
    "mutation",
    ["unknown_case", "unknown_decision", "stale_rule", "too_many", "extra_field"],
)
def test_diagnosis_rejects_bad_lineage_or_schema(mutation):
    call = diagnostic_call(["q1"])
    args = call["function"]["arguments"]
    d = args["diagnoses"][0]
    if mutation == "unknown_case":
        d["observations"][0]["case_id"] = "validation-1"
    elif mutation == "unknown_decision":
        d["observations"][0]["decisions"] = ["9"]
    elif mutation == "stale_rule":
        d["target_rule_ids"] = ["R99"]
    elif mutation == "too_many":
        args["diagnoses"] *= 3
    else:
        d["rule"] = {}
    with pytest.raises(ValueError):
        report(call, ["q1"])


def test_reflection_cannot_return_formal_edits(tmp_path):
    engine = make_engine(tmp_path, [])

    class Wrong:
        def chat(self, **kwargs):
            return {"message": {"tool_calls": [edits_call(ids=["q1", "q2"])]}}

    engine.client = Wrong()
    result = engine.diagnose(
        "success",
        rules(),
        "saved cases",
        ["q1", "q2"],
        tmp_path / "diagnostic.json",
        {"q1": ["1"], "q2": ["1"]},
    )
    assert result["status"] == "diagnosis_failed_after_retry"
    assert result["edits"] == [] and result["diagnoses"] == []
    assert len(result["audit"]) == 3
    assert "not a Skill edit tool" in result["audit"][0]["validation_error"]


def test_split_singletons_can_form_cross_question_edit_at_merge(tmp_path):
    engine = make_engine(tmp_path, [])
    engine.cfg = Phase1Config(reflection_minibatch_size=1)
    summary = engine.run(questions("t", 4), questions("v", 2))
    assert summary["accepted_updates"] == 1
    reports = list((tmp_path / "batches/001/reflections").glob("success-*.json"))
    reports = [p for p in reports if ".attempt" not in p.name]
    assert len(reports) == 2
    for p in reports:
        value = json.loads(p.read_text())["value"]
        assert value["edits"] == [] and len(value["diagnoses"][0]["observations"]) == 1
    merged = json.loads((tmp_path / "batches/001/merge.json").read_text())["value"]
    assert len(merged["edits"]) == 3
    assert all(
        "report_diagnoses" == r["tools"][0]["function"]["name"]
        for r in engine.client.requests[:-1]
    )


def test_neutral_and_counterexamples_do_not_pad_support(tmp_path):
    a, b = (report(diagnostic_call([qid]), [qid]) for qid in ("q1", "q2"))
    b["diagnoses"][0]["observations"][0]["interpretation"] = "neutral"
    assert support_ids([a, b]) == {"q1"}
    engine = make_engine(tmp_path, [])
    proposals = [
        {
            "proposal_id": str(n),
            "analysis_kind": "success",
            "case_ids": [qid],
            "status": "ok",
            "edits": [],
            **r,
        }
        for n, (qid, r) in enumerate(zip(["q1", "q2"], [a, b]))
    ]
    result = engine.merge(rules(), proposals, tmp_path)
    assert result["status"] == "insufficient_cross_question_diagnoses"
    assert not engine.client.requests
    b["diagnoses"][0]["observations"][0]["interpretation"] = "counterexample"
    assert support_ids([a, b]) == {"q1"}


def test_merge_cannot_cite_neutral_case_even_if_two_other_supports_exist(tmp_path):
    report1 = report(diagnostic_call(["q1", "q2", "q3"]), ["q1", "q2", "q3"])
    report1["diagnoses"][0]["observations"][2]["interpretation"] = "neutral"
    engine = make_engine(tmp_path, [])

    class Wrong:
        def chat(self, **kwargs):
            return {"message": {"tool_calls": [edits_call(ids=["q1", "q3"])]}}

    engine.client = Wrong()
    result = engine.merge(
        rules(),
        [
            {
                "proposal_id": "a",
                "analysis_kind": "success",
                "case_ids": ["q1", "q2", "q3"],
                "status": "ok",
                **report1,
            }
        ],
        tmp_path,
    )
    assert result["status"] == "no_change_after_retry" and result["edits"] == []


def test_changed_protocol_does_not_resume_old_contract(tmp_path):
    engine = make_engine(tmp_path, [])
    engine.run(questions("t", 2), questions("v", 2))
    contract_path = tmp_path / "contract.json"
    old = json.loads(contract_path.read_text())
    old["phase"] = "phase1-only-v1"
    contract_path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="NEW output"):
        make_engine(tmp_path, []).run(questions("t", 2), questions("v", 2))


def test_empty_diagnoses_skip_merge_without_changing_skill(tmp_path):
    engine = make_engine(tmp_path, [])

    class Empty:
        def __init__(self):
            self.requests = []

        def chat(self, **kwargs):
            self.requests.append(kwargs)
            return {"message": {"tool_calls": [diagnostic_call([])]}}

    engine.client = Empty()
    summary = engine.run(questions("t", 4), questions("v", 2))
    assert summary["accepted_updates"] == 0
    assert len(engine.client.requests) == 2  # success and failure, no merge/gate
    assert not (tmp_path / "episodes/validation").exists()


def test_meta_singleton_diagnoses_can_merge_across_questions(tmp_path):
    from test_meta_training import engine as make_workflow

    from arag_skillopt.meta_stage import MetaConfig

    trainer = make_workflow(tmp_path, [], MetaConfig(reflection_minibatch_size=1))
    summary = trainer.run(questions("t", 4), questions("v", 2))
    assert summary["meta"]["reflection_count"] == 4
    assert summary["meta"]["accepted_updates"] == 1
    for p in (tmp_path / "meta/batches/001/reflections").glob("meta-*.json"):
        if ".attempt" not in p.name:
            report = json.loads(p.read_text())["value"]
            assert report["edits"] == []
            assert len(report["diagnoses"][0]["observations"]) == 1
