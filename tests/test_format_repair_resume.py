"""Synthetic continuation: old updates/caches remain old, future calls repaired."""
import copy
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from test_phase1 import make_engine, questions, edits_call, rules
from test_diagnosis_v6 import state_action_call, ROOT
from arag_skillopt.checkpoint import load_checkpoint, save_checkpoint
from arag_skillopt.diagnostics import FORMAT_REPAIR_VERSION, STATE_ACTION_PROTOCOL
from arag_skillopt.format_repair_resume import continued_class, provenance, sha
from arag_skillopt.phase1 import Phase1, Phase1Config
from arag_skillopt.meta_stage import MetaStage, MetaConfig
from arag_skillopt.skill_rules import skill_sha256
from arag_skillopt.support import digest
from arag_skillopt.training_profile import NEW_PROFILE, REPAIRED_PROFILE, prompt_bundle


class Reply:
    def __init__(self, redundant=False):
        self.requests = []
        self.merges = 0
        self.redundant = redundant

    def chat(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        if kwargs["tools"][0]["function"]["name"] == "report_diagnoses":
            ids = re.findall(r"(?:Case|Question) ID: ([^\s]+)", kwargs["messages"][1]["content"])
            assert ids
            call = state_action_call(ids[:1])
            if self.redundant:
                call["function"]["arguments"]["diagnoses"][0]["case_id"] = ids[0]
        else:
            self.merges += 1
            ids = kwargs["tools"][0]["function"]["parameters"]["properties"]["supporting_case_ids"]["items"]["enum"]
            call = edits_call(ids=ids[:1])
            call["function"]["arguments"]["rule"]["title"] += str(self.merges)
        return {"message": {"tool_calls": [call]}, "raw_response": {"done_reason": "stop"}}


def preserved_run(tmp_path):
    source = tmp_path / "old/results/prior"
    root = source.parents[1]
    shutil.copytree(ROOT / "prompts/diagnosis_skill_v6", root / "prompts/diagnosis_skill_v6")
    engine = make_engine(source / "phase1", [])
    engine.cfg = Phase1Config(min_support_cases=1, analysis_protocol=STATE_ACTION_PROTOCOL, max_retries=1)
    engine.prompts = prompt_bundle(ROOT, {"optimizer_prompt_version": NEW_PROFILE}, workflow=True)
    engine.client = Reply()
    engine.replay_for_meta = True
    train, val = questions("t", 82), questions("v", 2)
    engine.run(train[:80], val)
    # A durable candidate and its evaluations exist, but the last batch was not sealed.
    (source / "phase1/batches/002/completed.json").unlink()
    (source / "phase1/batches/002/train_branches.json").unlink()
    settings = {"name": "old", "output": "old", "optimizer_prompt_version": NEW_PROFILE,
                "phase1": asdict(engine.cfg), "llm": {"seed": 42}, "agent": {"max_loops": 10}}
    runtime = {"settings": settings, "agent_prompt_hash": "same-agent", "tool_schemas_hash": "same-tools",
               "data_hashes": {"train": "fixed-train", "validation": "fixed-val"},
               "judge_prompt_hash": "same-judge", "model_digests": {"llm": {"endpoint": "same-model"}},
               "source_hashes": {str(p.relative_to(root)): sha(p) for p in (root / "prompts").rglob("*.md")}}
    old = {"runtime": runtime, "initial_skill_hash": skill_sha256(rules()),
           "train_hash": digest(train), "validation_hash": digest(val)}
    save_checkpoint(source / "contract.json", old)
    current = copy.deepcopy(runtime)
    current["settings"].update(name="new", output="new", optimizer_prompt_version=REPAIRED_PROFILE)
    current["settings"]["phase1"]["diagnostic_format_repair"] = FORMAT_REPAIR_VERSION
    return source, engine, train, val, current


def test_continuation_reuses_completed_and_pending_work_then_repairs_future_reports(tmp_path):
    source, old, train, val, runtime = preserved_run(tmp_path)
    record = provenance(source, runtime, rules())
    assert record["completed_batches"] == [1]
    assert record["pending_candidate_batches"] == [2]
    events = []
    adapters = make_engine(tmp_path / "adapters", events)
    client = Reply(redundant=True)
    client.merges = 2
    engine = continued_class(Phase1, source)(
        tmp_path / "new/phase1", rules(), client, adapters.rollout, adapters.evaluator,
        prompt_bundle(ROOT, {"optimizer_prompt_version": REPAIRED_PROFILE}, workflow=True),
        old.tool_guidance, Phase1Config(**runtime["settings"]["phase1"]), replay_for_meta=True)
    engine._format_continuation = record
    result = engine.run(train, val)
    assert result["completed_batches"] == 3
    assert (engine.root / "batches/001/completed.json").read_bytes() == (source / "phase1/batches/001/completed.json").read_bytes()
    assert (engine.root / "batches/002/merge.json").read_bytes() == (source / "phase1/batches/002/merge.json").read_bytes()
    assert all(e[1] in ("t80", "t81", "v00", "v01") for e in events if e[0] == "rollout")
    assert client.requests
    assert all("EXACT ARGUMENT FORMAT" in r["messages"][0]["content"] for r in client.requests if r["tools"][0]["function"]["name"] == "report_diagnoses")
    reflections = list((engine.root / "batches/003/reflections").glob("*.json"))
    assert any(load_checkpoint(p).get("value", {}).get("audit", [{}])[0].get("diagnostic_normalizations") for p in reflections)
    # The actual historical TRAIN pairs retain their original contract owner.
    receipt = load_checkpoint(engine.root / "batches/001/completed.json")["value"]
    meta = MetaStage(engine, tmp_path / "new/meta", load_checkpoint(engine.root / "final_skill.json"), MetaConfig())
    comparisons, audit = meta.comparisons(1, train[:40], receipt, tmp_path / "new/meta/batches/001")
    assert comparisons and engine.replay_contract_hash(1) == record["source_phase1_contract_hash"]
    assert engine.replay_contract_hash(3) == digest(engine.contract)
    # Exact resume uses the new contract and does not call either model adapter again.
    before = (len(events), len(client.requests))
    assert engine.run(train, val) == result
    assert (len(events), len(client.requests)) == before


@pytest.mark.parametrize("mutation", ["data", "agent", "gate", "model", "initial"])
def test_continuation_rejects_non_format_identity_changes(tmp_path, mutation):
    source, _, _, _, runtime = preserved_run(tmp_path)
    initial = rules()
    if mutation == "data": runtime["data_hashes"]["train"] = "wrong"
    elif mutation == "agent": runtime["settings"]["agent"]["max_loops"] = 11
    elif mutation == "gate": runtime["settings"]["phase1"]["max_token_cost_increase"] = .5
    elif mutation == "model": runtime["model_digests"]["llm"]["endpoint"] = "wrong"
    else: initial["retrieval_policy"][0]["title"] = "changed"
    with pytest.raises(ValueError): provenance(source, runtime, initial)
