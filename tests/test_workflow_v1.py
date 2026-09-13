from pathlib import Path

from agentic_rag.skillopt.workflow.config import WorkflowConfig
from agentic_rag.skillopt.workflow.demo import DemoBackend
from agentic_rag.skillopt.workflow.runner import WorkflowRunner
from agentic_rag.skillopt.workflow.skill_sections import SkillSections, validate_stage_patch
from agentic_rag.skillopt.workflow.candidate_replay import replay_purpose
from agentic_rag.skillopt.workflow.trajectory_views import batch_views
from agentic_rag.skillopt.workflow.validation import ValidationMetrics, evaluate_candidate
import pytest

def test_defaults_and_replay_are_train_only():
    c=WorkflowConfig(); assert c.rollout_batch_size==40 and c.reflection_minibatch_size==5
    assert replay_purpose('a','b',['q']).purpose=='meta_analysis_only'
    assert c.enable_meta is True
    assert WorkflowConfig.from_mapping({'enable_meta': False}).enable_meta is False

def test_stage_scope_rejects_fixed_change():
    s='''<!-- RETRIEVAL_POLICY_START -->a<!-- RETRIEVAL_POLICY_END -->\n<!-- RECOVERY_POLICY_START -->b<!-- RECOVERY_POLICY_END -->\n<!-- ANSWER_POLICY_START -->c<!-- ANSWER_POLICY_END -->'''
    with pytest.raises(ValueError): validate_stage_patch(s,s.replace('c','x'),'retrieval')
    validate_stage_patch(s,s.replace('c','x'),'answer')

def test_views_batch_without_mutating_rows():
    rows=[{'text':'secret','action':'READ'} for _ in range(6)]
    out=batch_views([rows], 'workflow', minibatch_size=5)
    assert len(out)==1 and len(out[0]['episodes'])==1 and rows[0]['text']=='secret'

def test_gate_requires_same_complete_nonzero_cost():
    a=ValidationMetrics(.80,1000,10,10); b=ValidationMetrics(.80,950,10,10)
    assert evaluate_candidate(a,b).accepted
    assert not evaluate_candidate(a,ValidationMetrics(.80,0,10,10)).accepted


def test_validation_gate_ablation_adopts_and_records_test(tmp_path):
    seed = (Path(__file__).parents[1] / "skills" / "workflow_seed.md").read_text()
    train = [{"id": "train-1", "question": "q", "answer": "a", "scope_id": "s", "split": "train"}]
    validation = [{"id": "val-1", "question": "v", "answer": "a", "scope_id": "s", "split": "validation"}]
    test = [{"id": "test-1", "question": "t", "answer": "a", "scope_id": "s", "split": "test"}]
    config = WorkflowConfig(use_validation_gate=False)
    runner = WorkflowRunner(backend=DemoBackend(), output=tmp_path, config=config,
        contract={"fake": True}, train=train, validation=validation, test=test, skill=seed)
    summary = runner.run()
    assert summary["validation_gate_enabled"] is False
    assert summary["test_executed"] is True
    assert summary["final_test_metrics"]["evaluated_questions"] == 1
    receipts = list((tmp_path / "retrieval").glob("batch_*/completed.json"))
    assert receipts
    receipt = __import__("json").loads(receipts[0].read_text())
    assert receipt["accepted"] is True
    assert receipt["reason"] == "validation_gate_disabled_candidate_adopted"
    assert receipt["test_metrics"]["candidate"]["evaluated_questions"] == 1
