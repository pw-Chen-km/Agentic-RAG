from agentic_rag.skillopt.workflow.config import WorkflowConfig
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
