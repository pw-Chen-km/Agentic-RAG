import copy
import json

import pytest

from arag.agent.base import BaseAgent
from arag.core.ollama_guard import GuardedOllamaClient, ContextOverflowError, OllamaRequestError
from arag.tools.registry import ToolRegistry
from arag_skillopt.runtime_audit import AuditClient, attach_audit
from arag_skillopt.endpoint_pool import PooledClient
from arag_skillopt.checkpoint import load_checkpoint, save_checkpoint
from arag_skillopt.context_repair_resume import provenance, continued_class
from arag_skillopt.training_profile import validate_ablation_runtime, CONTEXT_REVISION
from arag_skillopt.phase1 import Phase1
from arag_skillopt.meta_stage import MetaStage, MetaConfig
from arag_skillopt.support import digest
from test_phase1 import make_engine, rules
from test_format_repair_resume import preserved_run, Reply


def guard():
    return GuardedOllamaClient(base_url='http://127.0.0.1:11441/v1', api_key='x', model='qwen',
        context_length=65536, expected_digest='digest', input_token_limit=60928,
        safety_tokens=512, max_tokens=4096, reasoning_effort='none', require_complete_output=True)


@pytest.mark.parametrize('tokens,allowed', [(60927,True),(60928,True),(60929,False)])
def test_exact_capacity_boundary_no_retry_or_generation_over_limit(tokens,allowed):
    c=guard(); generated=[]
    c.measure=lambda p: {'input_tokens':tokens,'effective_input_limit':60928}
    c._generate=lambda p,n: generated.append(p) or {'ok':True}
    if allowed: assert c.chat([{'role':'user','content':'x'}])=={'ok':True}
    else:
        with pytest.raises(ContextOverflowError) as e: c.chat([{'role':'user','content':'x'}])
        assert e.value.details['generation_attempts']==[]
    assert len(generated)==int(allowed)
    payload=c.payload([{'role':'user','content':'x'}])
    assert payload['truncate'] is payload['shift'] is payload['think'] is False
    assert payload['options']=={'num_ctx':65536,'num_predict':4096,'temperature':0.0}


def test_budget_uses_qwen_including_tools_and_final_keeps_full_history():
    class Exact:
        enforces_context_window=True
        def inspect_context(self, **k): self.inspect=k; return {'input_tokens':28673}
        def chat(self,**k): self.final=k; return {'message':{'content':'answer'},'cost':0,'input_tokens':20,'output_tokens':3}
    c=Exact(); audit=AuditClient(c); tools=ToolRegistry()
    a=BaseAgent(audit,tools,system_prompt='system',max_token_budget=28672)
    out=a.run('question')
    assert out['token_budget_exceeded'] and out['answer']=='answer'
    assert c.inspect['tools']==[] and c.inspect['messages'][0]['content']=='system'
    assert c.final['tools'] is None and c.final['messages'][1]['content']=='question'
    assert len(c.final['messages'])==3


def test_hard_overflow_cannot_fall_back_to_unguarded_final():
    class Exact:
        enforces_context_window=True
        calls=0
        def inspect_context(self,**k):return {'input_tokens':65000}
        def chat(self,**k):
            self.calls+=1
            raise ContextOverflowError('input_too_long',{'stage':'context_preflight','generation_attempts':[]})
    c=Exact(); audit=AuditClient(c)
    raw=BaseAgent(audit,ToolRegistry(),max_token_budget=28672).run('q')
    result=attach_audit(raw,audit)
    assert c.calls==1 and result['error']=='input_too_long'
    assert result['execution_errors'] and not result['usage']['complete']
    assert result['decision_audit'][0]['error_details']['generation_attempts']==[]


@pytest.mark.parametrize('reason,measured', [('length',100),('stop',1),('stop',2000)])
def test_incomplete_output_and_bidirectional_token_mismatch_are_not_success(monkeypatch,reason,measured):
    c=guard()
    class Response:
        ok=True;status_code=200
        def iter_lines(self):
            yield json.dumps({'done':True,'done_reason':reason,'message':{'content':'partial'},
                              'prompt_eval_count':measured,'eval_count':4096}).encode()
        def close(self):pass
    monkeypatch.setattr('arag.core.ollama_guard.requests.post',lambda *a,**k:Response())
    expected=100 if reason=='length' else 1000
    with pytest.raises(OllamaRequestError) as e:
        c._generate(c.payload([{'role':'user','content':'q'}]),{'input_tokens':expected,'rendered_prompt_sha256':'x'})
    assert e.value.details['raw_response']['message']['content']=='partial'


def test_pool_preserves_exact_counter_capability():
    class Exact:
        enforces_context_window=True
        def inspect_context(self,**k):return {'input_tokens':123}
    c=PooledClient({'one':Exact(),'two':Exact()})
    assert c.enforces_context_window and c.inspect_context(messages=[])['input_tokens']==123


def context_fixture(tmp_path):
    source, old, train, val, _ = preserved_run(tmp_path)
    runtime=copy.deepcopy(load_checkpoint(source/'contract.json')['runtime'])
    runtime['settings']['llm'].update(transport='ollama_exact',context_length=65536,input_token_limit=60928,
        context_safety_tokens=512,max_tokens=4096,reasoning_effort='none',require_complete_output=True)
    runtime['settings'].setdefault('experiment', {})['runtime_context_revision']=dict(CONTEXT_REVISION)
    previous=load_checkpoint(source/'contract.json')
    previous['runtime']['settings'].setdefault('experiment', {})
    previous['runtime']['settings']['llm'].update(max_tokens=4096,reasoning_effort='none')
    save_checkpoint(source/'contract.json',previous)
    return source,old,train,val,runtime


def test_context_continuation_reuses_decisions_not_old_validation_or_pending_train(tmp_path):
    source, old, train, val, runtime=context_fixture(tmp_path)
    record=provenance(source,runtime,rules())
    assert not record['historical_validation_reused']
    assert not any(p.startswith('episodes/validation') for p in record['copied_files'])
    events=[]; adapters=make_engine(tmp_path/'adapter',events); opt=Reply();opt.merges=2
    engine=continued_class(Phase1,source)(tmp_path/'new/phase1',rules(),opt,adapters.rollout,adapters.evaluator,
        old.prompts,old.tool_guidance,old.cfg,replay_for_meta=True)
    engine._context_continuation=record
    summary=engine.run(train,val)
    assert summary['completed_batches']==3
    rollouts=[e[1] for e in events if isinstance(e,tuple) and e[0]=='rollout']
    assert not any(q in rollouts for q in [f't{i:02d}' for i in range(40)])
    assert all(rollouts.count(f't{i:02d}')==2 for i in range(40,80))
    assert 'v00' in rollouts and 'v01' in rollouts
    assert (engine.root/'batches/001/completed.json').read_bytes()==(source/'phase1/batches/001/completed.json').read_bytes()
    assert (engine.root/'batches/002/merge.json').read_bytes()==(source/'phase1/batches/002/merge.json').read_bytes()
    meta=MetaStage(engine,tmp_path/'new/meta',load_checkpoint(engine.root/'final_skill.json'),MetaConfig())
    receipt=load_checkpoint(engine.root/'batches/001/completed.json')['value']
    comparisons,_=meta.comparisons(1,train[:40],receipt,tmp_path/'new/meta/batches/001')
    assert comparisons and engine.replay_contract_hash(1)==record['historical_replay_owners']['1']
    excluded_id=comparisons[0]['question_id']
    engine._context_record['historical_meta_exclusions']={'1':{excluded_id:[{'reason':'confirmed_server_context_truncation'}]}}
    filtered,stats=meta.comparisons(1,train[:40],receipt,tmp_path/'filtered/meta')
    assert not any(c['question_id']==excluded_id for c in filtered)
    assert stats['skipped']['historical_context_integrity']==1
    engine._context_record['historical_meta_exclusions']={}
    assert engine.replay_contract_hash(2)==digest(engine.contract)
    before=len(events),len(opt.requests)
    assert engine.run(train,val)==summary
    assert before==(len(events),len(opt.requests))


@pytest.mark.parametrize('which', ['gate','model','question','initial','budget'])
def test_context_continuation_rejects_unrelated_changes(tmp_path,which):
    source,_,_,_,runtime=context_fixture(tmp_path); initial=rules()
    if which=='gate':runtime['settings']['phase1']['max_token_cost_increase']=0.99
    elif which=='model':runtime['model_digests']['llm']['endpoint']='different'
    elif which=='question':runtime['data_hashes']['train']='different'
    elif which=='initial':initial['retrieval_policy'][0]['title']='different'
    else:runtime['settings']['agent']['max_loops']=99
    with pytest.raises(ValueError):provenance(source,runtime,initial)


def test_ablation_revision_is_explicit_and_cannot_change_seed_or_budget():
    cfg={'llm':{'model':'qwen','expected_digest':'digest','seed':42,'transport':'ollama_exact',
        'context_length':65536,'input_token_limit':60928,'context_safety_tokens':512,
        'max_tokens':4096,'require_complete_output':True},'agent':{'token_budget':28672},
        'experiment':{'runtime_context_revision':dict(CONTEXT_REVISION)}}
    contract={'model':'qwen','model_expected_digest':'digest','seed':42,
        'context_length':32768,'agent_token_budget':28672}
    validate_ablation_runtime(cfg,contract)
    for field,value in [('seed',43),('context_length',131072),('max_tokens',1024)]:
        changed=copy.deepcopy(cfg);changed['llm'][field]=value
        with pytest.raises(ValueError):validate_ablation_runtime(changed,contract)
    del cfg['experiment']['runtime_context_revision']
    with pytest.raises(ValueError):validate_ablation_runtime(cfg,contract)
