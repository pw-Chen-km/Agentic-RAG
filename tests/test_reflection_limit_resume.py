"""The budget switch preserves caches/lineage and applies to Phase1 AND Meta."""
import copy
from dataclasses import replace

import pytest

from arag_skillopt.checkpoint import load_checkpoint, save_checkpoint
from arag_skillopt.meta_stage import MetaStage, MetaConfig
from arag_skillopt.phase1 import Phase1, Phase1Config
from arag_skillopt.reflection_limit_resume import REVISION, provenance, continued_class
from arag_skillopt.support import digest
from test_format_repair_resume import preserved_run, Reply
from test_phase1 import make_engine, rules


def fixture(tmp_path):
    source, old, train, val, _ = preserved_run(tmp_path)
    previous = load_checkpoint(source/'contract.json')
    settings = previous['runtime']['settings']
    settings.setdefault('experiment', {})
    settings['phase1']['max_merge_input_tokens'] = 48000
    settings['optimizer'] = dict(transport='ollama_exact', input_token_limit=48000,
        context_length=65536, max_tokens=8192, reasoning_effort='medium')
    save_checkpoint(source/'contract.json', previous)
    runtime = copy.deepcopy(previous['runtime'])
    runtime['settings']['phase1']['max_input_tokens'] = 48000
    runtime['settings']['experiment']['reflection_budget_revision'] = copy.deepcopy(REVISION)
    return source, old, train, val, runtime


def test_paid_agent_cache_and_pending_candidate_are_reused(tmp_path):
    source, old, train, val, runtime = fixture(tmp_path)
    record = provenance(source, runtime, rules())
    assert record['validation_cache_reused'] and not record['historical_reflections_backfilled']
    assert any(p.startswith('episodes/validation/') for p in record['copied_files'])
    events=[];adapters=make_engine(tmp_path/'adapter',events);client=Reply();client.merges=2
    cfg=Phase1Config(**runtime['settings']['phase1'])
    engine=continued_class(Phase1,source)(tmp_path/'new/phase1',rules(),client,adapters.rollout,
        adapters.evaluator,old.prompts,old.tool_guidance,cfg,replay_for_meta=True)
    engine._reflection_limit_continuation=record
    result=engine.run(train,val)
    assert result['completed_batches']==3
    rollouts=[e[1] for e in events if e[0]=='rollout']
    assert not any(q in rollouts for q in [f't{i:02d}' for i in range(80)])
    # Only the two new batch3 TRAIN cases and its new candidate validation run.
    assert rollouts.count('v00')==1 and rollouts.count('v01')==1
    assert client.merges==3
    assert (engine.root/'batches/001/completed.json').read_bytes()==(source/'phase1/batches/001/completed.json').read_bytes()
    assert (engine.root/'batches/002/merge.json').read_bytes()==(source/'phase1/batches/002/merge.json').read_bytes()
    meta=MetaStage(engine,tmp_path/'new/meta',load_checkpoint(engine.root/'final_skill.json'),MetaConfig())
    assert meta.cfg is engine.cfg and meta.cfg.max_input_tokens==48000
    receipt=load_checkpoint(engine.root/'batches/001/completed.json')['value']
    comparisons,_=meta.comparisons(1,train[:40],receipt,tmp_path/'new/meta/batches/001')
    assert comparisons and engine.replay_contract_hash(1)==record['historical_replay_owners']['1']
    before=len(events),len(client.requests)
    assert engine.run(train,val)==result
    assert before==(len(events),len(client.requests))


@pytest.mark.parametrize('stage', ['success','meta'])
def test_longer_than16k_diagnosis_fits48k_without_shortening(tmp_path,stage):
    source,old,_,_,runtime=fixture(tmp_path)
    payload='Question ID: t00\n'+' detail'*20000
    low=make_engine(tmp_path/'low',[]);low.cfg=replace(old.cfg,max_merge_input_tokens=48000)
    low.prompts=old.prompts;low.client=Reply()
    skipped=low.propose(stage,rules(),payload,{'t00'},tmp_path/'low-report.json',1,
        diagnostic=True,decision_refs={'t00':['1']})
    assert skipped['status']=='input_too_long' and not low.client.requests
    high=make_engine(tmp_path/'high',[]);high.cfg=Phase1Config(**runtime['settings']['phase1'])
    high.prompts=old.prompts;high.client=Reply()
    target=MetaStage(high,tmp_path/'meta',rules(),MetaConfig()) if stage=='meta' else high
    accepted=target.propose(stage,rules(),payload,{'t00'},tmp_path/'high-report.json',1,
        diagnostic=True,decision_refs={'t00':['1']})
    assert accepted['status']=='ok' and len(high.client.requests)==1
    assert payload in high.client.requests[0]['messages'][1]['content']


@pytest.mark.parametrize('mutation',['agent','gate','optimizer','data','prompt','initial','merge','source','meta_started'])
def test_other_changes_cannot_use_budget_only_continuation(tmp_path,mutation):
    source,_,_,_,runtime=fixture(tmp_path);initial=rules()
    if mutation=='agent':runtime['settings']['llm']['seed']=43
    elif mutation=='gate':runtime['settings']['phase1']['max_token_cost_increase']=.2
    elif mutation=='optimizer':runtime['settings']['optimizer']['reasoning_effort']='none'
    elif mutation=='data':runtime['data_hashes']['train']='different'
    elif mutation=='prompt':runtime['optimizer_prompts_hash']='different'
    elif mutation=='initial':initial['retrieval_policy'][0]['title']='different'
    elif mutation=='merge':runtime['settings']['phase1']['max_merge_input_tokens']=50000
    elif mutation=='source':runtime['source_hashes'][next(iter(runtime['source_hashes']))]='different'
    else:(source/'meta').mkdir()
    with pytest.raises(ValueError):provenance(source,runtime,initial)
