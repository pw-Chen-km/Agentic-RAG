"""Explicit reuse of compatible observed JJ episodes, not old update decisions."""
import hashlib
import shutil
import threading
from pathlib import Path
from .checkpoint import load_checkpoint
from .skill_rules import skill_sha256
from .support import digest


def provenance(source, contract, initial):
    source=Path(source).resolve();old=load_checkpoint(source/'contract.json');prior=old['runtime'];cfg=contract['settings']
    for key in ('agent_prompt_hash','tool_schemas_hash','data_hashes','optimizer_prompts_hash'):
        if prior[key]!=contract[key]:raise ValueError('migration identity changed: '+key)
    for key in ('model','expected_digest','temperature','reasoning_effort','seed','max_tokens','context_length','input_token_limit'):
        if prior['settings']['llm'][key]!=cfg['llm'][key]:raise ValueError('migration Agent setting changed: '+key)
    for key in ('agent','keyword_search'):
        if prior['settings'][key]!=cfg[key]:raise ValueError('migration setting changed: '+key)
    for key in ('model','device','query_prompt_name'):
        if prior['settings']['embedding'].get(key)!=cfg['embedding'].get(key):raise ValueError('embedding changed: '+key)
    for key in ('model', 'expected_digest', 'think', 'num_ctx'):
        if prior['settings']['judge'].get(key)!=cfg['judge'].get(key):raise ValueError('judge setting changed: '+key)
    for key in ('enabled','backend','top_entities','top_k','max_top_k','entity_source'):
        if prior['settings']['entity_hop'].get(key)!=cfg['entity_hop'].get(key):raise ValueError('hop setting changed: '+key)
    if old['initial_skill_hash']!=skill_sha256(initial):raise ValueError('Initial Skill changed')
    for role in ('llm','optimizer','judge'):
        if set(prior['model_digests'][role].values()) != set(contract['model_digests'][role].values()):raise ValueError('model digest changed')
    allowed={'src/arag/core/ollama_guard.py','src/arag_skillopt/phase1_runtime.py'}
    for name,h in prior['source_hashes'].items():
        if name.startswith('src/arag/') and name not in allowed and contract['source_hashes'].get(name)!=h:
            raise ValueError('Agent code changed beyond isolated retrieval repair: '+name)
    return {'source_output':str(source),'source_contract_sha256':hashlib.sha256((source/'contract.json').read_bytes()).hexdigest(),
            'scope':'compatible Initial TRAIN raw/evaluation and completed batch1 Reflection only; Merge/gate/checkpoints recomputed',
            'source_train_episodes':len(list((source/'phase1/episodes/train').glob('*/*/raw.json'))),
            'native_isolation_only_changes_observed_data_transport':True}


def migrated_class(base, source):
    source=Path(source).resolve()/'phase1'
    class Migrated(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self._migration_lock=threading.Lock();self._reflections_imported=False

        def episode(self,question,skill,split,version):
            if split=='train' and skill_sha256(skill)==skill_sha256(self.initial):
                suffix=Path('episodes')/split/skill_sha256(skill)/digest({'id':str(question['id'])})[:20]
                origin,dest=source/suffix,self.root/suffix
                raw,evaluation=load_checkpoint(origin/'raw.json'),load_checkpoint(origin/'evaluation.json')
                if raw is not None and evaluation is not None and not (dest/'raw.json').exists():
                    if (raw['input_hash']!=digest({'id':str(question['id']),'question':question['question']})
                        or evaluation['input_hash']!=digest({'question':question,'raw':digest(raw['value'])})):
                        raise ValueError('migration episode input mismatch')
                    dest.mkdir(parents=True,exist_ok=True)
                    for name in ('raw.json','evaluation.json'):shutil.copy2(origin/name,dest/name)
                    from .checkpoint import save_checkpoint
                    save_checkpoint(dest/'migration-lineage.json',{'source':str(origin),
                        'files':{name:hashlib.sha256((origin/name).read_bytes()).hexdigest() for name in ('raw.json','evaluation.json')}})
            return super().episode(question,skill,split,version)

        def reflect_group(self,kind,skill,episodes,directory,name):
            if directory.parent.name=='001' and skill_sha256(skill)==skill_sha256(self.initial):
                with self._migration_lock:
                    if not self._reflections_imported:
                        directory.mkdir(parents=True,exist_ok=True)
                        for p in (source/'batches/001/reflections').glob('*.json'):
                            record=load_checkpoint(p);report=record.get('value',{})
                            for attempt in report.get('audit',[]):
                                raw=attempt.get('response',{}).get('raw_response',{});choices=raw.get('choices') or []
                                if choices and choices[0].get('finish_reason') in {'length','max_tokens'}:
                                    raise ValueError('do not reuse truncated Reflection')
                            if not (directory/p.name).exists():shutil.copy2(p,directory/p.name)
                        self._reflections_imported=True
            return super().reflect_group(kind,skill,episodes,directory,name)
    return Migrated
