import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rag.skillopt.reflection_budget import (
    budget_settings, build_budgeted_dispatcher, capture_native_prompt,
    partition_group, prompt_messages, sha,
)


def native_stub():
    native = SimpleNamespace()
    native.generations = []

    def chat(**kwargs):
        native.generations.append(kwargs)
        return '{"patch":{"edits":[]}}', {"prompt_tokens": 20}

    def analyst(skill, items, prediction_dir, edit_budget=1, **kwargs):
        content = [json.loads((Path(prediction_dir) / str(item['id']) / 'conversation.json').read_text()) for item in items]
        try:
            response, usage = native.chat_optimizer(system=kwargs.get('system_prompt') or 'system',
                user=json.dumps({'skill': skill, 'items': items, 'content': content, 'context': kwargs}),
                max_completion_tokens=16, retries=3, stage='analyst')
            return json.loads(response).get('patch')
        except Exception:
            return None

    def dispatcher(results, skill_content, prediction_dir, patches_dir, workers=1, failure_only=False,
                   minibatch_size=5, edit_budget=1, random_seed=42, *, error_system=None,
                   success_system=None, rejection_context='', trajectory_memory_context='',
                   step_buffer_context='', meta_skill_context='', update_mode='patch',
                   skill_aware_reflection=None, skill_aware_appendix_source=None):
        raise AssertionError('The grouping wrapper must intercept this dispatcher')

    def shuffle(rows, seed):
        rows = list(rows)
        if seed is not None:
            random.Random(seed).shuffle(rows)
        return rows

    native.chat_optimizer = chat
    native.run_error_analyst_minibatch = native.run_success_analyst_minibatch = analyst
    native.run_minibatch_reflect = dispatcher
    native._shuffle_for_minibatch = shuffle
    native._split_minibatches = lambda rows, size: [rows[i:i+size] for i in range(0, len(rows), size)]
    native.is_skill_aware_enabled = lambda: False
    native.get_skill_aware_appendix_source = lambda: 'all'
    native.extract_json = lambda value: json.loads(value) if value else None
    return native


def config():
    return {'minibatch_size': 5, 'reflection_context_tokens': 100,
            'reflection_output_reserve_tokens': 16, 'reflection_safety_margin_tokens': 4}


def counter(messages):
    value = json.loads(messages[1]['content'])
    return {'input_tokens': 10 + len(value['items']) * 20, 'prompt_sha256': sha(messages)}


def setup(tmp_path, total=12):
    rows = [{'id': str(i), 'hard': int(i % 2 == 0)} for i in range(total)]
    for row in rows:
        folder = tmp_path / 'predictions' / row['id']
        folder.mkdir(parents=True)
        (folder / 'conversation.json').write_text(json.dumps(['PRESERVE_ALL_' + row['id']]))
    return dict(results=rows, skill_content='unchanged skill', prediction_dir=str(tmp_path/'predictions'),
                patches_dir=str(tmp_path/'patches'), workers=1, failure_only=False, minibatch_size=5,
                step_buffer_context='previous Skill changes stay in the prompt')


def test_capture_is_exact_and_never_generates(tmp_path):
    native = native_stub()
    data = setup(tmp_path, 2)
    original = native.chat_optimizer
    captured = capture_native_prompt(native, 'succ', data['skill_content'], data['results'], data['prediction_dir'], {})
    assert native.chat_optimizer is original and native.generations == []
    assert 'PRESERVE_ALL_0' in captured['user'] and 'PRESERVE_ALL_1' in captured['user']


def test_authorized_skip_only_overlong_reflection_keeps_rollout_and_resumes(tmp_path):
    import hashlib
    native = native_stub()
    data = setup(tmp_path, 40)
    cfg = config()
    conversation = tmp_path/'predictions/1/conversation.json'
    cfg['reflection_authorized_overlong_conversations'] = {'1': hashlib.sha256(conversation.read_bytes()).hexdigest()}
    def oversized(messages):
        result = counter(messages)
        if any(row['id'] == '1' for row in json.loads(messages[1]['content'])['items']):
            result['input_tokens'] = 101
        return result
    dispatch = build_budgeted_dispatcher(native, cfg, oversized)
    dispatch(**data)
    plan = json.loads((tmp_path/'patches/reflection_groups.json').read_text())
    ids = [i for group in plan['groups'] for i in group['episode_ids']]
    assert len(ids) == 39 and '1' not in ids and len(data['results']) == 40
    receipt = json.loads((tmp_path/'patches/reflection_skips.json').read_text())
    assert len(receipt['skipped']) == 1 and receipt['skipped'][0]['input_tokens'] == 101
    count = len(native.generations)
    dispatch(**data)
    assert len(native.generations) == count
    conversation.write_text('["different episode"]')
    with pytest.raises(ValueError, match='different conversation'):
        dispatch(**data)


def test_authorized_but_fitting_is_not_skipped(tmp_path):
    import hashlib
    data = setup(tmp_path, 2)
    cfg = config()
    cfg['reflection_authorized_overlong_conversations'] = {'1': hashlib.sha256((tmp_path/'predictions/1/conversation.json').read_bytes()).hexdigest()}
    build_budgeted_dispatcher(native_stub(), cfg, counter)(**data)
    assert json.loads((tmp_path/'patches/reflection_skips.json').read_text())['skipped'] == []


@pytest.mark.parametrize('total', [2, 12, 40])
def test_lossless_adaptive_split_and_completed_noop_resume(tmp_path, total):
    native = native_stub()
    data = setup(tmp_path, total)
    dispatch = build_budgeted_dispatcher(native, config(), counter)
    dispatch(**data)
    plan = json.loads((tmp_path/'patches/reflection_groups.json').read_text())
    ids = [identifier for group in plan['groups'] for identifier in group['episode_ids']]
    assert sorted(ids) == sorted(row['id'] for row in data['results'])
    assert len(ids) == len(set(ids)) == total
    assert all(len(group['episode_ids']) <= 3 for group in plan['groups'])
    assert all('previous Skill changes' in group['call']['user'] for group in plan['groups'])
    generated = len(native.generations)
    dispatch(**data)
    assert len(native.generations) == generated
    data['skill_content'] = 'changed Skill'
    with pytest.raises(ValueError, match='checkpoint differs'):
        dispatch(**data)


def test_malformed_noop_is_not_accepted(tmp_path):
    native = native_stub()
    data = setup(tmp_path, 2)
    count = []
    native.chat_optimizer = lambda **kw: (count.append(kw) or '{}', {})
    dispatch = build_budgeted_dispatcher(native, config(), counter)
    with pytest.raises(RuntimeError, match='not a valid no-op'):
        dispatch(**data)
    assert len(count) == 1


def test_single_oversize_aborts_before_any_model_calls(tmp_path):
    native = native_stub()
    data = setup(tmp_path, 12)
    too_big = lambda messages: {'input_tokens': 101, 'prompt_sha256': sha(messages)}
    with pytest.raises(ValueError, match='Single complete trajectory'):
        build_budgeted_dispatcher(native, config(), too_big)(**data)
    assert native.generations == []


def test_late_oversize_prevents_earlier_safe_groups_generating(tmp_path):
    native = native_stub()
    data = setup(tmp_path, 12)
    def count(messages):
        row = counter(messages)
        if 'PRESERVE_ALL_10' in messages[1]['content']:
            row['input_tokens'] = 101
        return row
    with pytest.raises(ValueError, match='Single complete trajectory'):
        build_budgeted_dispatcher(native, config(), count)(**data)
    assert native.generations == []


def test_service_error_cannot_be_swallowed_as_no_patch(tmp_path):
    native = native_stub()
    data = setup(tmp_path, 2)
    def fail(**kwargs):
        raise ConnectionError('service offline')
    native.chat_optimizer = fail
    with pytest.raises(RuntimeError, match='no Skill update'):
        build_budgeted_dispatcher(native, config(), counter)(**data)
    assert list((tmp_path/'patches/group_errors').glob('*.json'))


def test_missing_conversation_does_not_silently_skip(tmp_path):
    native = native_stub()
    data = setup(tmp_path, 2)
    (tmp_path/'predictions/1/conversation.json').write_text('[]')
    with pytest.raises(ValueError, match='Missing complete'):
        build_budgeted_dispatcher(native, config(), counter)(**data)
    assert not native.generations


@pytest.mark.parametrize('field,value', [('reflection_safety_margin_tokens',-1), ('minibatch_size',6), ('reflection_context_tokens',True)])
def test_invalid_budget_rejected(field, value):
    settings = config()
    settings[field] = value
    with pytest.raises(ValueError):
        budget_settings(settings)


def test_output_reserve_must_match_native_request(tmp_path):
    native = native_stub()
    data = setup(tmp_path, 2)
    settings = config()
    settings['reflection_output_reserve_tokens'] = 17
    with pytest.raises(ValueError, match='output limit differs'):
        build_budgeted_dispatcher(native, settings, counter)(**data)
    assert not native.generations
