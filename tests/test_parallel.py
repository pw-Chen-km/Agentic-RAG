import copy
import threading

import pytest
from test_meta_training import engine
from test_phase1 import questions

from arag_skillopt.endpoint_pool import EndpointPool, PooledClient
from arag_skillopt.phase1 import Phase1Config


def test_endpoint_is_returned_on_exception_and_records_replica():
    pool = EndpointPool(["a", "b"])
    with pytest.raises(RuntimeError), pool.lease() as url:
        assert url == "a"
        raise RuntimeError("provider failure")
    with pool.lease() as url:
        assert url == "b"
    assert pool.available.qsize() == 2

    class Fake:
        def chat(self, **kwargs):
            return {"message": {"content": "ok"}}

    client = PooledClient({"a": Fake(), "b": Fake()})
    assert [client.chat()["endpoint"] for _ in range(4)] == ["a", "b", "a", "b"]


def test_ordered_map_really_uses_two_workers_without_reordering(tmp_path):
    trainer = engine(tmp_path, [])
    barrier = threading.Barrier(2, timeout=5)
    threads = set()
    guard = threading.Lock()

    def run(item):
        with guard:
            threads.add(threading.get_ident())
        barrier.wait()
        return item * 2

    assert trainer.phase1.map_ordered(run, [3, 1, 4, 2], 2) == [6, 2, 8, 4]
    assert len(threads) == 2


def test_parallel_workflow_preserves_batch_boundary_and_resume(tmp_path):
    trainer = engine(tmp_path, [])
    cfg = Phase1Config(episode_workers=2, reflection_workers=2)
    trainer.cfg = trainer.phase1.cfg = cfg
    # Fresh contract must record workers too, before any files are written.
    from dataclasses import asdict

    trainer.contract["phase1_config"] = asdict(cfg)
    trainer.phase1.contract["config"] = asdict(cfg)
    summary = trainer.run(questions("t", 40), questions("v", 4))
    assert summary["meta"]["comparison_count"] == 40
    assert summary["meta"]["accepted_updates"] == 1
    assert summary["phase1"]["accepted_updates"] == 1
    before = copy.deepcopy(trainer.phase1.client.requests)
    assert trainer.run(questions("t", 40), questions("v", 4)) == summary
    assert trainer.phase1.client.requests == before


def test_invalid_parallel_settings_rejected():
    with pytest.raises(ValueError):
        Phase1Config(episode_workers=0)
    with pytest.raises(ValueError):
        Phase1Config(reflection_workers=9)
