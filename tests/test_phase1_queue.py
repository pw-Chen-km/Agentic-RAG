"""Queue guards tested offline, without importing any model runtime."""

import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "phase1_after_hro", Path(__file__).resolve().parents[1] / "operations/phase1_after_hro.py"
)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def completed_results(tmp_path):
    upstream = tmp_path / "hro" / "results"
    upstream.mkdir(parents=True)
    (upstream / "checkpoint.json").write_text(json.dumps({"stage": "complete", "status": "complete"}))
    return upstream


def test_uses_wrapper_receipt_beside_results(tmp_path):
    upstream = completed_results(tmp_path)
    (upstream.parent / "run.exit").write_text("EXIT=0\n")
    assert queue.upstream_state(upstream)[0] == "ready"


def test_unrelated_ancestor_receipt_cannot_release_queue(tmp_path):
    upstream = completed_results(tmp_path)
    (upstream.parent.parent / "run.exit").write_text("EXIT=0\n")
    assert queue.upstream_state(upstream)[0] == "waiting_for_hro_complete"


def test_complete_checkpoint_still_waits_for_wrapper_exit(tmp_path):
    assert queue.upstream_state(completed_results(tmp_path))[0] == "waiting_for_hro_complete"


def test_exit_zero_cannot_release_incomplete_checkpoint(tmp_path):
    upstream = completed_results(tmp_path)
    (upstream / "checkpoint.json").write_text(json.dumps({"stage": "interleaved", "status": "running"}))
    (upstream.parent / "run.exit").write_text("EXIT=0\n")
    assert queue.upstream_state(upstream)[0] == "waiting_for_hro_complete"


def test_failed_upstream_does_not_wait_forever(tmp_path):
    upstream = completed_results(tmp_path)
    (upstream / "checkpoint.json").write_text(json.dumps({"stage": "interleaved", "status": "running"}))
    (upstream.parent / "run.exit").write_text("EXIT=1\n")
    assert queue.upstream_state(upstream)[0] == "blocked_upstream_failure"


def test_invalid_checkpoint_fails_closed(tmp_path):
    upstream = completed_results(tmp_path)
    (upstream / "checkpoint.json").write_text("{")
    assert queue.upstream_state(upstream)[0] == "upstream_checkpoint_invalid"
