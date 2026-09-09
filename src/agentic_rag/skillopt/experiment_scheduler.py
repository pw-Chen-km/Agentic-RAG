"""Durable outer queue for independent SkillOpt runs, not an optimizer resume engine.

The scheduler may disappear while its detached worker continues. A worker owns a
single command and writes a durable exit receipt. Recovery never guesses that a
missing receipt means it is safe to execute the command again. In particular,
native SkillOpt does not promise transactional recovery inside a training step;
failed training requires a separately audited, immutable resume authorization.

All model traffic is in the invoked runners. Endpoint preflight only reads
Ollama's model inventory. Constructing this module or validating a plan does not
contact an endpoint or start a process.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml


SCHEDULER_VERSION = "skillopt-outer-scheduler-v1"
CAPACITY_VERSION = "amd-capacity-v1"
RESUME_VERSION = "skillopt-training-resume-authorization-v1"
_ROOT = Path(__file__).resolve().parents[3]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_sha256(argv: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(argv), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _check_file(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.is_file() or _sha(path) != record.get("sha256"):
        raise ValueError(f"Pinned file changed or missing: {path}")
    if "size_bytes" in record and path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"Pinned file size changed: {path}")
    return path


def _argv(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(part, str) or not part or "\0" in part for part in value):
        raise ValueError("Commands must be nonempty argument arrays, never shell strings")
    return value


def process_identity(pid: int) -> dict[str, Any] | None:
    """Distinguish a live process from reuse of its PID; never kill a process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return {"pid": pid, "unverifiable": True}
    stat = Path(f"/proc/{pid}/stat")
    try:
        # comm may contain spaces or ')'; the remaining fields start at state.
        fields = stat.read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start": fields[19], "host": socket.gethostname(), "source": "proc_starttime"}
    except (OSError, IndexError):
        result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart=", "-o", "stat="], capture_output=True, text=True, check=False)
        parts = result.stdout.strip().split()
        if result.returncode or len(parts) < 6 or parts[-1].startswith("Z"):
            return None
        return {"pid": pid, "start": " ".join(parts[:-1]), "host": socket.gethostname(), "source": "ps_lstart"}


def _live(identity: Mapping[str, Any] | None) -> str:
    if not identity or not isinstance(identity.get("pid"), int):
        return "unknown"
    current = process_identity(identity["pid"])
    if current is None:
        return "dead"
    if current.get("unverifiable") or identity.get("unverifiable"):
        return "unknown"
    return "same" if dict(identity) == current else "reused"


@contextlib.contextmanager
def exclusive_scheduler_lock(directory: Path) -> Iterator[None]:
    """Kernel releases flock on exit; a leftover pathname is not a stale lock."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "scheduler.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another scheduler owns this queue; no processes were started") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ProcessBackend:
    """Small replaceable boundary; unit tests never start a runner or model."""

    def spawn(self, spec_path: Path) -> dict[str, Any]:
        log_path = spec_path.parent / "worker.log"
        with log_path.open("ab", buffering=0) as handle:
            child = subprocess.Popen(
                [sys.executable, str(_ROOT / "scripts" / "run_multidataset_experiments.py"), "--worker-spec", str(spec_path)],
                cwd=_ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
        # Keep the object to reap children while this scheduler remains alive.
        if not hasattr(self, "_children"):
            self._children: dict[int, subprocess.Popen] = {}
        self._children[child.pid] = child
        return process_identity(child.pid) or {"pid": child.pid, "unverifiable": True}

    def live(self, identity: Mapping[str, Any] | None) -> str:
        children = getattr(self, "_children", {})
        for pid, child in list(children.items()):
            if child.poll() is not None:
                del children[pid]
        return _live(identity)

    def preflight(self, lane: Mapping[str, Any], model: str, expected_digest: str) -> dict[str, Any]:
        host = str(lane["host"]).rstrip("/")

        def get(path: str) -> dict[str, Any]:
            request = urllib.request.Request(host + path, method="GET")
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read())
            if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
                raise ValueError("Ollama inventory response is unavailable or malformed")
            return payload

        payload = get("/api/tags")
        matches = [row for row in payload.get("models", []) if model in (row.get("name"), row.get("model"))]
        if len(matches) != 1 or matches[0].get("digest", "").removeprefix("sha256:") != expected_digest:
            raise ValueError("Endpoint model is absent, ambiguous, or has a different digest")
        loaded = get("/api/ps")["models"]
        for row in loaded:
            if row.get("digest", "").removeprefix("sha256:") != expected_digest or model not in (row.get("name"), row.get("model")):
                raise ValueError("Another model is loaded; refusing to replace it or start this run")
            size, size_vram = row.get("size"), row.get("size_vram")
            if row.get("context_length") != 262144:
                raise ValueError("Loaded model context differs from the pinned context")
            if type(size) is not int or type(size_vram) is not int or size <= 0 or size_vram < size:
                raise ValueError("Loaded model has CPU offload or unverified GPU residency")
        return {"at": _now(), "host": lane["host"], "lane_id": lane["lane_id"], "model": model,
                "actual_model_digest": matches[0]["digest"].removeprefix("sha256:"),
                "method": "GET /api/tags + GET /api/ps", "loaded_models_count": len(loaded),
                "loaded_context_length": loaded[0]["context_length"] if loaded else None,
                "gpu_residency": "verified_full" if loaded else "not_loaded_yet"}

    def final_gate(self, manifest: Path, entry: Mapping[str, Any], lane_id: str) -> None:
        from agentic_rag.skillopt.multidataset_configs import validate_final_test_request
        validate_final_test_request(manifest, entry["dataset"], entry["arm"],
                                    source_dataset=entry.get("source_dataset"), lane_id=lane_id)

    def validate_capacity(self, manifest: Path, report: Path, concurrency: int) -> dict[str, Any]:
        from agentic_rag.skillopt.amd_capacity import validate_capacity_report
        return validate_capacity_report(manifest, report, concurrency)


def run_worker(spec_path: Path) -> int:
    """One command, one immutable spec, one durable exit receipt; no retries."""
    spec = _json(spec_path)
    argv = _argv(spec["argv"])
    if command_sha256(argv) != spec.get("command_sha256"):
        raise ValueError("Worker command hash mismatch")
    receipt_path = spec_path.parent / "receipt.json"
    # A repeated invocation of a worker must not execute its command again.
    with (spec_path.parent / "worker.claim").open("x") as claim:
        claim.write(spec["token"])
        claim.flush()
        os.fsync(claim.fileno())
    receipt = {"version": SCHEDULER_VERSION, "token": spec["token"], "task_id": spec["task_id"],
               "stage": spec["stage"], "command_sha256": spec["command_sha256"],
               "state": "running", "started_at": _now(), "worker": process_identity(os.getpid())}
    _atomic_json(receipt_path, receipt)
    try:
        with (spec_path.parent / "command.log").open("ab", buffering=0) as output:
            process = subprocess.Popen(argv, cwd=spec["cwd"], stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT)
            receipt["child"] = process_identity(process.pid) or {"pid": process.pid, "unverifiable": True}
            _atomic_json(receipt_path, receipt)
            code = process.wait()
        receipt.update(state="exited", returncode=code, finished_at=_now())
    except Exception as exc:
        # No traceback or exception string in the status: either might contain secrets.
        receipt.update(state="worker_error", error_type=type(exc).__name__, finished_at=_now())
        code = 125
    _atomic_json(receipt_path, receipt)
    return code


class ExperimentScheduler:
    def __init__(self, manifest_path: Path, progress_dir: Path, *, max_lanes: int = 1,
                 capacity_report: Path | None = None, backend: ProcessBackend | None = None):
        self.manifest_path, self.progress_dir = manifest_path.resolve(), progress_dir.resolve()
        self.manifest = _json(self.manifest_path)
        self.manifest_hash = _sha(self.manifest_path)
        self.backend = backend or ProcessBackend()
        self.status_path = self.progress_dir / "status.json"
        self.max_lanes = max_lanes
        self.capacity_report = capacity_report.resolve() if capacity_report else None
        self._validate_manifest()
        self.lanes = getattr(self, "_approved_lanes", self.manifest["lanes"][:max_lanes])
        self.entries = self.manifest["experiments"] + self.manifest["final_test_commands"]
        if self.manifest.get("aggregation_argv"):
            summary_path = self.manifest.get("aggregation_summary_path")
            if not isinstance(summary_path, str) or not summary_path:
                raise ValueError("Aggregation needs a pinned summary output path")
            aggregation = _argv(self.manifest["aggregation_argv"])
            self.entries.append({"task_id": "aggregate:all", "stage": "aggregate", "ready": True,
                                 "output": str(Path(summary_path).parent), "completion_path": summary_path,
                                 "lane_variants": [{**lane, "execute_argv": aggregation} for lane in self.manifest["lanes"]]})
        self.by_id = {entry["task_id"]: entry for entry in self.entries}
        self.train_ids = [entry["task_id"] for entry in self.manifest["experiments"]]
        self.state: dict[str, Any] = {}

    def _validate_manifest(self) -> None:
        bundle = self.manifest
        if bundle.get("design") != "single_source_cross_dataset_v1":
            raise ValueError("Scheduler only accepts the pinned twelve-training cross-dataset design")
        if len(bundle.get("experiments", [])) != 12 or len(bundle.get("final_test_commands", [])) != 65:
            raise ValueError("Expected exactly twelve training and sixty-five final-test tasks")
        if bundle.get("eval_test_during_training") is not False:
            raise ValueError("Final test must be disabled during training")
        if bundle.get("target_workers_per_task") != 1 or bundle.get("analyst_workers_per_task") != 1:
            raise ValueError("Every training task must use one Target and one reflection worker")
        if self.max_lanes not in (1, 2, 4) or len(bundle.get("lanes", [])) < self.max_lanes:
            raise ValueError("Use one, two, or four lanes only after measured capacity approval")
        lanes = bundle["lanes"]
        if len({lane["lane_id"] for lane in lanes}) != len(lanes) or len({lane["host"] for lane in lanes}) != len(lanes):
            raise ValueError("Lane IDs and endpoints must be unique")
        if not re.fullmatch(r"[0-9a-f]{64}", bundle.get("expected_model_digest", "")):
            raise ValueError("A complete pinned model digest is required")
        entries = bundle["experiments"] + bundle["final_test_commands"]
        if len({entry["task_id"] for entry in entries}) != len(entries):
            raise ValueError("Task IDs must be unique")
        outputs = [entry.get("training_output", entry.get("output")) for entry in entries]
        if any(not value for value in outputs) or len(set(outputs)) != len(outputs):
            raise ValueError("Every task must own a distinct output directory")
        for entry in entries:
            variants = entry.get("lane_variants", [])
            if {variant["lane_id"] for variant in variants} != {lane["lane_id"] for lane in lanes} or len(variants) != len(lanes):
                raise ValueError("Every task needs exactly one pinned variant for every lane")
            for variant in variants:
                lane = next(lane for lane in lanes if lane["lane_id"] == variant["lane_id"])
                if variant.get("host") != lane["host"]:
                    raise ValueError("A task's endpoint does not match its lane")
                _argv(variant["train_argv"] if "training_output" in entry else variant["execute_argv"])
        if self.capacity_report:
            self._validate_capacity()

    def _validate_capacity(self) -> None:
        if self.capacity_report is None:
            raise ValueError("Execution requires an explicit measured capacity report, including one lane")
        report = self.backend.validate_capacity(self.manifest_path, self.capacity_report, self.max_lanes)
        if (report.get("version") != CAPACITY_VERSION or report.get("measured") is not True
                or report.get("execution_manifest_sha256") != self.manifest_hash
                or report.get("model_digest") != self.manifest["expected_model_digest"]
                or report.get("context_length") != 262144
                or type(report.get("approved_concurrency")) is not int
                or report["approved_concurrency"] < self.max_lanes
                or not re.fullmatch(r"[0-9a-f]{64}", report.get("workload_sha256", ""))
                or not isinstance(report.get("headroom_fraction_required"), (int, float))
                or report["headroom_fraction_required"] < 0.1):
            raise ValueError("Capacity report is incomplete, unmeasured, or belongs to another execution bundle")
        stages = [stage for stage in report.get("stages", []) if stage.get("concurrency") == self.max_lanes
                  and stage.get("status") == "passed" and not stage.get("errors")]
        if len(stages) != 1:
            raise ValueError("Requested lane set has no unique passed measured capacity stage")
        lane_ids = stages[0].get("lane_ids", [])
        by_id = {lane["lane_id"]: lane for lane in self.manifest["lanes"]}
        if (len(lane_ids) != self.max_lanes or len(set(lane_ids)) != len(lane_ids)
                or not set(lane_ids).issubset(report.get("approved_lane_ids", []))
                or any(lane_id not in by_id for lane_id in lane_ids)):
            raise ValueError("Measured lane set is not approved in this manifest")
        self._approved_lanes = [by_id[lane_id] for lane_id in lane_ids]
        self.lanes = self._approved_lanes
        for record in report.get("evidence_files", []):
            _check_file(record)

    def plan(self) -> dict[str, Any]:
        return {"version": SCHEDULER_VERSION, "manifest_sha256": self.manifest_hash,
                "training_tasks": 12, "final_test_tasks": 65, "max_lanes": self.max_lanes,
                "aggregation_tasks": int(bool(self.manifest.get("aggregation_argv"))),
                "lanes": [lane["lane_id"] for lane in self.lanes], "model_calls": 0,
                "queue": [entry["task_id"] for entry in self.entries],
                "resume_limit": "Live workers and durable exit receipts resume automatically; failed training needs an audited safe boundary."}

    def _save(self) -> None:
        tasks = self.state["tasks"]
        self.state["updated_at"] = _now()
        self.state["counts"] = {label: sum(task["state"] == label for task in tasks.values())
                                for label in ("pending", "starting", "running", "succeeded", "paused")}
        self.state["state"] = ("complete" if all(task["state"] == "succeeded" for task in tasks.values())
                               else "paused" if any(task["state"] == "paused" for task in tasks.values()) else "running")
        _atomic_json(self.status_path, self.state)

    def _load(self) -> None:
        if self.status_path.exists():
            self.state = _json(self.status_path)
            if self.state.get("version") != SCHEDULER_VERSION or self.state.get("manifest_sha256") != self.manifest_hash:
                raise ValueError("Scheduler checkpoint belongs to a different manifest or format")
            if self.state.get("queue") != [entry["task_id"] for entry in self.entries]:
                raise ValueError("Saved queue order differs from the fixed manifest")
            allowed = {lane["lane_id"] for lane in self.lanes}
            if any(task.get("lane_id") not in allowed for task in self.state["tasks"].values()
                   if task["state"] in {"starting", "running", "paused"} and task.get("lane_id")):
                raise ValueError("Cannot remove an assigned lane from an unfinished queue")
        else:
            # Never adopt old output silently: there may be a live manually started run.
            for entry in self.entries:
                output = Path(entry.get("training_output", entry.get("output")))
                if output.exists() and any(output.iterdir()):
                    raise ValueError(f"Output already exists without this scheduler's ledger: {output}")
            self.state = {"version": SCHEDULER_VERSION, "manifest": str(self.manifest_path),
                          "manifest_sha256": self.manifest_hash, "created_at": _now(),
                          "host": socket.gethostname(), "queue": [entry["task_id"] for entry in self.entries],
                          "tasks": {entry["task_id"]: {"state": "pending", "attempts": []} for entry in self.entries},
                          "events": []}
        if self.state.get("host") != socket.gethostname():
            raise ValueError("This queue's process identities belong to a different host")
        self.state["max_lanes"] = self.max_lanes
        self.state["capacity_report"] = ({"path": str(self.capacity_report), "sha256": _sha(self.capacity_report)}
                                          if self.capacity_report else None)

    def _pause(self, task: dict[str, Any], code: str, **details: Any) -> None:
        task.update(state="paused", pause={"code": code, "at": _now(), **details})
        self._save()

    def _variant(self, entry: Mapping[str, Any], lane_id: str) -> dict[str, Any]:
        return next(variant for variant in entry["lane_variants"] if variant["lane_id"] == lane_id)

    def _validate_files(self, entry: Mapping[str, Any], variant: Mapping[str, Any]) -> None:
        for key in ("initial_skill", "source_optimizer_config", "source_agent_config"):
            _check_file(self.manifest[key])
        for record in self.manifest.get("code_files", []) + self.manifest.get("core_prompt_files", []):
            _check_file(record)
        agent = yaml.safe_load(_check_file(variant["agent_config"]).read_text())
        if agent.get("policy", {}).get("host", "").rstrip("/") != variant["host"]:
            raise ValueError("Target config endpoint disagrees with the assigned lane")
        if "training_output" in entry:
            optimizer = yaml.safe_load(_check_file(variant["skillopt_config"]).read_text())
            if optimizer.get("env", {}).get("workers") != 1 or optimizer.get("gradient", {}).get("analyst_workers") != 1:
                raise ValueError("Nested workers would exceed the lane capacity")
            if optimizer.get("train", {}).get("batch_size") != self.manifest["train_batch_size"] or optimizer.get("gradient", {}).get("minibatch_size") != self.manifest["reflection_minibatch_size"]:
                raise ValueError("Optimizer batch sizes differ from the experiment contract")
            model = optimizer.get("model", {})
            for prefix in ("optimizer_", "target_"):
                host = model.get(prefix + "qwen_chat_base_url", model.get("qwen_chat_base_url", "")).rstrip("/")
                if host != variant["host"] + "/v1":
                    raise ValueError("Optimizer config endpoint disagrees with the assigned lane")

    def _launch(self, entry: dict[str, Any], task: dict[str, Any], lane: dict[str, Any], stage: str) -> None:
        variant = self._variant(entry, lane["lane_id"])
        try:
            if not entry.get("ready"):
                raise ValueError("Task data/configuration is marked unready")
            self._validate_files(entry, variant)
            if stage not in {"seal", "aggregate"}:
                preflight = self.backend.preflight(lane, self.manifest["target_model"], self.manifest["expected_model_digest"])
                if preflight.get("actual_model_digest") != self.manifest["expected_model_digest"] or preflight.get("host") != lane["host"]:
                    raise ValueError("Endpoint preflight did not verify the pinned model")
                task.setdefault("endpoint_audits", []).append(preflight)
            if stage == "test":
                if any(self.state["tasks"][task_id]["state"] != "succeeded" for task_id in self.train_ids):
                    raise ValueError("All twelve training completions are required before any test")
                self.backend.final_gate(self.manifest_path, entry, lane["lane_id"])
            if stage == "aggregate" and any(task["state"] != "succeeded" for key, task in self.state["tasks"].items()
                                             if key != entry["task_id"]):
                raise ValueError("Aggregation requires all twelve trainings and sixty-five tests")
        except Exception as exc:
            task.setdefault("lane_id", lane["lane_id"])
            self._pause(task, "preflight_or_gate_failed", error_type=type(exc).__name__)
            return
        argv = _argv(entry["seal_completion_argv"] if stage == "seal" else variant["train_argv"] if stage == "train" else variant["execute_argv"])
        attempt_number = len(task["attempts"]) + 1
        directory = self.progress_dir / "tasks" / hashlib.sha256(entry["task_id"].encode()).hexdigest()[:20] / f"attempt_{attempt_number:04d}"
        spec_path = directory / "spec.json"
        spec = {"version": SCHEDULER_VERSION, "token": uuid.uuid4().hex, "task_id": entry["task_id"],
                "stage": stage, "argv": argv, "command_sha256": command_sha256(argv), "cwd": str(_ROOT),
                "manifest_sha256": self.manifest_hash, "lane_id": lane["lane_id"], "endpoint": lane["host"],
                "agent_config_sha256": variant["agent_config"]["sha256"],
                "skillopt_config_sha256": variant.get("skillopt_config", {}).get("sha256")}
        # First commit intent; crash at any later point cannot cause silent replay.
        task.update(state="starting", stage=stage, lane_id=lane["lane_id"], endpoint=lane["host"])
        task.pop("pause", None)
        task["attempts"].append({"spec_path": str(spec_path), "receipt_path": str(directory / "receipt.json"),
                                 "token": spec["token"], "stage": stage, "command_sha256": spec["command_sha256"],
                                 "agent_config_sha256": spec["agent_config_sha256"],
                                 "skillopt_config_sha256": spec["skillopt_config_sha256"], "created_at": _now()})
        self._save()
        if spec_path.exists():
            self._pause(task, "attempt_spec_already_exists")
            return
        _atomic_json(spec_path, spec)
        try:
            task["attempts"][-1]["worker"] = self.backend.spawn(spec_path)
            self._save()
        except Exception as exc:
            self._pause(task, "worker_launch_uncertain", error_type=type(exc).__name__)

    def _reconcile(self, entry: dict[str, Any], task: dict[str, Any]) -> None:
        if task["state"] not in {"starting", "running", "paused"} or not task["attempts"]:
            return
        attempt = task["attempts"][-1]
        receipt_path = Path(attempt["receipt_path"])
        if not receipt_path.exists():
            if self.backend.live(attempt.get("worker")) != "same":
                self._pause(task, "missing_exit_receipt_unknown_execution")
            return
        try:
            receipt = _json(receipt_path)
            if any(receipt.get(key) != value for key, value in (
                    ("token", attempt["token"]), ("task_id", entry["task_id"]),
                    ("command_sha256", attempt["command_sha256"]), ("stage", attempt["stage"]))):
                raise ValueError("Receipt identity mismatch")
        except (OSError, ValueError) as exc:
            self._pause(task, "invalid_exit_receipt", error_type=type(exc).__name__)
            return
        attempt["receipt_sha256"] = _sha(receipt_path)
        if receipt.get("state") == "running":
            if self.backend.live(receipt.get("worker")) == "same":
                task.update(state="running")
                task.pop("pause", None)
            else:
                self._pause(task, "worker_disappeared_child_may_still_run", child=receipt.get("child"))
            return
        if receipt.get("state") != "exited" or type(receipt.get("returncode")) is not int or receipt["returncode"] != 0:
            self._pause(task, "command_failed_no_automatic_replay", returncode=receipt.get("returncode"))
            return
        attempt["finished_at"] = receipt.get("finished_at")
        lane = next(lane for lane in self.lanes if lane["lane_id"] == task["lane_id"])
        if attempt["stage"] == "train":
            self._launch(entry, task, lane, "seal")
        else:
            artifact = Path(entry["completion_path"] if attempt["stage"] in {"seal", "aggregate"} else entry["output"])
            if attempt["stage"] == "test":
                artifact = artifact / "summary.json"
            try:
                value = _json(artifact)
                if attempt["stage"] == "seal" and value.get("status") != "complete":
                    raise ValueError("Incomplete training seal")
                if attempt["stage"] == "test" and value.get("complete") is not True:
                    raise ValueError("Incomplete final-test result")
                if attempt["stage"] == "aggregate" and (
                        value.get("final_task_count") != 65 or value.get("additional_judge_calls") != 0
                        or value.get("execution_manifest", {}).get("sha256") != self.manifest_hash):
                    raise ValueError("Incomplete or mismatched aggregate report")
            except (OSError, ValueError) as exc:
                self._pause(task, "success_without_completion_artifact", error_type=type(exc).__name__)
                return
            task.update(state="succeeded", completion={"path": str(artifact), "sha256": _sha(artifact)}, completed_at=_now())
            task.pop("pause", None)

    def tick(self) -> dict[str, Any]:
        """One lock-protected reconciliation/dispatch cycle; no sleeps."""
        with exclusive_scheduler_lock(self.progress_dir):
            self._validate_capacity()
            self._load()
            for entry in self.entries:
                self._reconcile(entry, self.state["tasks"][entry["task_id"]])
            # Fail closed for new work. Already-running independent jobs may finish.
            if not any(task["state"] == "paused" for task in self.state["tasks"].values()):
                busy = {task["lane_id"] for task in self.state["tasks"].values() if task["state"] in {"starting", "running"}}
                all_trained = all(self.state["tasks"][task_id]["state"] == "succeeded" for task_id in self.train_ids)
                for entry in self.entries:
                    task = self.state["tasks"][entry["task_id"]]
                    if task["state"] != "pending" or (entry["task_id"] not in self.train_ids and not all_trained):
                        continue
                    if entry.get("stage") == "aggregate" and any(row["state"] != "succeeded" for key, row in self.state["tasks"].items()
                                                                 if key != entry["task_id"]):
                        continue
                    free = [lane for lane in self.lanes if lane["lane_id"] not in busy and lane["lane_id"] == task.get("lane_id", lane["lane_id"])]
                    if not free:
                        continue
                    lane = free[0]
                    self._launch(entry, task, lane, "train" if entry["task_id"] in self.train_ids else entry.get("stage", "test"))
                    busy.add(lane["lane_id"])
                    if task["state"] == "paused":
                        break
            self._save()
            return self.state

    def authorize_training_resume(self, task_id: str, audit_path: Path) -> None:
        """Explicit audited recovery only; no user data or partial steps are removed."""
        with exclusive_scheduler_lock(self.progress_dir):
            self._load()
            task, entry = self.state["tasks"][task_id], self.by_id[task_id]
            if task_id not in self.train_ids or task["state"] != "paused" or task.get("stage") != "train":
                raise ValueError("Only paused training can use a training-boundary authorization")
            attempt = task["attempts"][-1]
            identities = [attempt.get("worker")]
            receipt_path = Path(attempt["receipt_path"])
            if receipt_path.exists():
                receipt = _json(receipt_path)
                identities.extend([receipt.get("worker"), receipt.get("child")])
                if receipt.get("state") != "exited" or type(receipt.get("returncode")) is not int:
                    raise ValueError("A confirmed command exit is required before training replay")
            else:
                raise ValueError("Missing exit receipt is ambiguous; do not restart training")
            if any(identity and self.backend.live(identity) not in {"dead", "reused"} for identity in identities):
                raise ValueError("Prior worker/child may still be alive; do not restart")
            audit = _json(audit_path)
            expected = {"version": RESUME_VERSION, "status": "verified_safe_boundary", "task_id": task_id,
                        "execution_manifest_sha256": self.manifest_hash, "lane_id": task["lane_id"],
                        "train_command_sha256": command_sha256(self._variant(entry, task["lane_id"])["train_argv"]),
                        "safe_to_resume_without_repeating_update": True}
            if any(audit.get(key) != value for key, value in expected.items()) or not audit.get("checkpoint_files"):
                raise ValueError("Resume authorization is incomplete or belongs to another run")
            for record in audit["checkpoint_files"]:
                _check_file(record)
            task.setdefault("resume_authorizations", []).append({"path": str(audit_path.resolve()), "sha256": _sha(audit_path), "at": _now()})
            task.update(state="pending")
            task.pop("pause", None)
            self._save()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--progress-dir", type=Path)
    parser.add_argument("--max-lanes", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--capacity-report", type=Path)
    parser.add_argument("--execute", action="store_true", help="Actually dispatch the fixed queue; omitted means model-free plan only")
    parser.add_argument("--once", action="store_true", help="One reconciliation tick; workers keep running independently")
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--resume-task")
    parser.add_argument("--resume-authorization", type=Path)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_spec:
        return run_worker(args.worker_spec)
    if not args.manifest:
        parser.error("--manifest is required")
    if not 0 < args.poll_seconds <= 60:
        parser.error("--poll-seconds must be in (0, 60]")
    scheduler = ExperimentScheduler(args.manifest, args.progress_dir or args.manifest.parent / "progress",
                                    max_lanes=args.max_lanes, capacity_report=args.capacity_report)
    if not args.execute:
        if args.resume_task or args.resume_authorization:
            parser.error("Recovery changes state and requires --execute")
        print(json.dumps(scheduler.plan(), ensure_ascii=False, indent=2))
        return 0
    if bool(args.resume_task) != bool(args.resume_authorization):
        parser.error("Recovery requires both --resume-task and --resume-authorization")
    if args.resume_task:
        scheduler.authorize_training_resume(args.resume_task, args.resume_authorization)
    stopped = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopped:
        status = scheduler.tick()
        print(json.dumps({"state": status["state"], "counts": status["counts"], "status_path": str(scheduler.status_path)}), flush=True)
        if args.once or status["state"] == "complete" or (status["state"] == "paused" and not any(
                task["state"] in {"starting", "running"} for task in status["tasks"].values())):
            return 0 if status["state"] == "complete" else 2 if status["state"] == "paused" else 0
        time.sleep(args.poll_seconds)
    # Detached workers are intentionally not terminated by scheduler shutdown.
    return 0
