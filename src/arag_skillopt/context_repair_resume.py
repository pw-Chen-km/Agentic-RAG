"""Explicit runtime migration: retain decisions, never mix old/new validation.

Completed TRAIN branches stay under their verified historical contract owner.
A pending candidate is reused without new diagnosis; its gate and both TRAIN
replay branches are measured under the repaired runtime. No TEST data is read.
"""
import copy
import shutil
from pathlib import Path

from .checkpoint import load_checkpoint, save_checkpoint
from .format_repair_resume import sha
from .skill_rules import skill_sha256
from .support import digest
from .training_profile import CONTEXT_REVISION

VERSION = "qwen-context64-continuation-v1"


def provenance(source, runtime, initial):
    source = Path(source).resolve()
    old = load_checkpoint(source / "contract.json")
    prior = old["runtime"]
    a, b = copy.deepcopy(prior["settings"]), copy.deepcopy(runtime["settings"])
    required = {"transport": "ollama_exact", "context_length": 65536,
                "input_token_limit": 60928, "context_safety_tokens": 512,
                "max_tokens": 4096, "reasoning_effort": "none",
                "require_complete_output": True}
    if any(b["llm"].get(k) != v for k, v in required.items()):
        raise ValueError("context migration requires the validated64k Agent profile")
    if b.get("experiment", {}).pop("runtime_context_revision", None) != CONTEXT_REVISION:
        raise ValueError("context migration requires its explicit context revision")
    for cfg in (a, b):
        for key in ("name", "output"):
            cfg.pop(key, None)
        for key in required:
            if key not in {"max_tokens", "reasoning_effort"}:
                cfg["llm"].pop(key, None)
    if a != b:
        raise ValueError("context migration changed non-context settings")
    for key in ("agent_prompt_hash", "tool_schemas_hash", "data_hashes", "judge_prompt_hash", "model_digests"):
        if prior[key] != runtime[key]:
            raise ValueError("context continuation identity changed: " + key)
    if old["initial_skill_hash"] != skill_sha256(initial):
        raise ValueError("Initial Skill changed")
    allowed = {"src/arag/agent/base.py", "src/arag/core/ollama_guard.py",
               "src/arag_skillopt/endpoint_pool.py", "src/arag_skillopt/runtime_audit.py",
               "src/arag_skillopt/phase1_runtime.py", "src/arag_skillopt/phase1.py",
               "src/arag_skillopt/training_profile.py", "scripts/run_phase1.py",
               "src/arag_skillopt/meta_stage.py"}
    for name, expected in prior["source_hashes"].items():
        if sha(source.parents[1] / name) != expected:
            raise ValueError("historical source changed: " + name)
        if runtime["source_hashes"].get(name) != expected and name not in allowed:
            raise ValueError("unrelated source change: " + name)
    additions = set(runtime["source_hashes"]) - set(prior["source_hashes"])
    if additions - {"src/arag_skillopt/context_repair_resume.py"}:
        raise ValueError("unregistered added runtime source")
    phase = source / "phase1"
    completed = sorted(int(p.parent.name) for p in (phase / "batches").glob("*/completed.json"))
    if not completed or completed != list(range(1, len(completed) + 1)):
        raise ValueError("need contiguous completed historical batches")
    old_contract = load_checkpoint(phase / "contract.json")
    old_owner = digest(old_contract)
    inherited = old_contract.get("format_continuation") or prior.get("format_continuation") or {}
    files, owners, excluded = {}, {}, {}
    evidence = runtime.get("launcher", {}).get("context_evidence")
    issues = {}
    if evidence:
        ep = Path(evidence["path"])
        if sha(ep) != evidence["sha256"]:
            raise ValueError("historical context evidence changed")
        for item in load_checkpoint(ep)["confirmed_train_issues"]:
            if item["split"] != "train" or item["reason"] != "confirmed_server_context_truncation":
                raise ValueError("invalid historical context issue")
            raw_path = Path(item["path"]).resolve()
            if not raw_path.is_relative_to(phase / "episodes/train") or sha(raw_path) != item["raw_file_sha256"]:
                raise ValueError("historical context evidence does not match immutable raw")
            issues[str(raw_path)] = item
    def include(path):
        if path.is_file(): files[str(path.relative_to(phase))] = sha(path)
    for n in completed:
        directory = phase / "batches" / f"{n:03d}"
        done = load_checkpoint(directory / "completed.json")["value"]
        replay = done.get("meta_replay")
        owner = inherited["source_phase1_contract_hash"] if n in inherited.get("completed_batches", []) else old_owner
        if not replay or replay["runtime_contract_hash"] != owner:
            raise ValueError("historical replay contract owner mismatch")
        owners[str(n)] = owner
        for p in directory.rglob("*"): include(p)
        for pair in replay["pairs"]:
            for name in ("parent", "candidate"):
                ref = pair[name]
                expected = Path("episodes/train") / ref["skill_hash"] / digest({"id": ref["question_id"]})[:20]
                if ref["split"] != "train" or Path(ref["episode_directory"]) != expected:
                    raise ValueError("historical TRAIN reference mismatch")
                raw_path = phase / expected / "raw.json"
                raw = load_checkpoint(raw_path)["value"]
                incomplete = any(
                    d.get("response", {}).get("raw_response", {}).get("done_reason") == "length"
                    or any(c.get("finish_reason") == "length" for c in d.get("response", {}).get("raw_response", {}).get("choices", []))
                    for d in raw.get("decision_audit", []))
                issue = issues.get(str(raw_path.resolve()))
                if issue and (issue["question_id"] != ref["question_id"] or issue["skill_hash"] != ref["skill_hash"]):
                    raise ValueError("historical context issue identity mismatch")
                if incomplete or issue:
                    excluded.setdefault(str(n), {}).setdefault(ref["question_id"], []).append({
                        "branch": name, "raw_file_sha256": sha(raw_path),
                        "reason": issue["reason"] if issue else "incomplete_output_length",
                        "evidence": issue})
                for p in (phase / expected).rglob("*"): include(p)
    pending = []
    for p in (phase / "batches").glob("*/candidate_skill.json"):
        n = int(p.parent.name)
        if n in completed: continue
        if n != completed[-1] + 1 or not (p.parent / "merge.json").exists():
            raise ValueError("unsupported pending candidate")
        pending.append(n)
        for f in p.parent.rglob("*"):
            if f.name not in {"validation.json", "train_branches.json", "completed.json"}: include(f)
    if len(pending) != 1:
        raise ValueError("require exactly one saved pending candidate")
    return {"version": VERSION, "source_output": str(source),
            "source_contract_sha256": sha(source / "contract.json"),
            "source_phase1_contract_sha256": sha(phase / "contract.json"),
            "completed_batches": completed, "pending_candidate_batches": pending,
            "historical_replay_owners": owners, "copied_files": files,
            "copied_files_hash": digest(files), "historical_validation_reused": False,
            "historical_gate_decisions_recomputed": False,
            "historical_reflections_backfilled": False, "mixed_protocol_continuation": True,
            "pending_replay_policy": "both_branches_new_runtime",
            "historical_meta_exclusions": excluded,
            "context_evidence": evidence,
            "historical_context_limitations": "Historical TRAIN paths may include server truncation; retained as historical, not repaired evidence."}


def continued_class(base, source):
    source = Path(source).resolve()
    class Continued(base):
        def _run(self, train, validation):
            record = self._context_continuation
            for path, expected in ((source / "contract.json", record["source_contract_sha256"]),
                                   (source / "phase1/contract.json", record["source_phase1_contract_sha256"])):
                if sha(path) != expected: raise ValueError("historical contract changed")
            previous = load_checkpoint(source / "contract.json")
            if previous["train_hash"] != digest(train) or previous["validation_hash"] != digest(validation):
                raise ValueError("continuation split changed")
            self.contract.update(train_hash=digest(train), validation_hash=digest(validation), context_continuation=record)
            self.root.mkdir(parents=True, exist_ok=True)
            saved = load_checkpoint(self.root / "contract.json")
            if saved and saved != self.contract: raise ValueError("context continuation contract changed")
            if saved is None:
                if any(p.name != ".phase1.lock" for p in self.root.iterdir()):
                    raise ValueError("new context output is not empty")
                save_checkpoint(self.root / "contract.json", self.contract)
            for name, expected in record["copied_files"].items():
                src, dst = source / "phase1" / name, self.root / name
                if sha(src) != expected: raise ValueError("historical result changed: " + name)
                # Pending parent/candidate are rewritten identically by the runner;
                # historical bytes must remain unchanged on subsequent resumes too.
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
                if sha(dst) != expected: raise ValueError("copied historical file changed: " + name)
            self._context_record = record
            return super()._run(train, validation)

        def prepare_candidate(self, number, questions, parent, directory):
            if number not in self._context_record["pending_candidate_batches"]:
                return super().prepare_candidate(number, questions, parent, directory)
            original_parent = load_checkpoint(source / "phase1/batches" / f"{number:03d}" / "parent_skill.json")
            if parent != original_parent: raise ValueError("pending candidate parent mismatch")
            merged = load_checkpoint(directory / "merge.json")["value"]
            candidate, change = self.store(parent, self.cfg.max_batch_edits).apply_with_receipt(merged["edits"])
            if candidate != load_checkpoint(directory / "candidate_skill.json"):
                raise ValueError("pending edit set does not reconstruct saved candidate")
            count = sum(load_checkpoint(p).get("value", {}).get("status") in {"ok", "diagnosis_failed_after_retry", "input_too_long"}
                        for p in (directory / "reflections").glob("*.json") if ".attempt" not in p.name)
            return candidate, change, count, merged["status"]

        def replay_contract_hash(self, batch):
            owner = self._context_record["historical_replay_owners"].get(str(batch))
            return owner if owner is not None else super().replay_contract_hash(batch)

        def historical_context_exclusion(self, batch, question_id):
            return self._context_record["historical_meta_exclusions"].get(str(batch), {}).get(question_id)
    return Continued
