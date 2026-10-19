"""Raise Phase1/Meta Reflection budgets while retaining paid Agent results.

Agent, Judge, prompts, data and gate must be identical. Completed decisions and a
formed pending candidate remain historical. No failed diagnosis is backfilled.
Raw/evaluation caches remain valid because their runtime has not changed.
"""
import copy
import shutil
from pathlib import Path

from .checkpoint import load_checkpoint, save_checkpoint
from .format_repair_resume import sha
from .skill_rules import skill_sha256
from .support import digest

VERSION = "reflection48k-continuation-v1"
REVISION = {"version": VERSION, "from_input_tokens": 16000,
            "to_input_tokens": 48000, "applies_to": ["phase1_reflection", "meta_reflection"]}


def provenance(source, runtime, initial):
    source = Path(source).resolve()
    old = load_checkpoint(source / "contract.json")
    prior = old["runtime"]
    a, b = copy.deepcopy(prior["settings"]), copy.deepcopy(runtime["settings"])
    if (a["phase1"]["max_input_tokens"] != 16000
            or b["phase1"]["max_input_tokens"] != 48000
            or b["phase1"]["max_merge_input_tokens"] != 48000
            or b["optimizer"].get("transport") != "ollama_exact"
            or b["optimizer"].get("input_token_limit") != 48000
            or b["optimizer"].get("context_length") != 65536
            or b["optimizer"].get("max_tokens") != 8192
            or b["optimizer"].get("reasoning_effort") != "medium"):
        raise ValueError("requires Reflection16000->48000 with unchanged medium8192/64k transport")
    if b.get("experiment", {}).pop("reflection_budget_revision", None) != REVISION:
        raise ValueError("missing explicit Phase1/Meta Reflection revision")
    for cfg in (a, b):
        for key in ("name", "output"):
            cfg.pop(key, None)
        cfg["phase1"].pop("max_input_tokens")
    if a != b:
        raise ValueError("Reflection continuation changed other runtime settings")
    for key in ("agent_prompt_hash", "tool_schemas_hash", "data_hashes", "judge_prompt_hash",
                "model_digests", "optimizer_prompts_hash"):
        if prior.get(key) != runtime.get(key):
            raise ValueError("Reflection continuation identity changed: " + key)
    if old["initial_skill_hash"] != skill_sha256(initial):
        raise ValueError("Reflection continuation Initial Skill changed")
    for name, expected in prior["source_hashes"].items():
        if sha(source.parents[1] / name) != expected or runtime["source_hashes"].get(name) != expected:
            raise ValueError("Reflection-only continuation requires unchanged existing source: " + name)
    added = set(runtime["source_hashes"]) - set(prior["source_hashes"])
    if added - {"src/arag_skillopt/reflection_limit_resume.py"}:
        raise ValueError("unregistered added source")
    if (source / "meta").exists():
        raise ValueError("this continuation supports switching before Meta starts")
    phase = source / "phase1"
    completed = sorted(int(p.parent.name) for p in (phase / "batches").glob("*/completed.json"))
    if completed != list(range(1, len(completed) + 1)):
        raise ValueError("historical completed batches are not contiguous")
    owner = digest(load_checkpoint(phase / "contract.json"))
    phase_contract = load_checkpoint(phase / "contract.json")
    inherited = (phase_contract.get("reflection_limit_continuation")
                 or phase_contract.get("context_continuation")
                 or prior.get("context_continuation") or {})
    owners = {}
    for n in completed:
        value = load_checkpoint(phase / "batches" / f"{n:03d}" / "completed.json")["value"]
        expected = inherited.get("historical_replay_owners", {}).get(str(n), owner)
        if value.get("meta_replay", {}).get("runtime_contract_hash") != expected:
            raise ValueError("historical TRAIN owner mismatch")
        owners[str(n)] = expected
    pending = []
    for d in (phase / "batches").iterdir():
        if int(d.name) in completed:
            continue
        if (int(d.name) != len(completed) + 1 or not (d / "candidate_skill.json").exists()
                or not (d / "merge.json").exists()):
            raise ValueError("need a fully formed pending candidate to preserve optimizer work")
        pending.append(int(d.name))
        # A sealed pending replay, if present, retains its exact owner as well.
        replay = load_checkpoint(d / "train_branches.json")
        if replay:
            if replay["value"]["runtime_contract_hash"] != owner:
                raise ValueError("pending TRAIN owner mismatch")
            owners[d.name.lstrip('0') or '0'] = owner
    if len(pending) != 1:
        raise ValueError("requires one pending candidate")
    files = {str(p.relative_to(phase)): sha(p) for p in sorted(phase.rglob("*"))
             if p.is_file() and not p.name.startswith('.')
             and str(p.relative_to(phase)) not in {"contract.json", "checkpoint.json"}}
    # Derived views can be re-rendered for a later purpose. Raw/evaluation and
    # historical optimizer/decision receipts remain immutable in the new output.
    immutable = [name for name in files if name.startswith("batches/")
                 or (name.startswith("episodes/")
                     and Path(name).name in {"raw.json", "evaluation.json"})]
    return {"version": VERSION, "revision": REVISION, "source_output": str(source),
            "source_contract_sha256": sha(source / "contract.json"),
            "source_phase1_contract_sha256": sha(phase / "contract.json"),
            "completed_batches": completed, "pending_candidate_batches": pending,
            "historical_replay_owners": owners,
            "historical_meta_exclusions": inherited.get("historical_meta_exclusions", {}),
            "copied_files": files, "immutable_files": immutable, "copied_files_hash": digest(files),
            "agent_and_judge_runtime_unchanged": True, "validation_cache_reused": True,
            "historical_reflections_backfilled": False, "pending_candidate_reused": True,
            "mixed_protocol_continuation": True,
            "source_context_continuation": prior.get("context_continuation"),
            "source_checkpoint": load_checkpoint(source / "checkpoint.json")}


def continued_class(base, source):
    source = Path(source).resolve()
    class Continued(base):
        def _run(self, train, validation):
            record = self._reflection_limit_continuation
            for path, expected in ((source / "contract.json", record["source_contract_sha256"]),
                                   (source / "phase1/contract.json", record["source_phase1_contract_sha256"])):
                if sha(path) != expected:
                    raise ValueError("historical contract changed")
            previous = load_checkpoint(source / "contract.json")
            if previous["train_hash"] != digest(train) or previous["validation_hash"] != digest(validation):
                raise ValueError("continuation split changed")
            self.contract.update(train_hash=digest(train), validation_hash=digest(validation),
                                 reflection_limit_continuation=record)
            self.root.mkdir(parents=True, exist_ok=True)
            saved = load_checkpoint(self.root / "contract.json")
            if saved and saved != self.contract:
                raise ValueError("Reflection continuation contract changed")
            if saved is None:
                if any(p.name != ".phase1.lock" for p in self.root.iterdir()):
                    raise ValueError("new Reflection output is not empty")
                save_checkpoint(self.root / "contract.json", self.contract)
            for name, expected in record["copied_files"].items():
                src, dst = source / "phase1" / name, self.root / name
                if sha(src) != expected:
                    raise ValueError("historical result changed: " + name)
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    if sha(dst) != expected:
                        raise ValueError("copied result hash mismatch: " + name)
                elif name in record["immutable_files"] and sha(dst) != expected:
                    raise ValueError("immutable continued result changed: " + name)
            self._reflection_record = record
            return super()._run(train, validation)

        def prepare_candidate(self, number, questions, parent, directory):
            if number not in self._reflection_record["pending_candidate_batches"]:
                return super().prepare_candidate(number, questions, parent, directory)
            old_parent = load_checkpoint(source / "phase1/batches" / f"{number:03d}" / "parent_skill.json")
            if parent != old_parent:
                raise ValueError("pending candidate parent changed")
            merged = load_checkpoint(directory / "merge.json")["value"]
            candidate, change = self.store(parent, self.cfg.max_batch_edits).apply_with_receipt(merged["edits"])
            if candidate != load_checkpoint(directory / "candidate_skill.json"):
                raise ValueError("saved edits do not reconstruct pending candidate")
            count = sum(load_checkpoint(p).get("value", {}).get("status") in {
                "ok", "diagnosis_failed_after_retry", "input_too_long"}
                for p in (directory / "reflections").glob("*.json") if ".attempt" not in p.name)
            return candidate, change, count, merged["status"]

        def replay_contract_hash(self, batch):
            return self._reflection_record["historical_replay_owners"].get(str(batch)) or super().replay_contract_hash(batch)

        def historical_context_exclusion(self, batch, question_id):
            return self._reflection_record["historical_meta_exclusions"].get(str(batch), {}).get(question_id)
    return Continued
