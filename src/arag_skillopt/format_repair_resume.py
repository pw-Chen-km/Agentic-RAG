"""Continue paid v6 work across an explicit, format-only optimizer repair.

The source run remains immutable. Completed decisions and a fully formed pending
candidate retain their original protocol. Old failed Reflection reports are not
backfilled. Future batches and Meta use the repaired prompts/parser. This is a
mixed-protocol continuation, not a clean rerun of the repaired protocol.
"""
import copy
import hashlib
import shutil
from dataclasses import asdict
from pathlib import Path

from .checkpoint import load_checkpoint, save_checkpoint
from .diagnostics import FORMAT_REPAIR_VERSION, STATE_ACTION_PROTOCOL
from .phase1 import Phase1Config
from .skill_rules import skill_sha256
from .support import digest
from .training_profile import NEW_PROFILE, REPAIRED_PROFILE, prompt_bundle


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(source, runtime, initial):
    source = Path(source).resolve()
    old = load_checkpoint(source / "contract.json")
    prior = old["runtime"]
    previous, current = copy.deepcopy(prior["settings"]), copy.deepcopy(runtime["settings"])
    if (previous["optimizer_prompt_version"] != NEW_PROFILE
            or current["optimizer_prompt_version"] != REPAIRED_PROFILE
            or current["phase1"].get("diagnostic_format_repair") != FORMAT_REPAIR_VERSION):
        raise ValueError("format continuation requires v6 -> v6r1 only")
    for cfg in (previous, current):
        for key in ("name", "output", "optimizer_prompt_version"):
            cfg.pop(key, None)
        cfg["phase1"].pop("diagnostic_format_repair", None)
    if previous != current:
        raise ValueError("format continuation changed non-format settings")
    for key in ("agent_prompt_hash", "tool_schemas_hash", "data_hashes", "judge_prompt_hash", "model_digests"):
        if prior[key] != runtime[key]:
            raise ValueError("format continuation identity changed: " + key)
    if old["initial_skill_hash"] != skill_sha256(initial):
        raise ValueError("format continuation Initial Skill changed")
    allowed = {"src/arag_skillopt/diagnostics.py", "src/arag_skillopt/phase1.py",
               "src/arag_skillopt/training_profile.py", "src/arag_skillopt/meta_stage.py",
               "src/arag_skillopt/candidate_replay.py"}
    root = source.parents[1]
    for name, expected in prior["source_hashes"].items():
        if sha(root / name) != expected:
            raise ValueError("frozen source changed: " + name)
        if runtime["source_hashes"].get(name) != expected and name not in allowed:
            raise ValueError("non-format source changed: " + name)
    files = {str(p.relative_to(source / "phase1")): sha(p)
             for p in sorted((source / "phase1").rglob("*"))
             if p.is_file() and not p.name.startswith(".") and p.name != "contract.json"}
    completed = sorted(int(p.parent.name) for p in (source / "phase1/batches").glob("*/completed.json"))
    if completed != list(range(1, len(completed) + 1)):
        raise ValueError("completed batches are not contiguous")
    pending = []
    for p in (source / "phase1/batches").glob("*/candidate_skill.json"):
        number = int(p.parent.name)
        if number not in completed:
            if number != len(completed) + 1 or not (p.parent / "merge.json").exists():
                raise ValueError("incomplete pending candidate cannot inherit optimizer decisions")
            pending.append(number)
    if len(pending) > 1:
        raise ValueError("multiple pending candidates")
    return {"version": "v6-to-v6r1-format-only-continuation-v1",
            "source_output": str(source), "source_contract_sha256": sha(source / "contract.json"),
            "source_phase1_contract_sha256": sha(source / "phase1/contract.json"),
            "source_phase1_contract_hash": digest(load_checkpoint(source / "phase1/contract.json")),
            "completed_batches": completed, "pending_candidate_batches": pending,
            "phase1_files": files, "phase1_files_hash": digest(files),
            "historical_failed_reflections_backfilled": False,
            "historical_optimizer_decisions_unchanged": True,
            "future_reflections_use": REPAIRED_PROFILE,
            "mixed_protocol_continuation": True}


class NoHistoricalInference:
    def chat(self, **kwargs):
        raise RuntimeError("inherited candidate needs missing optimizer work; refusing historical inference")


def continued_class(base, source):
    source = Path(source).resolve()

    class Continued(base):
        def _run(self, train, validation):
            record = self.contract.get("format_continuation")
            if not record:
                # Workflow passes only its hash to Phase1; attach provenance explicitly.
                record = self._format_continuation
            if sha(source / "contract.json") != record["source_contract_sha256"]:
                raise ValueError("source contract changed after continuation registration")
            if sha(source / "phase1/contract.json") != record["source_phase1_contract_sha256"]:
                raise ValueError("source Phase1 contract changed")
            for name, expected in record["phase1_files"].items():
                if sha(source / "phase1" / name) != expected:
                    raise ValueError("source result changed: " + name)
            previous = load_checkpoint(source / "contract.json")
            if previous["train_hash"] != digest(train) or previous["validation_hash"] != digest(validation):
                raise ValueError("continuation train/validation changed")
            self.contract.update(train_hash=digest(train), validation_hash=digest(validation),
                                 format_continuation=record)
            self.root.mkdir(parents=True, exist_ok=True)
            existing = load_checkpoint(self.root / "contract.json")
            if existing is not None and existing != self.contract:
                raise ValueError("continuation contract changed: use a NEW output")
            if existing is None:
                if any(p.name != ".phase1.lock" for p in self.root.iterdir()):
                    raise ValueError("nonempty continuation without contract")
                save_checkpoint(self.root / "contract.json", self.contract)
                for name, expected in record["phase1_files"].items():
                    target = self.root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / "phase1" / name, target)
                    if sha(target) != expected:
                        raise ValueError("copied continuation result hash mismatch: " + name)
            self._format_record = record
            old_cfg = Phase1Config(**previous["runtime"]["settings"]["phase1"])
            self._historical_reader = base(
                self.root, self.initial, NoHistoricalInference(), self.rollout, self.evaluator,
                prompt_bundle(source.parents[1], previous["runtime"]["settings"], workflow=True),
                self.tool_guidance, old_cfg, self.support_map, replay_for_meta=self.replay_for_meta)
            return super()._run(train, validation)

        def reflect_group(self, kind, skill, episodes, directory, name):
            if int(directory.parent.name) in self._format_record["pending_candidate_batches"]:
                return self._historical_reader.reflect_group(kind, skill, episodes, directory, name)
            return super().reflect_group(kind, skill, episodes, directory, name)

        def merge(self, skill, proposals, directory, stage="merge"):
            if int(directory.name) in self._format_record["pending_candidate_batches"]:
                return self._historical_reader.merge(skill, proposals, directory, stage)
            return super().merge(skill, proposals, directory, stage)

        def replay_contract_hash(self, batch):
            if batch in self._format_record["completed_batches"]:
                return self._format_record["source_phase1_contract_hash"]
            return super().replay_contract_hash(batch)

    return Continued
