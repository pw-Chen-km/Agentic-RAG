"""Runnable Phase 1 -> candidate TRAIN replay -> batchwise Meta workflow."""

import copy
import fcntl
from dataclasses import asdict
from pathlib import Path

from .checkpoint import load_checkpoint, save_checkpoint
from .compact_rendering import DISPLAY_VERSION
from .context_rendering import CONTEXT_RENDERER_VERSION
from .diagnostics import VERSION
from .meta_comparison import COMPARISON_VERSION
from .meta_stage import MetaConfig, MetaStage
from .candidate_replay import collect_train_branches
from .phase1 import Phase1, Phase1Config
from .skill_rules import compile_initial_skill, skill_sha256
from .support import digest
from .action_space import DEFAULT_ACTION_SPACE, action_space_contract


class WorkflowTrainer:
    def __init__(
        self,
        output,
        skill,
        client,
        rollout,
        evaluator,
        prompts,
        tool_guidance,
        config=None,
        support_map=None,
        contract=None,
        meta_config=None,
        action_space=None,
    ):
        self.root = Path(output)
        self.cfg = config or Phase1Config()
        self.meta_cfg = meta_config or MetaConfig()
        self.action_space = tuple(action_space or DEFAULT_ACTION_SPACE)
        self.contract = {
            "workflow": "arag-phase1-meta-diagnosis-merge-v3",
            "analysis_protocol": self.cfg.analysis_protocol,
            "optimizer_renderer_version": DISPLAY_VERSION,
            "optimizer_context_renderer_version": CONTEXT_RENDERER_VERSION,
            "meta_comparison_version": COMPARISON_VERSION,
            "initial_skill_hash": skill_sha256(skill),
            "phase1_config": asdict(self.cfg),
            "meta_config": asdict(self.meta_cfg),
            "prompts_hash": digest(prompts),
            "tool_guidance_hash": digest(tool_guidance),
            "support_map_hash": digest(support_map or {"questions": {}}),
            "action_space": action_space_contract(self.action_space),
            "runtime": copy.deepcopy(contract or {}),
        }
        self.phase1 = Phase1(
            self.root / "phase1",
            skill,
            client,
            rollout,
            evaluator,
            prompts,
            tool_guidance,
            self.cfg,
            support_map,
            {"workflow_contract_hash": digest(self.contract)},
            replay_for_meta=self.meta_cfg.enabled,
            action_space=self.action_space,
        )

    def run(self, train, validation):
        ids = [[str(q["id"]) for q in rows] for rows in (train, validation)]
        if (
            not all(ids)
            or any(len(x) != len(set(x)) for x in ids)
            or set(ids[0]) & set(ids[1])
        ):
            raise ValueError("train/validation must be nonempty, unique and disjoint")
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".workflow.lock").open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "another workflow runner owns this output directory"
                ) from exc
            self.contract.update(
                train_hash=digest(train), validation_hash=digest(validation)
            )
            saved = load_checkpoint(self.root / "contract.json")
            if saved and saved != self.contract:
                raise ValueError(
                    "workflow contract changed: use a NEW output directory"
                )
            if not saved and any(
                p.name != ".workflow.lock" for p in self.root.iterdir()
            ):
                # A fresh run may intentionally inherit only the deterministic
                # Phase 1 episode cache from an interrupted/previous run.  The
                # cache is immutable input for replay; every other artifact
                # must still be rejected so a stale workflow cannot be resumed
                # under a different contract.
                allowed = True
                for entry in self.root.iterdir():
                    if entry.name in {".workflow.lock"}:
                        continue
                    if entry.name != "phase1" or not entry.is_dir():
                        allowed = False
                        break
                    children = list(entry.iterdir())
                    if any(
                        child.name != "episodes" or not child.is_dir()
                        for child in children
                    ):
                        allowed = False
                        break
                if not allowed:
                    raise ValueError("nonempty workflow output without contract")
            save_checkpoint(self.root / "contract.json", self.contract)
            save_checkpoint(
                self.root / "checkpoint.json", {"stage": "interleaved", "status": "running"}
            )
            active = copy.deepcopy(self.phase1.initial)
            phase_receipts, meta_receipts = [], []
            self.meta = MetaStage(
                self.phase1, self.root / "meta", active, self.meta_cfg
            ) if self.meta_cfg.enabled else None
            if self.meta is not None:
                self.meta.root.mkdir(parents=True, exist_ok=True)
                save_checkpoint(self.meta.root / "contract.json", self.meta.contract)
            for offset in range(0, len(train), self.cfg.rollout_batch_size):
                number = offset // self.cfg.rollout_batch_size + 1
                questions = train[offset : offset + self.cfg.rollout_batch_size]
                directory = self.phase1.root / "batches" / f"{number:03d}"
                directory.mkdir(parents=True, exist_ok=True)
                done = load_checkpoint(directory / "completed.json")
                if done:
                    active = done["value"]["active_skill"]
                    phase_receipts.append(done["value"])
                    meta_value = done["value"].get("meta_receipt")
                    if meta_value is not None:
                        meta_receipts.append(meta_value)
                    continue
                parent = copy.deepcopy(active)
                save_checkpoint(directory / "parent_skill.json", parent)
                candidate, change, reflection_count, optimizer_status = self.phase1.prepare_candidate(
                    number, questions, parent, directory
                )
                save_checkpoint(directory / "candidate_skill.json", candidate)
                parent_hash, candidate_hash = skill_sha256(parent), skill_sha256(candidate)
                meta_receipt = None
                refined = candidate
                replay = None
                if self.meta is not None:
                    replay = collect_train_branches(
                        self.phase1, directory, number, questions, parent, candidate, None
                    )
                if self.meta is not None and parent_hash != candidate_hash:
                    phase_input = {
                        "batch": number,
                        "train_ids": [str(q["id"]) for q in questions],
                        "parent_skill_hash": parent_hash,
                        "candidate_skill_hash": candidate_hash,
                        "meta_replay": replay,
                    }
                    refined, meta_receipt = self.meta.prepare_batch(
                        number, questions, phase_input, candidate
                    )
                elif self.meta is not None:
                    meta_receipt = {
                        "batch": number,
                        "source_stage": "meta",
                        "comparison_count": 0,
                        "reflection_count": 0,
                        "optimizer_status": "phase1_no_change",
                        "validation_status": "not_run_no_phase1_change",
                        "active_skill_hash": candidate_hash,
                    }
                final_hash = skill_sha256(refined)
                if final_hash == parent_hash:
                    gate = {"accepted": False, "reason": "no_change"}
                else:
                    parent_val = self.phase1.episodes(
                        validation, parent, "validation", f"batch {number} parent validation"
                    )
                    final_val = self.phase1.episodes(
                        validation, refined, "validation", f"batch {number} final candidate validation"
                    )
                    gate = self.phase1.cached(
                        directory / "validation.json",
                        {"parent": parent_hash, "candidate": final_hash},
                        lambda a=parent_val, b=final_val: self.phase1.validation_gate(a, b),
                    )
                if gate["accepted"]:
                    active = refined
                if meta_receipt is not None:
                    meta_receipt["gate"] = gate
                    meta_receipt["validation_status"] = (
                        "completed" if final_hash != parent_hash else "not_run_no_change"
                    )
                    meta_receipt["active_skill"] = active
                    meta_receipt["active_skill_hash"] = skill_sha256(active)
                    meta_receipts.append(meta_receipt)
                    meta_directory = self.meta.root / "batches" / f"{number:03d}"
                    save_checkpoint(
                        meta_directory / "completed.json",
                        {
                            "input_hash": digest({"parent": parent_hash, "questions": digest(questions)}),
                            "value": meta_receipt,
                        },
                    )
                    save_checkpoint(
                        self.meta.root / "checkpoint.json",
                        {
                            "stage": "meta",
                            "completed_batch": number,
                            "active_skill_hash": skill_sha256(active),
                        },
                    )
                receipt = {
                    **change,
                    "batch": number,
                    "train_ids": [q["id"] for q in questions],
                    "reflection_count": reflection_count,
                    "optimizer_status": optimizer_status,
                    "phase1_candidate_skill_hash": candidate_hash,
                    "meta_receipt": meta_receipt,
                    "gate": gate,
                    "active_skill": active,
                    "active_skill_hash": skill_sha256(active),
                    "meta_replay": (
                        replay
                        if (directory / "train_branches.json").exists() else None
                    ),
                }
                save_checkpoint(directory / "completed.json", {"input_hash": digest({"parent": parent_hash, "questions": digest(questions)}), "value": receipt})
                save_checkpoint(
                    self.phase1.root / "checkpoint.json",
                    {"completed_batch": number, "active_skill_hash": skill_sha256(active)},
                )
                phase_receipts.append(receipt)
            phase1_summary = {
                "phase": "phase1_interleaved",
                "completed_batches": len(phase_receipts),
                "accepted_updates": sum(bool(r["gate"].get("accepted")) for r in phase_receipts),
                "train_count": len(train),
                "validation_count": len(validation),
                "test_executed": False,
                "meta_executed": False,
                "final_skill_hash": skill_sha256(active),
            }
            save_checkpoint(self.phase1.root / "final_skill.json", active)
            (self.phase1.root / "final_skill.md").write_text(compile_initial_skill(active), encoding="utf-8")
            save_checkpoint(self.phase1.root / "summary.json", phase1_summary)
            meta_summary = None
            if self.meta is not None:
                meta_summary = {
                    "status": "complete",
                    "phase": "meta_interleaved",
                    "completed_batches": len(meta_receipts),
                    "accepted_updates": sum(
                        bool(r.get("gate", {}).get("accepted")) for r in meta_receipts
                    ),
                    "comparison_count": sum(r.get("comparison_count", 0) for r in meta_receipts),
                    "reflection_count": sum(r.get("reflection_count", 0) for r in meta_receipts),
                    "test_executed": False,
                    "final_skill_hash": skill_sha256(active),
                }
                save_checkpoint(self.meta.root / "final_skill.json", active)
                (self.meta.root / "final_skill.md").write_text(compile_initial_skill(active), encoding="utf-8")
                save_checkpoint(self.meta.root / "summary.json", meta_summary)
            summary = {
                "status": "complete",
                "phase1": phase1_summary,
                "meta": meta_summary,
                "meta_executed": self.meta_cfg.enabled,
                "test_executed": False,
                "final_skill_hash": skill_sha256(active),
            }
            save_checkpoint(self.root / "final_skill.json", active)
            (self.root / "final_skill.md").write_text(
                compile_initial_skill(active), encoding="utf-8"
            )
            save_checkpoint(self.root / "summary.json", summary)
            save_checkpoint(
                self.root / "checkpoint.json",
                {
                    "stage": "complete",
                    "status": "complete",
                    "active_skill_hash": skill_sha256(active),
                },
            )
            return summary
