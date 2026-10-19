"""Runnable Phase 1 -> candidate TRAIN replay -> batchwise Meta workflow."""

import copy
import fcntl
from dataclasses import asdict
from pathlib import Path

from .checkpoint import load_checkpoint, save_checkpoint
from .compact_rendering import DISPLAY_VERSION
from .diagnostics import VERSION
from .meta_comparison import COMPARISON_VERSION
from .meta_stage import MetaConfig, MetaStage
from .phase1 import Phase1, Phase1Config
from .skill_rules import compile_initial_skill, skill_sha256
from .support import digest


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
    ):
        self.root = Path(output)
        self.cfg = config or Phase1Config()
        self.meta_cfg = meta_config or MetaConfig()
        self.contract = {
            "workflow": "arag-phase1-meta-diagnosis-merge-v3",
            "analysis_protocol": self.cfg.analysis_protocol,
            "optimizer_renderer_version": DISPLAY_VERSION,
            "meta_comparison_version": COMPARISON_VERSION,
            "initial_skill_hash": skill_sha256(skill),
            "phase1_config": asdict(self.cfg),
            "meta_config": asdict(self.meta_cfg),
            "prompts_hash": digest(prompts),
            "tool_guidance_hash": digest(tool_guidance),
            "support_map_hash": digest(support_map or {"questions": {}}),
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
                raise ValueError("nonempty workflow output without contract")
            save_checkpoint(self.root / "contract.json", self.contract)
            save_checkpoint(
                self.root / "checkpoint.json", {"stage": "phase1", "status": "running"}
            )
            phase1_summary = self.phase1.run(train, validation)
            active = load_checkpoint(self.phase1.root / "final_skill.json")
            meta_summary = None
            if self.meta_cfg.enabled:
                bank = [
                    load_checkpoint(
                        self.phase1.root / "batches" / f"{i:03d}" / "completed.json"
                    )["value"]
                    for i in range(1, phase1_summary["completed_batches"] + 1)
                ]
                self.meta = MetaStage(
                    self.phase1, self.root / "meta", active, self.meta_cfg
                )
                save_checkpoint(
                    self.root / "checkpoint.json",
                    {
                        "stage": "meta",
                        "status": "running",
                        "active_skill_hash": skill_sha256(active),
                    },
                )
                meta_summary = self.meta.run(train, validation, bank)
                active = load_checkpoint(self.meta.root / "final_skill.json")
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
