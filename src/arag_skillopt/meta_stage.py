"""Batchwise Meta optimization of verified, historical TRAIN branches."""

import copy
import fcntl
from collections import Counter
from dataclasses import asdict, dataclass

from .checkpoint import load_checkpoint, save_checkpoint
from .diagnostics import STATE_ACTION_PROTOCOL, diagnostic_tools
from .meta_comparison import (
    comparison_views,
    load_branch,
    make_comparison,
    render_comparisons_with_audit,
)
from .phase1 import Phase1
from .skill_rules import compile_initial_skill, skill_sha256
from .support import digest
from .tokens import count


@dataclass(frozen=True)
class MetaConfig:
    enabled: bool = True
    comparison_batch_size: int = 40
    reflection_minibatch_size: int = 5

    def __post_init__(self):
        if (
            self.comparison_batch_size != 40
            or not 1 <= self.reflection_minibatch_size <= 5
        ):
            raise ValueError("Meta requires comparison batch 40 and reflection <=5")


class MetaStage(Phase1):
    """Reuse bounded native edits/gate; branch collection and scheduling are separate."""

    def __init__(self, phase1, output, initial, config):
        super().__init__(
            output,
            initial,
            phase1.client,
            phase1.rollout,
            phase1.evaluator,
            phase1.prompts,
            phase1.tool_guidance,
            phase1.cfg,
            phase1.support_map,
        )
        self.phase1 = phase1
        self.meta_cfg = config
        self.contract.update(
            phase="arag-meta-diagnosis-merge-v3",
            meta_config=asdict(config),
            phase1_contract_hash=digest(phase1.contract),
        )

    def episode(self, question, skill, split, version):
        # Reuse Phase 1's fixed evaluator/runtime cache. Only gate calls request validation.
        return self.phase1.episode(question, skill, split, version)

    def reflect_comparisons(self, skill, comparisons, directory, name,
                            historical_skill_a=None, historical_skill_b=None):
        ids = [c["question_id"] for c in comparisons]
        payload, render_audit = render_comparisons_with_audit(
            comparisons, plain_language=self.cfg.analysis_protocol == STATE_ACTION_PROTOCOL,
            historical_skill_a=historical_skill_a, historical_skill_b=historical_skill_b,
        )
        save_checkpoint(
            directory / "render_audit" / (name + ".json"),
            {"rendered_text_hash": digest(payload), **render_audit},
        )
        refs = {c["question_id"]: [f"{branch}{d['decision']}"
                                  for branch in ("A", "B")
                                  for d in c["views"][branch]["decisions"]]
                for c in comparisons}
        messages = self.messages("meta", skill, payload, decision_refs=refs)
        tools = diagnostic_tools(self.cfg.analysis_protocol)
        path = directory / (name + ".json")
        if (
            count({"messages": messages, "tools": tools}) > self.cfg.max_input_tokens
            and len(comparisons) > 1
        ):
            middle = len(comparisons) // 2
            save_checkpoint(
                path.with_suffix(".split.json"),
                {
                    "stage": "meta",
                    "case_ids": ids,
                    "reason": "input_token_budget",
                    "split_at": middle,
                },
            )
            return self.reflect_comparisons(
                skill, comparisons[:middle], directory, name + "a",
                historical_skill_a, historical_skill_b,
            ) + self.reflect_comparisons(
                skill, comparisons[middle:], directory, name + "b",
                historical_skill_a, historical_skill_b,
            )
        proposal = self.diagnose(
            "meta",
            skill,
            payload,
            ids,
            path,
            refs,
        )
        return [
            {"proposal_id": name, "analysis_kind": "meta", "case_ids": ids, **proposal}
        ]

    def comparisons(self, number, questions, receipt, directory):
        replay = receipt.get("meta_replay")
        parent = load_checkpoint(
            self.phase1.root / "batches" / f"{number:03d}" / "parent_skill.json"
        )
        candidate = load_checkpoint(
            self.phase1.root / "batches" / f"{number:03d}" / "candidate_skill.json"
        )
        if not parent or not candidate or not replay:
            raise ValueError("Meta requires a completed Phase 1 replay archive")
        a, b = skill_sha256(parent), skill_sha256(candidate)
        expected_ids = [str(q["id"]) for q in questions]
        if (
            receipt["train_ids"] != expected_ids
            or receipt["batch"] != number
            or receipt["parent_skill_hash"] != a
            or receipt["candidate_skill_hash"] != b
            or replay["parent_skill_hash"] != a
            or replay["candidate_skill_hash"] != b
            or replay["batch"] != number
            or replay["question_rows"] != digest(questions)
            or replay["runtime_contract_hash"] != self.phase1.replay_contract_hash(number)
            or replay["purpose"] != "meta_analysis_only"
        ):
            raise ValueError("Meta replay archive lineage mismatch")
        pairs = replay["pairs"]
        if a == b:
            if pairs or replay["status"] != "no_changed_candidate":
                raise ValueError("no-change candidate has unexpected replay branches")
            return [], {"reason": "no_changed_candidate", "outcomes": {}, "skipped": {}}
        if replay["status"] != "completed" or len(pairs) != len(questions):
            raise ValueError("Meta requires every same-batch train branch receipt")
        by_id = {p["question_id"]: p for p in pairs}
        if len(by_id) != len(pairs) or set(by_id) != set(expected_ids):
            raise ValueError("Meta replay IDs are duplicated, missing, or unrelated")
        comparisons, skipped, outcomes = [], Counter(), Counter()
        for question in questions:
            qid = str(question["id"])
            pair = by_id[qid]
            # Archive hash/path/ID errors stop rather than silently skipping corrupt data.
            branch_a = load_branch(self.phase1, pair["parent"], a, qid, question)
            branch_b = load_branch(self.phase1, pair["candidate"], b, qid, question)
            check = getattr(self.phase1, "historical_context_exclusion", None)
            issue = check(number, qid) if check else None
            if issue:
                # Keep both immutable branches. Do not describe client-side
                # history as model-visible when the server is known to have
                # discarded it; exclude the whole pair, not just the bad side.
                save_checkpoint(directory / "comparisons" / (digest(qid)[:20] + ".json"), {
                    "question_id": qid, "eligible_for_reflection": False,
                    "skip_reason": "historical_context_integrity", "evidence": issue})
                skipped["historical_context_integrity"] += 1
                continue
            try:
                comparison = make_comparison(
                    question, parent, candidate, branch_a, branch_b, number
                )
                entry = self.support_map["questions"].get(qid)
                comparison["views"] = comparison_views(branch_a, branch_b, entry)
            except (ValueError, KeyError, TypeError) as exc:
                comparison = {
                    "question_id": qid,
                    "eligible_for_reflection": False,
                    "skip_reason": "malformed_trajectory",
                    "error": str(exc),
                }
            save_checkpoint(
                directory / "comparisons" / (digest(qid)[:20] + ".json"), comparison
            )
            if "outcomes" in comparison:
                va, vb = comparison["outcomes"]["A"], comparison["outcomes"]["B"]
                category = (
                    "unknown"
                    if type(va) is not bool or type(vb) is not bool
                    else "both_correct"
                    if va and vb
                    else "candidate_only"
                    if vb
                    else "parent_only"
                    if va
                    else "both_wrong"
                )
                outcomes[category] += 1
            if comparison["eligible_for_reflection"]:
                comparisons.append(comparison)
            else:
                skipped[comparison["skip_reason"]] += 1
        return comparisons, {
            "reason": "collected",
            "outcomes": dict(outcomes),
            "skipped": dict(skipped),
        }

    def run(self, train, validation, phase1_receipts):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".meta.lock").open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "another Meta runner owns this output directory"
                ) from exc
            return self._run_meta(train, validation, phase1_receipts)

    def _run_meta(self, train, validation, phase1_receipts):
        if self.phase1.contract.get("train_hash") != digest(
            train
        ) or self.phase1.contract.get("validation_hash") != digest(validation):
            raise ValueError(
                "Meta must use the exact completed Phase 1 train/validation split"
            )
        self.contract.update(
            train_hash=digest(train),
            validation_hash=digest(validation),
            bank_hash=digest(phase1_receipts),
        )
        saved = load_checkpoint(self.root / "contract.json")
        if saved and saved != self.contract:
            raise ValueError("Meta contract changed: use a NEW output directory")
        if not saved and any(p.name != ".meta.lock" for p in self.root.iterdir()):
            raise ValueError("nonempty Meta output without contract")
        save_checkpoint(self.root / "contract.json", self.contract)
        batches = [
            train[i : i + self.meta_cfg.comparison_batch_size]
            for i in range(0, len(train), self.meta_cfg.comparison_batch_size)
        ]
        if len(batches) != len(phase1_receipts):
            raise ValueError("Meta bank must correspond to the Phase 1 train batches")
        active, receipts = copy.deepcopy(self.initial), []
        for number, (questions, historical) in enumerate(
            zip(batches, phase1_receipts), 1
        ):
            directory = self.root / "batches" / f"{number:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            inputs = {
                "current_active_skill_hash": skill_sha256(active),
                "historical_receipt": digest(historical),
                "questions": digest(questions),
            }
            # Re-validate immutable branch files on resume as well as fresh runs.
            comparisons, stats = self.comparisons(
                number, questions, historical, directory
            )
            done = load_checkpoint(directory / "completed.json")
            if done:
                if done["input_hash"] != digest(inputs):
                    raise ValueError("Meta completed batch lineage mismatch")
                active = done["value"]["active_skill"]
                receipts.append(done["value"])
                continue
            parent = copy.deepcopy(active)
            save_checkpoint(directory / "parent_skill.json", parent)
            size = self.meta_cfg.reflection_minibatch_size
            historical_skill_a = load_checkpoint(
                self.phase1.root / "batches" / f"{number:03d}" / "parent_skill.json"
            )
            historical_skill_b = load_checkpoint(
                self.phase1.root / "batches" / f"{number:03d}" / "candidate_skill.json"
            )
            jobs = [
                (comparisons[i : i + size], f"meta-{i:03d}")
                for i in range(0, len(comparisons), size)
            ]
            nested = self.map_ordered(
                lambda job, parent=parent, directory=directory: (
                    self.reflect_comparisons(
                        parent, job[0], directory / "reflections", job[1],
                        historical_skill_a, historical_skill_b,
                    )
                ),
                jobs,
                self.cfg.reflection_workers,
            )
            proposals = [p for group in nested for p in group]
            merged = self.merge(parent, proposals, directory, stage="meta_merge")
            candidate, change = self.store(
                parent, self.cfg.max_batch_edits
            ).apply_with_receipt(merged["edits"])
            save_checkpoint(directory / "candidate_skill.json", candidate)
            if skill_sha256(parent) == skill_sha256(candidate):
                gate = {
                    "accepted": False,
                    "reason": "no_change",
                    "optimizer_status": merged["status"],
                }
            else:
                parent_val = self.episodes(
                    validation, parent, "validation", f"meta {number} parent validation"
                )
                candidate_val = self.episodes(
                    validation,
                    candidate,
                    "validation",
                    f"meta {number} candidate validation",
                )
                gate = self.cached(
                    directory / "validation.json",
                    {
                        "parent": skill_sha256(parent),
                        "candidate": skill_sha256(candidate),
                    },
                    lambda a=parent_val, b=candidate_val: self.validation_gate(a, b),
                )
            if gate["accepted"]:
                active = candidate
            receipt = {
                **change,
                "source_stage": "meta",
                "batch": number,
                "evidence_parent_skill_hash": historical["parent_skill_hash"],
                "evidence_candidate_skill_hash": historical["candidate_skill_hash"],
                "current_active_skill_hash": skill_sha256(parent),
                "proposal_parent_skill_hash": skill_sha256(parent),
                "train_ids": [q["id"] for q in questions],
                "comparison_count": len(comparisons),
                "comparison_statistics": stats,
                "reflection_count": len(proposals),
                "gate": gate,
                "active_skill": active,
                "active_skill_hash": skill_sha256(active),
            }
            save_checkpoint(
                directory / "completed.json",
                {"input_hash": digest(inputs), "value": receipt},
            )
            save_checkpoint(
                self.root / "checkpoint.json",
                {
                    "stage": "meta",
                    "completed_batch": number,
                    "active_skill_hash": skill_sha256(active),
                },
            )
            receipts.append(receipt)
        summary = {
            "status": "complete",
            "phase": "meta",
            "completed_batches": len(receipts),
            "accepted_updates": sum(r["gate"]["accepted"] for r in receipts),
            "comparison_count": sum(r["comparison_count"] for r in receipts),
            "reflection_count": sum(r["reflection_count"] for r in receipts),
            "test_executed": False,
            "final_skill_hash": skill_sha256(active),
        }
        save_checkpoint(self.root / "final_skill.json", active)
        (self.root / "final_skill.md").write_text(
            compile_initial_skill(active), encoding="utf-8"
        )
        save_checkpoint(self.root / "summary.json", summary)
        return summary
