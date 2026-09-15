"""Resumable Retrieval -> Meta -> Answer training, independent of providers.

All expensive operations are journaled. A completed batch receipt, not the
presence of an LLM response, is the authoritative active-Skill transition.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Protocol

from .checkpoint import WorkflowCheckpoint
from .config import WorkflowConfig
from .skill_sections import MARKERS, SECTIONS, SkillSections, validate_stage_patch
from .validation import ValidationMetrics, evaluate_candidate


VERSION = "workflow-runner-v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def skill_hash(skill: str) -> str:
    return hashlib.sha256(skill.encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")
    temp.replace(path)


class Backend(Protocol):
    def rollout(self, questions: list[dict], skill: str, split: str,
                output: Path, purpose: str) -> list[dict]: ...
    def optimize(self, stage: str, operation: str, payload: dict,
                 output: Path) -> dict: ...


def metrics(rows: list[dict]) -> ValidationMetrics:
    return ValidationMetrics(sum(r["correct"] for r in rows) / len(rows),
                             sum(r["tokens"] for r in rows),
                             sum(r["calls"] for r in rows), len(rows))


def apply_sections(skill: str, patch: dict, stage: str) -> str:
    allowed = {"answer_policy"} if stage == "answer" else {"retrieval_policy", "recovery_policy"}
    if not isinstance(patch, dict) or not set(patch) <= allowed:
        raise ValueError("proposal edits forbidden sections")
    result = skill
    for name, value in patch.items():
        if not isinstance(value, str) or not value.strip() or "<!--" in value:
            raise ValueError("invalid section replacement")
        # Fixed query/ref literals are disallowed; semantic scope also gets a
        # separate model review. This guard is not a proof of generalization.
        if re.search(r'\b[ESC]\d+\b|\bquery\s*[:=]|\b(?:gold|ground.truth)\b', value, re.I):
            raise ValueError("literal reference/query or ground-truth runtime rule")
        start, end = MARKERS[name]
        result = result.split(start)[0] + start + "\n" + value.strip() + "\n" + end + result.split(end)[1]
    validate_stage_patch(skill, result, stage)
    return result


class WorkflowRunner:
    def __init__(self, *, backend: Backend, output: Path, config: WorkflowConfig,
                 contract: dict, train: list[dict], validation: list[dict],
                 skill: str, test: list[dict] | None = None):
        self.backend, self.output, self.config = backend, output, config
        self.train, self.validation, self.test, self.initial = train, validation, test or [], skill
        if set(SkillSections.parse(skill).blocks) != set(SECTIONS):
            raise ValueError("seed needs all five marked sections")
        for name, rows in (("train", train), ("validation", validation), ("test", self.test)):
            if name == "test" and not rows:
                continue
            if not rows or len({r["id"] for r in rows}) != len(rows):
                raise ValueError(f"{name} must be nonempty with unique IDs")
            if any(r.get("split", name) != name for r in rows):
                raise ValueError(f"non-{name} row in {name}")
        if {r["id"] for r in train} & {r["id"] for r in validation}:
            raise ValueError("train/validation IDs overlap")
        if self.test and (({r["id"] for r in train} & {r["id"] for r in self.test}) or
                          ({r["id"] for r in validation} & {r["id"] for r in self.test})):
            raise ValueError("train/validation/test IDs overlap")
        normalize = lambda q: " ".join(q.casefold().split())
        if {normalize(r["question"]) for r in train} & {normalize(r["question"]) for r in validation}:
            raise ValueError("train/validation questions overlap")
        if self.test:
            if ({normalize(r["question"]) for r in train} & {normalize(r["question"]) for r in self.test}) or \
               ({normalize(r["question"]) for r in validation} & {normalize(r["question"]) for r in self.test}):
                raise ValueError("train/validation/test questions overlap")
        self.contract = {"version": VERSION, "runtime": contract, "config": asdict(config),
                         "train": digest(train), "validation": digest(validation),
                         "test": digest(self.test) if self.test else None,
                         "initial_skill": skill_hash(skill)}

    def _call(self, stage: str, operation: str, payload: dict, path: Path) -> dict:
        fingerprint = digest({"stage": stage, "operation": operation, "payload": payload})
        if path.exists():
            saved = read_json(path)
            if saved["request_hash"] != fingerprint:
                raise ValueError(f"cached optimizer request mismatch: {path}")
            return saved["response"]
        response = self.backend.optimize(stage, operation, payload, path.with_suffix(".audit.json"))
        write_json(path, {"request_hash": fingerprint, "request": payload, "response": response})
        return response

    def _roll(self, rows: list[dict], skill: str, split: str, purpose: str) -> list[dict]:
        key = digest({"ids": [r["id"] for r in rows], "skill": skill_hash(skill),
                      "split": split, "purpose": purpose})
        folder = self.output / "rollouts" / key
        receipt = folder / "completed.json"
        if receipt.exists():
            result = read_json(receipt)
        else:
            result = self.backend.rollout(rows, skill, split, folder, purpose)
            if [r["id"] for r in result] != [r["id"] for r in rows]:
                raise ValueError("rollout missing/reordered question IDs")
            write_json(receipt, result)
        if [r["id"] for r in result] != [r["id"] for r in rows]:
            raise ValueError("cached rollout IDs mismatch")
        if any(r["split"] != split or r["skill_sha256"] != skill_hash(skill) for r in result):
            raise ValueError("rollout split/skill mismatch")
        return result

    def _merge_drafts(self, stage: str, skill: str, drafts: list[dict],
                      folder: Path) -> dict:
        """Merge reflection results without sending an oversized request.

        A direct merge is retained for small inputs.  For a large set of
        drafts, each group is merged first; only those short merge results are
        then passed to a final merge.  This keeps the proposal content while
        preventing the Meta request from containing every full trajectory.
        """
        def payload_size(items: list[dict]) -> int:
            return len(json.dumps({"skill": skill, "drafts": items},
                                  ensure_ascii=False))

        if payload_size(drafts) <= self.config.max_reflection_input_chars:
            return self._call(stage, "merge", {"skill": skill, "drafts": drafts},
                              folder / "merge.json")

        current = drafts
        level = 0
        while payload_size(current) > self.config.max_reflection_input_chars and len(current) > 1:
            groups: list[list[dict]] = []
            offset = 0
            while offset < len(current):
                group = [current[offset]]
                offset += 1
                while offset < len(current):
                    trial = group + [current[offset]]
                    if payload_size(trial) > self.config.max_reflection_input_chars:
                        break
                    group = trial
                    offset += 1
                groups.append(group)
            merged_groups = []
            for group_index, group in enumerate(groups):
                merged_groups.append(self._call(
                    stage, "summarize_merge",
                    {"skill": skill, "drafts": group},
                    folder / f"merge_summary_level_{level:02d}_{group_index:04d}.json"))
            current = merged_groups
            level += 1

        if len(current) == 1:
            return current[0]
        return self._call(stage, "merge", {"skill": skill, "drafts": current},
                          folder / "merge_final.json")

    def _update(self, stage: str, index: int, skill: str, cases: list[dict],
                questions: list[dict], parent_rows: list[dict]) -> tuple[str, dict]:
        folder = self.output / stage / f"batch_{index:04d}"
        receipt = folder / "completed.json"
        if receipt.exists():
            saved = read_json(receipt)
            if saved["parent_hash"] != skill_hash(skill):
                raise ValueError("batch parent mismatch")
            return saved["active_skill"], saved
        drafts = []
        size = self.config.stages[1].reflection_minibatch_size if stage == "meta" else self.config.reflection_minibatch_size
        # Keep the normal minibatch size, but split an oversized serialized
        # request.  The request number is independent of the case offset so
        # resumed runs retain stable paths (including prior successful calls).
        offset = 0
        request_index = 0
        while offset < len(cases):
            chunk_size = min(size, len(cases) - offset)
            while chunk_size > 1:
                payload = {"skill": skill, "cases": cases[offset:offset + chunk_size]}
                if len(json.dumps(payload, ensure_ascii=False)) <= self.config.max_reflection_input_chars:
                    break
                chunk_size = max(1, chunk_size // 2)
            payload = {"skill": skill, "cases": cases[offset:offset + chunk_size]}
            drafts.append(self._call(stage, "reflect", payload,
                                     folder / f"reflect_{request_index:04d}.json"))
            offset += chunk_size
            request_index += 1
        candidate = skill
        reason = "no_comparison_cases" if not cases else "no_proposal"
        decision = None
        if drafts:
            merged = self._merge_drafts(stage, skill, drafts, folder)
            patch = merged.get("sections", {})
            if patch:
                try:
                    candidate = apply_sections(skill, patch, stage)
                except ValueError as exc:
                    candidate, reason = skill, f"scope_rejected: {exc}"
        candidate_rows = []
        validation_record = None
        test_record = None
        if candidate != skill:
            baseline = self._roll(self.validation, skill, "validation", "validation")
            proposed = self._roll(self.validation, candidate, "validation", "validation")
            baseline_metrics = metrics(baseline)
            proposed_metrics = metrics(proposed)
            validation_record = {"baseline": asdict(baseline_metrics),
                                 "candidate": asdict(proposed_metrics)}
            if self.test:
                baseline_test = self._roll(self.test, skill, "test", "baseline_evaluation")
                proposed_test = self._roll(self.test, candidate, "test", "candidate_evaluation")
                test_record = {"baseline": asdict(metrics(baseline_test)),
                               "candidate": asdict(metrics(proposed_test))}
            if self.config.use_validation_gate:
                decision = asdict(evaluate_candidate(baseline_metrics, proposed_metrics))
                reason = decision["reason"]
            else:
                # Record the same gate statistics for auditability, but make
                # adoption independent of validation.  Test is never used to
                # choose a candidate; it is an external evaluation only.
                observed = evaluate_candidate(baseline_metrics, proposed_metrics)
                decision = {"accepted": True,
                            "reason": "validation_gate_disabled_candidate_adopted",
                            "accuracy_delta": observed.accuracy_delta,
                            "token_gain": observed.token_gain,
                            "call_gain": observed.call_gain}
                reason = decision["reason"]
            # Both accepted branches and rejected replay must be paired on the
            # SAME training questions. Neither validation nor test enters Meta.
            if stage == "retrieval" and (decision["accepted"] or self.config.replay_rejected_candidates):
                candidate_rows = self._roll(questions, candidate, "train", "meta_analysis_only")
        accepted = decision is not None and decision["accepted"]
        active = candidate if accepted else skill
        saved = {"stage": stage, "index": index, "parent_hash": skill_hash(skill),
                 "candidate_hash": skill_hash(candidate), "candidate_skill": candidate,
                 "active_skill": active, "active_hash": skill_hash(active),
                 "accepted": accepted, "reason": reason, "validation": decision,
                 "validation_metrics": validation_record,
                 "test_metrics": test_record,
                 "question_ids": [q["id"] for q in questions],
                 "parent_train": parent_rows, "candidate_train": candidate_rows}
        write_json(receipt, saved)
        return active, saved

    def run(self) -> dict:
        contract_path = self.output / "contract.json"
        if contract_path.exists() and read_json(contract_path) != json.loads(json.dumps(self.contract)):
            raise ValueError("run contract changed; use a fresh output directory")
        write_json(contract_path, self.contract)
        checkpoint = WorkflowCheckpoint(self.output / "checkpoint.json")
        skill = self.initial
        batches = [self.train[i:i+self.config.rollout_batch_size]
                   for i in range(0, len(self.train), self.config.rollout_batch_size)]
        retrieval_receipts, receipts = [], []
        stages = ("retrieval", "meta", "answer") if self.config.enable_meta else ("retrieval", "answer")
        for stage in stages:
            for index, questions in enumerate(batches):
                if stage == "meta":
                    previous = retrieval_receipts[index]
                    pairs = zip(previous["parent_train"], previous["candidate_train"])
                    cases = []
                    for a, b in pairs:
                        if a["id"] != b["id"] or a["split"] != "train" or b["split"] != "train":
                            raise ValueError("Meta requires same-question TRAIN branches")
                        cases.append({"match_level": "same_question", "branch_a": a["view"],
                                      "branch_b": b["view"], "outcomes": [a["correct"], b["correct"]],
                                      "costs": [[a["tokens"], a["calls"]], [b["tokens"], b["calls"]]],
                                      "causal_claim": False})
                    rows = []
                else:
                    rows = self._roll(questions, skill, "train", "training")
                    cases = [r["answer_view"] if stage == "answer" else r["view"] for r in rows]
                skill, receipt = self._update(stage, index, skill, cases, questions, rows)
                if stage == "retrieval":
                    retrieval_receipts.append(receipt)
                receipts.append({k: receipt[k] for k in ("stage", "index", "accepted", "reason", "active_hash")})
                checkpoint.save({"stage": stage, "batch": index, "skill_sha256": skill_hash(skill)})
        # This is the latest accepted version, NOT highest raw-accuracy Skill:
        # the documented gate can also accept an efficiency improvement.
        (self.output / "final_skill.md").write_text(skill)
        final_test = None
        if self.test:
            final_test = asdict(metrics(self._roll(self.test, skill, "test", "final_evaluation")))
        summary = {"status": "complete", "final_skill_sha256": skill_hash(skill),
                   "meta_enabled": self.config.enable_meta, "updates": receipts,
                   "test_executed": bool(self.test), "final_test_metrics": final_test,
                   "validation_gate_enabled": self.config.use_validation_gate}
        write_json(self.output / "summary.json", summary)
        return summary
