"""Batchwise Meta optimization of verified, historical TRAIN branches."""

import copy
import fcntl
from collections import Counter
from dataclasses import asdict, dataclass

from .checkpoint import load_checkpoint, save_checkpoint
from .context_rendering import (
    CONTEXT_RENDERER_VERSION,
    render_stage_a_context,
    render_stage_b_context,
)
from .diagnostics import STATE_ACTION_PROTOCOL, diagnostic_tools
from .meta_comparison import (
    comparison_views,
    load_branch,
    make_comparison,
    render_comparisons_with_audit,
    rule_differences,
)
from .paired_summary import HRO_VERSION, compact_pair_record
from .hro_v1 import (
    GROUPS,
    UNSCORED,
    build_stage_a_context,
    build_stage_b_context,
    group_summary,
    pair_group,
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
    informative_pairs_only: bool = True
    max_edits_per_update: int = 3
    min_informative_pairs: int = 3
    accuracy_floor: float | None = None
    hro_v1: bool = False

    def __post_init__(self):
        if (
            self.comparison_batch_size != 40
            or not 1 <= self.reflection_minibatch_size <= 5
        ):
            raise ValueError("Meta requires comparison batch 40 and reflection <=5")
        if type(self.informative_pairs_only) is not bool:
            raise ValueError("informative_pairs_only must be boolean")
        if type(self.max_edits_per_update) is not int or not 1 <= self.max_edits_per_update <= 3:
            raise ValueError("max_edits_per_update must be between 1 and 3")
        if type(self.min_informative_pairs) is not int or self.min_informative_pairs < 0:
            raise ValueError("min_informative_pairs must be nonnegative")
        if self.accuracy_floor is not None and (
            isinstance(self.accuracy_floor, bool)
            or not isinstance(self.accuracy_floor, (int, float))
            or not 0 <= self.accuracy_floor <= 1
        ):
            raise ValueError("accuracy_floor must be a fraction in [0,1]")
        if type(self.hro_v1) is not bool:
            raise ValueError("hro_v1 must be boolean")


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
            action_space=phase1.action_space,
        )
        self.phase1 = phase1
        self.meta_cfg = config
        self.contract.update(
            phase="hro-meta-paired-trajectory-v1",
            hro_version=HRO_VERSION,
            meta_config=asdict(config),
            phase1_contract_hash=digest(phase1.contract),
            hro_v1=self.meta_cfg.hro_v1,
        )

    def episode(self, question, skill, split, version):
        # Reuse Phase 1's fixed evaluator/runtime cache. Only gate calls request validation.
        return self.phase1.episode(question, skill, split, version)

    def validation_gate(self, parent, candidate):
        result = super().validation_gate(parent, candidate)
        floor = self.meta_cfg.accuracy_floor
        if floor is None or not result.get("accepted"):
            return result
        parent_accuracy = result.get("parent", {}).get("accuracy")
        candidate_accuracy = result.get("candidate", {}).get("accuracy")
        if (
            isinstance(parent_accuracy, (int, float))
            and isinstance(candidate_accuracy, (int, float))
            and candidate_accuracy < parent_accuracy + floor
        ):
            return {
                **result,
                "accepted": False,
                "reason": "accuracy_floor",
                "accuracy_floor": floor,
            }
        return result

    def reflect_comparisons(self, skill, comparisons, directory, name,
                            historical_skill_a=None, historical_skill_b=None):
        ids = [c["question_id"] for c in comparisons]
        if self.meta_cfg.hro_v1:
            if not comparisons:
                raise ValueError("HRO Stage A requires at least one informative pair")
            diffs = comparisons[0].get("rule_differences", [])
            context = build_stage_a_context(
                pair_group(comparisons[0]),
                comparisons,
                changed_rule_ids=[d["rule_id"] for d in diffs],
                parent_rules=(historical_skill_a or {}).get("retrieval_policy", []),
                candidate_rules=(historical_skill_b or {}).get("retrieval_policy", []),
                action_space=self.action_space,
            )
            save_checkpoint(
                directory / "stage_a_context" / (name + ".json"), context
            )
        comparison_payload, comparison_audit = render_comparisons_with_audit(
            comparisons, plain_language=self.cfg.analysis_protocol == STATE_ACTION_PROTOCOL,
            historical_skill_a=historical_skill_a, historical_skill_b=historical_skill_b,
        )
        # HRO has a bounded deterministic context contract.  The detailed
        # comparison rendering remains an audit artifact, while the model sees
        # the explicit Stage A sections (group meaning, rule delta, metrics,
        # and paired cases) instead of having to reconstruct them itself.
        payload = comparison_payload
        render_audit = comparison_audit
        if self.meta_cfg.hro_v1:
            payload = render_stage_a_context(context)
            render_audit = {
                "comparison": comparison_audit,
                "context_renderer_version": CONTEXT_RENDERER_VERSION,
            }
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
            {
                "proposal_id": name,
                "analysis_kind": "meta",
                "group": pair_group(comparisons[0]) if comparisons else UNSCORED,
                "case_ids": ids,
                **proposal,
            }
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
            markers = [
                compact_pair_record({
                    "question_id": str(q["id"]),
                    "question": q.get("question"),
                    "eligible_for_reflection": False,
                    "skip_reason": "no_changed_candidate",
                    "transition_group": UNSCORED,
                    "outcomes": {"A": None, "B": None},
                })
                for q in questions
            ] if self.meta_cfg.hro_v1 else []
            save_checkpoint(
                directory / "comparison_pairs.json",
                {
                    "schema_version": HRO_VERSION,
                    "batch": number,
                    "pairs": markers,
                    "informative_pair_count": 0,
                    "excluded_pair_count": len(markers),
                    "group_counts": {group: 0 for group in GROUPS},
                    "unscored_count": len(markers),
                },
            )
            return [], {
                "reason": "no_changed_candidate",
                "outcomes": {},
                "skipped": {"no_changed_candidate": len(markers)} if markers else {},
                "paired_count": len(markers),
                "informative_pair_count": 0,
                "group_summaries": {group: group_summary(markers, group) for group in GROUPS},
                "unscored_count": len(markers),
            }
        if replay["status"] != "completed" or len(pairs) != len(questions):
            raise ValueError("Meta requires every same-batch train branch receipt")
        by_id = {p["question_id"]: p for p in pairs}
        if len(by_id) != len(pairs) or set(by_id) != set(expected_ids):
            raise ValueError("Meta replay IDs are duplicated, missing, or unrelated")
        comparisons, skipped, outcomes = [], Counter(), Counter()
        pair_records = []
        for question in questions:
            qid = str(question["id"])
            pair = by_id[qid]
            # Archive hash/path/ID errors stop rather than silently skipping corrupt data.
            try:
                branch_a = load_branch(self.phase1, pair["parent"], a, qid, question)
                branch_b = load_branch(self.phase1, pair["candidate"], b, qid, question)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                if not self.meta_cfg.hro_v1:
                    raise
                # HRO keeps a visible row for every TRAIN question while
                # excluding corrupt artifacts from Stage A.  The marker is
                # explicit and never treated as an answer-level failure.
                marker = {
                    "question_id": qid,
                    "question": question.get("question"),
                    "eligible_for_reflection": False,
                    "skip_reason": "missing_or_corrupt_branch",
                    "error": str(exc),
                    "transition_group": UNSCORED,
                    "outcomes": {"A": None, "B": None},
                }
                save_checkpoint(
                    directory / "comparisons" / (digest(qid)[:20] + ".json"), marker
                )
                pair_records.append(compact_pair_record(marker))
                skipped["missing_or_corrupt_branch"] += 1
                continue
            check = getattr(self.phase1, "historical_context_exclusion", None)
            issue = check(number, qid) if check else None
            if issue:
                # Keep both immutable branches. Do not describe client-side
                # history as model-visible when the server is known to have
                # discarded it; exclude the whole pair, not just the bad side.
                marker = {
                    "question_id": qid, "eligible_for_reflection": False,
                    "skip_reason": "historical_context_integrity", "evidence": issue,
                    "transition_group": UNSCORED,
                    "outcomes": {"A": None, "B": None},
                }
                save_checkpoint(directory / "comparisons" / (digest(qid)[:20] + ".json"), marker)
                pair_records.append(compact_pair_record(marker))
                skipped["historical_context_integrity"] += 1
                continue
            try:
                comparison = make_comparison(
                    question, parent, candidate, branch_a, branch_b, number
                )
                entry = self.support_map["questions"].get(qid)
                comparison["views"] = comparison_views(branch_a, branch_b, entry)
                comparison["transition_group"] = pair_group(comparison)
            except (ValueError, KeyError, TypeError) as exc:
                comparison = {
                    "question_id": qid,
                    "eligible_for_reflection": False,
                    "skip_reason": "malformed_trajectory",
                    "error": str(exc),
                    "transition_group": UNSCORED,
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
            pair_records.append(compact_pair_record(comparison))
            if comparison.get("eligible_for_reflection") and comparison.get("transition_group") != UNSCORED and (
                self.meta_cfg.informative_pairs_only
                or comparison.get("informative")
            ):
                comparisons.append(comparison)
            else:
                reason = comparison.get("skip_reason") or "not_informative_pair"
                if comparison.get("eligible_for_reflection") and self.meta_cfg.informative_pairs_only:
                    reason = "not_informative_pair"
                skipped[reason] += 1
        save_checkpoint(
            directory / "comparison_pairs.json",
            {
                "schema_version": HRO_VERSION,
                "batch": number,
                "pairs": pair_records,
                "informative_pair_count": len(comparisons),
                "excluded_pair_count": len(pair_records) - len(comparisons),
                "group_counts": {
                    group: sum(1 for p in pair_records if p.get("transition_group") == group)
                    for group in GROUPS
                },
                "unscored_count": sum(
                    1 for p in pair_records if p.get("transition_group") == UNSCORED
                ),
            },
        )
        return comparisons, {
            "reason": "collected",
            "outcomes": dict(outcomes),
            "skipped": dict(skipped),
            "paired_count": len(pair_records),
            "informative_pair_count": len(comparisons),
            "group_summaries": {
                group: group_summary(pair_records, group) for group in GROUPS
            },
            "unscored_count": sum(
                1 for p in pair_records if p.get("transition_group") == UNSCORED
            ),
        }

    def _propose_from_comparisons(self, number, parent, directory, comparisons, stats):
        """Produce one Meta candidate without validation or active-Skill adoption."""
        size = self.meta_cfg.reflection_minibatch_size
        historical_skill_a = load_checkpoint(
            self.phase1.root / "batches" / f"{number:03d}" / "parent_skill.json"
        )
        historical_skill_b = load_checkpoint(
            self.phase1.root / "batches" / f"{number:03d}" / "candidate_skill.json"
        )
        if self.meta_cfg.hro_v1:
            # Stage A is grouped by deterministic answer transition.  A
            # group can be split into multiple <=5-case calls, but a call
            # never mixes G00/G01/G10/G11.
            jobs = []
            for group in GROUPS:
                members = [c for c in comparisons if pair_group(c) == group]
                jobs.extend(
                    (members[i : i + size], f"{group.lower()}-{i:03d}")
                    for i in range(0, len(members), size)
                )
        else:
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
        if self.meta_cfg.hro_v1:
            stage_b = build_stage_b_context(
                parent,
                historical_skill_b,
                [d["rule_id"] for d in rule_differences(historical_skill_a, historical_skill_b)],
                stats.get("group_summaries") or {
                    group: group_summary(comparisons, group) for group in GROUPS
                },
                proposals,
                [c for group in GROUPS for c in comparisons if pair_group(c) == group][:20],
                stats,
                action_space=self.action_space,
                max_edits=self.meta_cfg.max_edits_per_update,
            )
            save_checkpoint(directory / "stage_b_context.json", stage_b)
            stage_b_payload = render_stage_b_context(stage_b)
            save_checkpoint(
                directory / "stage_b_render_audit.json",
                {
                    "renderer_version": CONTEXT_RENDERER_VERSION,
                    "rendered_text_hash": digest(stage_b_payload),
                    "canonical_fields": sorted(stage_b),
                },
            )
        else:
            stage_b_payload = None
        merged = self.merge(
            parent,
            proposals,
            directory,
            stage="meta_merge",
            edit_limit=self.meta_cfg.max_edits_per_update,
            context_payload=stage_b_payload,
        )
        candidate, change = self.store(
            parent, self.meta_cfg.max_edits_per_update
        ).apply_with_receipt(merged["edits"])
        save_checkpoint(directory / "candidate_skill.json", candidate)
        return candidate, change, len(proposals), merged["status"]

    def prepare_batch(self, number, questions, receipt, candidate):
        """Refine this batch's Phase 1 candidate before its first validation."""
        directory = self.root / "batches" / f"{number:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        if skill_sha256(candidate) != receipt["candidate_skill_hash"]:
            raise ValueError("Meta input must be this batch's Phase 1 candidate")
        comparisons, stats = self.comparisons(number, questions, receipt, directory)
        save_checkpoint(directory / "parent_skill.json", candidate)
        if len(comparisons) < self.meta_cfg.min_informative_pairs:
            refined, change = self.store(candidate, self.meta_cfg.max_edits_per_update).apply_with_receipt([])
            reflection_count, status = 0, "insufficient_informative_pairs"
            save_checkpoint(directory / "candidate_skill.json", refined)
        else:
            refined, change, reflection_count, status = self._propose_from_comparisons(
                number, candidate, directory, comparisons, stats
            )
        return refined, {
            **change,
            "batch": number,
            "source_stage": "meta",
            "train_ids": [str(q["id"]) for q in questions],
            "evidence_parent_skill_hash": receipt["parent_skill_hash"],
            "evidence_candidate_skill_hash": receipt["candidate_skill_hash"],
            "proposal_parent_skill_hash": skill_sha256(candidate),
            "comparison_count": len(comparisons),
            "comparison_statistics": stats,
            "reflection_count": reflection_count,
            "optimizer_status": status,
            "validation_status": "pending_workflow_gate",
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
            if len(comparisons) < self.meta_cfg.min_informative_pairs:
                _, change = self.store(
                    parent, self.meta_cfg.max_edits_per_update
                ).apply_with_receipt([])
                gate = {
                    "accepted": False,
                    "reason": "skip_meta_update",
                    "skip_reason": "insufficient_informative_pairs",
                    "informative_pair_count": len(comparisons),
                    "required_informative_pair_count": self.meta_cfg.min_informative_pairs,
                }
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
                    "reflection_count": 0,
                    "gate": gate,
                    "active_skill": parent,
                    "active_skill_hash": skill_sha256(parent),
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
                        "active_skill_hash": skill_sha256(parent),
                    },
                )
                receipts.append(receipt)
                active = parent
                continue
            candidate, change, reflection_count, optimizer_status = self._propose_from_comparisons(
                number, parent, directory, comparisons, stats
            )
            if skill_sha256(parent) == skill_sha256(candidate):
                gate = {
                    "accepted": False,
                    "reason": "no_change",
                    "optimizer_status": optimizer_status,
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
                "reflection_count": reflection_count,
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
