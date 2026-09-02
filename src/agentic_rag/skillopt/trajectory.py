"""Deterministic, leakage-safe trajectory views for SkillOpt reflection.

The target Agent always writes one complete :class:`EpisodeResult`.  This
module projects that immutable episode into alternative reflection inputs; it
never changes or replays the target rollout.  Ground-truth annotations are
accepted only for training reflection and are never added to target Policy
messages.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from agentic_rag.agent.models import (
    EpisodeResult,
    Observation,
    ObservationOutcome,
    ObservationStatus,
    ResolvedExpandAction,
    StepRecord,
)


REFLECTION_SCHEMA_VERSION = "agentic-rag-skillopt-reflection-v3"
REFLECTION_MANIFEST_SCHEMA_VERSION = "agentic-rag-skillopt-reflection-manifest-v2"
_TOKEN_RE = re.compile(r"(?u)\b\w+\b|[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


class TrajectoryRepresentation(StrEnum):
    RAW = "raw"
    ORGANIZED = "organized"
    ORGANIZED_SUPPORT_LABELS = "organized_support_labels"
    PROGRESS_ABSTRACTED = "progress_abstracted"


TRAJECTORY_REPRESENTATIONS = tuple(item.value for item in TrajectoryRepresentation)


class ExecutionState(StrEnum):
    EXECUTED = "executed"
    NOT_EXECUTED = "not_executed"
    EXECUTION_FAILED = "execution_failed"


def normalize_trajectory_representation(
    value: str | TrajectoryRepresentation,
) -> TrajectoryRepresentation:
    try:
        return TrajectoryRepresentation(str(value))
    except ValueError as exc:
        raise ValueError(
            "trajectory_representation must be one of "
            f"{TRAJECTORY_REPRESENTATIONS}; found {value!r}"
        ) from exc


def build_training_reference(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return the one canonical hidden-reference payload used by all arms."""

    if item.get("answer") is None:
        raise ValueError("training reference requires an answer")
    return {
        "answer": str(item["answer"]),
        "supporting_facts": _normalize_gold_facts(item.get("supporting_facts")),
    }


def build_training_reference_text(item: Mapping[str, Any]) -> str:
    """Serialize the arm-independent training reference deterministically."""

    return json.dumps(
        build_training_reference(item),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_reflection_input(
    *,
    episode: EpisodeResult,
    item: Mapping[str, Any],
    skill_content: str,
    rollout_phase: str,
    rollout_split: str,
    trajectory_representation: str | TrajectoryRepresentation,
    evaluation: Mapping[str, Any] | BaseModel | None = None,
    target_system_prompt: str | None = None,
    effective_config: Mapping[str, Any] | None = None,
    sentence_provenance: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one reflection input and its deterministic audit manifest.

    ``supporting_facts`` and the reference answer are included only when
    ``rollout_phase == "train"``.  Passing them for validation/test is safe:
    they are deliberately ignored before either the rendered input or its
    manifest is constructed.
    """

    representation = normalize_trajectory_representation(
        trajectory_representation
    )
    if not skill_content.strip():
        raise ValueError("skill_content must not be blank")
    if rollout_phase not in {"train", "eval"}:
        raise ValueError(f"unsupported rollout phase: {rollout_phase}")

    training = rollout_phase == "train"
    hidden_reference = build_training_reference(item) if training else None
    gold_facts = hidden_reference["supporting_facts"] if training else []
    config = dict(effective_config or {})
    rendered: dict[str, Any] = {
        "schema_version": REFLECTION_SCHEMA_VERSION,
        "trajectory_representation": representation.value,
        "rollout_context": _rollout_context(
            episode=episode,
            item=item,
            skill_content=skill_content,
            rollout_phase=rollout_phase,
            rollout_split=rollout_split,
            target_system_prompt=target_system_prompt,
            effective_config=config,
        ),
        "episode_outcome": _episode_outcome(
            episode,
            evaluation,
            include_evidence_text=(
                representation is not TrajectoryRepresentation.PROGRESS_ABSTRACTED
            ),
        ),
        "hidden_reference": hidden_reference,
    }

    diagnostic_meta: dict[str, Any] = {}
    if representation is TrajectoryRepresentation.RAW:
        rendered["trajectory"] = {
            "raw_steps": [
                _render_raw_step(
                    step,
                    subsequent_policy_call_exists=(
                        index + 1 < len(episode.trajectory)
                    ),
                )
                for index, step in enumerate(episode.trajectory)
            ]
        }
        labels_included = False
    elif representation is TrajectoryRepresentation.ORGANIZED:
        rendered["trajectory"] = _render_organized(episode)
        labels_included = False
    else:
        diagnostics, diagnostic_meta = _support_diagnostics(
            episode.trajectory,
            gold_facts=gold_facts,
            sentence_provenance=(sentence_provenance if training else None),
        )
        labels_included = bool(training and gold_facts)
        if representation is TrajectoryRepresentation.PROGRESS_ABSTRACTED:
            rendered["trajectory"] = _render_progress_abstracted(
                episode,
                diagnostics=diagnostics,
                diagnostic_meta=diagnostic_meta,
            )
        else:
            assert (
                representation
                is TrajectoryRepresentation.ORGANIZED_SUPPORT_LABELS
            )
            organized = _render_organized(episode)
            organized["step_diagnostics"] = diagnostics
            organized["support_diagnostic_metadata"] = diagnostic_meta
            rendered["trajectory"] = organized

    payload = _canonical_json_bytes(rendered)
    provenance_for_manifest = sentence_provenance if labels_included else None
    manifest = {
        "schema_version": REFLECTION_MANIFEST_SCHEMA_VERSION,
        "trajectory_representation": representation.value,
        "episode_id": episode.episode_id,
        "rollout_phase": rollout_phase,
        "rollout_split": rollout_split,
        "source_episode_sha256": _sha256_object(
            episode.model_dump(mode="json")
        ),
        "skill_sha256": hashlib.sha256(
            skill_content.encode("utf-8")
        ).hexdigest(),
        "reflection_input_sha256": hashlib.sha256(payload).hexdigest(),
        "reflection_input_utf8_bytes": len(payload),
        "reflection_input_estimated_tokens": len(
            _TOKEN_RE.findall(payload.decode("utf-8"))
        ),
        "truncated": False,
        "hidden_reference_included": training,
        "support_labels_included": labels_included,
        "retrieved_evidence_text_included": (
            representation is not TrajectoryRepresentation.PROGRESS_ABSTRACTED
        ),
        "shared_decision_context_included": True,
        "supporting_fact_count": len(gold_facts) if training else None,
        "sentence_provenance_entry_count": (
            len(provenance_for_manifest) if provenance_for_manifest else 0
        ),
        "sentence_provenance_sha256": (
            _sha256_object(provenance_for_manifest)
            if provenance_for_manifest
            else None
        ),
        "unmatched_visible_sentence_count": (
            diagnostic_meta.get("unmatched_visible_sentence_count")
            if representation
            in {
                TrajectoryRepresentation.ORGANIZED_SUPPORT_LABELS,
                TrajectoryRepresentation.PROGRESS_ABSTRACTED,
            }
            else None
        ),
    }
    return rendered, manifest


def _rollout_context(
    *,
    episode: EpisodeResult,
    item: Mapping[str, Any],
    skill_content: str,
    rollout_phase: str,
    rollout_split: str,
    target_system_prompt: str | None,
    effective_config: Mapping[str, Any],
) -> dict[str, Any]:
    initial = episode.trajectory[0].state_before if episode.trajectory else None
    final = episode.final_state
    policy = effective_config.get("agent", {})
    policy = policy.get("policy", {}) if isinstance(policy, Mapping) else {}
    model = policy.get("model") if isinstance(policy, Mapping) else None
    skill_sha = hashlib.sha256(skill_content.encode("utf-8")).hexdigest()
    return {
        "dataset": str(item.get("source") or ""),
        "episode_id": episode.episode_id,
        "split": rollout_split,
        "rollout_phase": rollout_phase,
        "question": episode.query,
        "scope_id": episode.scope_id,
        "question_type": str(
            item.get("question_type") or item.get("task_type") or "qa"
        ),
        "target_model": str(model) if model is not None else None,
        "target_system_prompt": target_system_prompt,
        "skill": {
            "version": f"sha256:{skill_sha}",
            "sha256": skill_sha,
            "content": skill_content,
        },
        "budget": {
            "configured_steps": (
                initial.remaining_step_budget if initial is not None else None
            ),
            "configured_policy_attempts": (
                initial.remaining_policy_attempt_budget if initial is not None else None
            ),
            "configured_retrieval_tokens": (
                initial.remaining_retrieved_token_budget
                if initial is not None
                else None
            ),
            "used_steps": final.step if final is not None else None,
            "used_policy_attempts": (
                final.policy_attempts if final is not None else len(episode.trajectory)
            ),
            "used_retrieval_tokens": episode.usage.retrieved_tokens,
        },
    }


def _episode_outcome(
    episode: EpisodeResult,
    evaluation: Mapping[str, Any] | BaseModel | None,
    *,
    include_evidence_text: bool = True,
) -> dict[str, Any]:
    evaluation_value = _safe_evaluation(evaluation)
    return {
        "final_answer": episode.answer,
        "finish_evidence_refs": [
            item.model_dump(mode="json") for item in episode.evidence_refs
        ],
        "resolved_finish_evidence": [
            item.model_dump(
                mode="json",
                include=(
                    None if include_evidence_text else {
                        "ref", "document_id", "parent_chunk_id",
                        "contained_sentence_ids",
                    }
                ),
            )
            for item in episode.resolved_evidence
        ],
        "termination_reason": episode.termination_reason.value,
        "error_code": episode.error_code,
        "error_message": episode.error_message,
        "evaluation": evaluation_value,
    }


def _render_raw_step(
    step: StepRecord,
    *,
    subsequent_policy_call_exists: bool,
) -> dict[str, Any]:
    observation = step.observation
    return {
        "step": step.step,
        "policy_attempt": step.policy_attempt,
        "assessment": (
            step.decision.assessment.model_dump(mode="json")
            if step.decision is not None
            else None
        ),
        "submitted_action": (
            step.decision.action.model_dump(mode="json")
            if step.decision is not None
            else None
        ),
        "resolved_action": (
            step.resolved_decision.action.model_dump(mode="json")
            if step.resolved_decision is not None
            else None
        ),
        "validator": {
            "status": step.validation_status.value,
            "error": step.validation_error,
        },
        # Exact cumulative state shown to the Policy before it chose this
        # action.  This is distinct from the post-tool state and is essential
        # for diagnosing repeated actions without hindsight leakage.
        "policy_input_state": (
            step.policy_view.model_dump(mode="json")
            if step.policy_view is not None
            else None
        ),
        "interface_context": _interface_context(step),
        "decision_context": _decision_context(step),
        # This is the complete router/controller audit record.  It can include
        # ranking scores and path metadata that were not rendered verbatim in
        # the next Policy prompt, so it is explicitly separated from the
        # policy-visible delta below.
        "tool_raw": (
            observation.model_dump(mode="json")
            if observation is not None
            else None
        ),
        "policy_visible_delta": _policy_visible_delta(
            step,
            subsequent_policy_call_exists=subsequent_policy_call_exists,
        ),
        "remaining_budget": {
            "before": _budget(step.state_before),
            "after": _budget(step.state_after),
        },
    }


def _policy_visible_delta(
    step: StepRecord,
    *,
    subsequent_policy_call_exists: bool,
) -> dict[str, Any]:
    before = step.state_before
    after = step.state_after
    before_visible = _visible_ids(before)
    after_visible = _visible_ids(after)
    result_units = _result_units(step)
    new_ids = sorted(after_visible - before_visible)
    repeated_ids = sorted(_returned_ids(step) & before_visible)
    return {
        "exposure_timing": "available in the subsequent Policy input",
        "subsequent_policy_call_exists": subsequent_policy_call_exists,
        "new_visible_units": [
            _unit_for_output(
                stable_id,
                result_units.get(stable_id),
                state=after,
            )
            for stable_id in new_ids
        ],
        "repeated_visible_refs": [
            _typed_ref(after, stable_id) for stable_id in repeated_ids
        ],
        "newly_eligible_sentence_refs": [
            _typed_ref(after, stable_id)
            for stable_id in sorted(
                after.eligible_sentence_ids - before.eligible_sentence_ids
            )
        ],
        "newly_read_chunk_refs": [
            _typed_ref(after, stable_id)
            for stable_id in sorted(after.read_chunk_ids - before.read_chunk_ids)
        ],
        "removed_visible_refs": [],
    }


def _render_organized(episode: EpisodeResult) -> dict[str, Any]:
    ledger: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for step in episode.trajectory:
        ledger.append(_action_ledger_row(step, seen_sources=seen_sources))
        if _execution(step)["progress_evaluable"]:
            seen_sources.update(_source_keys(step))
    return {
        "problem_index": _problem_index(ledger),
        "action_ledger": ledger,
        "retrieved_units": _retrieved_units(episode),
    }


def _render_progress_abstracted(
    episode: EpisodeResult,
    *,
    diagnostics: Sequence[Mapping[str, Any]],
    diagnostic_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Render step progress without any retrieved result text.

    The retrospective bridge records attach a later EXPAND back to the step
    that first made its parent unit visible.  They describe direct observed
    progress only and deliberately make no causal claim about final success.
    """

    if len(diagnostics) != len(episode.trajectory):
        raise ValueError("one progress diagnostic is required for every step")
    later_bridge_uses, unattributed_bridge_uses = _retrospective_bridge_uses(
        episode.trajectory,
        diagnostics,
    )
    abstract_steps: list[dict[str, Any]] = []
    for step, diagnostic in zip(episode.trajectory, diagnostics, strict=True):
        if int(diagnostic["step"]) != step.step:
            raise ValueError("progress diagnostics are not aligned to trajectory")
        execution = diagnostic["execution"]
        abstract_steps.append(
            {
                "step": step.step,
                "policy_attempt": step.policy_attempt,
                "submitted_action": (
                    step.decision.action.model_dump(mode="json")
                    if step.decision is not None
                    else None
                ),
                "resolved_action": (
                    step.resolved_decision.action.model_dump(mode="json")
                    if step.resolved_decision is not None
                    else None
                ),
                "interface_context": _interface_context(step),
                "decision_context": _decision_context(step),
                "execution": execution,
                "error": (
                    {
                        "code": (
                            step.observation.error_code
                            if step.observation is not None
                            else "policy_error"
                        ),
                        "message": (
                            step.observation.message
                            if step.observation is not None
                            else step.validation_error
                        ),
                    }
                    if not execution["progress_evaluable"]
                    else None
                ),
                "information_progress": diagnostic["information_progress"],
                "supporting_fact_progress": diagnostic[
                    "supporting_fact_progress"
                ],
                "direct_expand_progress": diagnostic[
                    "direct_expand_utility"
                ],
                "later_expand_uses_of_acquired_information": later_bridge_uses.get(
                    step.step, []
                ),
                "remaining_budget": _budget(step.state_after),
            }
        )
    return {
        "abstract_steps": abstract_steps,
        "support_diagnostic_metadata": dict(diagnostic_meta),
        "unattributed_later_expand_uses": unattributed_bridge_uses,
        "retrieval_text_included": False,
        "bridge_attribution_scope": (
            "A later EXPAND is linked to the step that first exposed its "
            "parent unit. Progress covers only that EXPAND's direct result "
            "and is retrospective, not causal."
        ),
    }


def _retrospective_bridge_uses(
    trajectory: Sequence[StepRecord],
    diagnostics: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    first_visible_step: dict[str, int] = {}
    for step in trajectory:
        for stable_id in sorted(
            _visible_ids(step.state_after) - _visible_ids(step.state_before)
        ):
            first_visible_step.setdefault(stable_id, step.step)

    attributed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    unattributed: list[dict[str, Any]] = []
    for step, diagnostic in zip(trajectory, diagnostics, strict=True):
        utility = diagnostic.get("direct_expand_utility")
        if not isinstance(utility, Mapping):
            continue
        parent_id = utility.get("parent_stable_id")
        if not isinstance(parent_id, str) or not parent_id:
            continue
        information = diagnostic.get("information_progress")
        information = information if isinstance(information, Mapping) else None
        policy_visible_gold = utility.get(
            "new_policy_visible_gold_fact_ids"
        )
        policy_eligible_gold = utility.get(
            "new_policy_eligible_gold_fact_ids"
        )
        gt_progress_evaluable = isinstance(policy_visible_gold, list) and isinstance(
            policy_eligible_gold, list
        )
        new_unit_count = (
            int(information.get("new_unit_count", 0))
            if information is not None
            else None
        )
        visibility_upgrades = (
            information.get("visibility_upgrades", [])
            if information is not None
            else None
        )
        event = {
            "evidence_ref": utility.get("parent_ref"),
            "evidence_stable_id": parent_id,
            "expand_step": step.step,
            "relation": utility.get("relation"),
            "execution": diagnostic.get("execution"),
            "new_unit_count": new_unit_count,
            "new_source_count": (
                int(information.get("new_source_count", 0))
                if information is not None
                else None
            ),
            "visibility_upgrades": visibility_upgrades,
            "produced_new_information": (
                bool(new_unit_count or visibility_upgrades)
                if information is not None
                else None
            ),
            "new_policy_visible_gold_fact_ids": policy_visible_gold,
            "new_policy_eligible_gold_fact_ids": policy_eligible_gold,
            "improved_supporting_fact_progress": (
                bool(policy_visible_gold or policy_eligible_gold)
                if gt_progress_evaluable
                else None
            ),
            "attribution_scope": (
                "direct EXPAND observation only; retrospective, not causal"
            ),
        }
        acquisition_step = first_visible_step.get(parent_id)
        if acquisition_step is None:
            unattributed.append(event)
        else:
            attributed[acquisition_step].append(event)
    return dict(attributed), unattributed


def _action_ledger_row(
    step: StepRecord,
    *,
    seen_sources: set[str],
) -> dict[str, Any]:
    execution = _execution(step)
    returned = sorted(_returned_ids(step)) if execution["progress_evaluable"] else []
    before_visible = _visible_ids(step.state_before)
    after_visible = _visible_ids(step.state_after)
    new_ids = sorted(after_visible - before_visible) if returned else []
    duplicate_ids = sorted(set(returned) & before_visible)
    returned_sources = _source_keys(step) if execution["progress_evaluable"] else set()
    new_sources = returned_sources - seen_sources
    return {
        "step": step.step,
        "policy_attempt": step.policy_attempt,
        "submitted_action": (
            step.decision.action.model_dump(mode="json")
            if step.decision is not None
            else None
        ),
        "resolved_action": (
            step.resolved_decision.action.model_dump(mode="json")
            if step.resolved_decision is not None
            else None
        ),
        "interface_context": _interface_context(step),
        "decision_context": _decision_context(step),
        "execution": execution,
        "error": (
            {
                "code": (
                    step.observation.error_code
                    if step.observation is not None
                    else "policy_error"
                ),
                "message": (
                    step.observation.message
                    if step.observation is not None
                    else step.validation_error
                ),
            }
            if not execution["progress_evaluable"]
            else None
        ),
        "top_level_result_count": _top_level_result_count(step),
        "returned_unit_count": len(returned),
        "returned_refs": [
            _typed_ref(step.state_after, stable_id) for stable_id in returned
        ],
        "new_unit_count": len(new_ids),
        "duplicate_unit_count": len(duplicate_ids),
        "returned_source_count": len(returned_sources),
        "new_source_count": len(new_sources),
        "empty_result": bool(
            execution["progress_evaluable"]
            and _action_type(step) in {"SEARCH", "EXPAND", "READ"}
            and not returned
        ),
        "remaining_budget": _budget(step.state_after),
    }


def _decision_context(step: StepRecord) -> dict[str, Any]:
    """Common decision-time facts, without retrieved text or supported_facts.

    Prefer the frozen references actually supplied for this call. Legacy
    records fall back to the saved Policy view, then to state_before only;
    neither references nor missing information are inferred from later steps.
    """
    before = step.state_before
    references: list[dict[str, Any]] = []
    if step.context_reference_map is not None:
        source = "context_reference_map"
        for ref, node in sorted(step.context_reference_map.typed_refs.items()):
            references.append({
                "ref": ref,
                "stable_id": node.stable_id,
                "node_type": node.node_type,
                "has_been_read": (
                    node.stable_id in before.read_chunk_ids
                    if node.node_type == "CHUNK" else None
                ),
                "can_read": node.can_read,
                "can_use_as_evidence": node.can_use_as_evidence,
            })
    elif step.policy_view is not None:
        source = "policy_input_state"
        ref_to_id = {
            ref: stable_id
            for stable_id, ref in before.reference_registry.stable_id_to_ref.items()
        }
        for item in sorted(
            step.policy_view.policy_state.semantic_memory, key=lambda item: item.ref
        ):
            is_chunk = item.node_type == "CHUNK"
            read = bool(getattr(item, "has_been_read", False))
            references.append({
                "ref": item.ref,
                "stable_id": ref_to_id.get(item.ref),
                "node_type": item.node_type,
                "has_been_read": read if is_chunk else None,
                "can_read": is_chunk and not read,
                "can_use_as_evidence": item.node_type == "SENTENCE" or (is_chunk and read),
            })
    else:
        source = "state_before"
        for stable_id in sorted(_visible_ids(before)):
            ref = _typed_ref(before, stable_id)
            node_type = _node_type_from_ref(before, stable_id)
            is_chunk = stable_id in before.visible_chunk_ids
            read = stable_id in before.read_chunk_ids
            references.append({
                "ref": ref,
                "stable_id": stable_id,
                "node_type": node_type,
                "has_been_read": read if is_chunk else None,
                "can_read": is_chunk and not read,
                "can_use_as_evidence": (
                    stable_id in before.eligible_sentence_ids or (is_chunk and read)
                ),
            })
    return {
        "missing_information": (
            list(step.decision.assessment.missing_information)
            if step.decision is not None else None
        ),
        "reference_source": source,
        "visible_references": references,
        "available_action_space": _interface_context(step)["available_action_space"],
        "remaining_budget": {
            "before": _budget(before),
            "after": _budget(step.state_after),
        },
    }


def _interface_context(step: StepRecord) -> dict[str, Any]:
    """Render Controller constraints that governed this Policy call."""

    action_space = (
        step.available_action_space.model_dump(mode="json")
        if step.available_action_space is not None
        else None
    )
    mode = action_space.get("mode") if isinstance(action_space, Mapping) else None
    return {
        "mode": mode,
        "available_action_space": action_space,
        "decision_schema_sha256": step.decision_schema_sha256,
    }


def _problem_index(ledger: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ledger:
        execution = row["execution"]
        error = row.get("error")
        category: str | None = None
        code = ""
        message = ""
        if execution["state"] == ExecutionState.NOT_EXECUTED.value:
            code = str((error or {}).get("code") or execution["outcome"])
            category = (
                "invalid_reference"
                if "reference" in code
                else (
                    "duplicate_action"
                    if execution["outcome"] == ObservationOutcome.DUPLICATE_ACTION.value
                    else "validator_rejection"
                )
            )
            message = str((error or {}).get("message") or "")
        elif execution["state"] == ExecutionState.EXECUTION_FAILED.value:
            code = str((error or {}).get("code") or execution["outcome"])
            category = (
                "budget_rejection"
                if execution["outcome"] == ObservationOutcome.BUDGET_REJECTED.value
                else "tool_error"
            )
            message = str((error or {}).get("message") or "")
        elif row.get("empty_result"):
            category = "empty_result"
            code = "empty_result"
        elif (
            int(row.get("new_unit_count", 0)) == 0
            and int(row.get("duplicate_unit_count", 0)) > 0
        ):
            category = "duplicate_results_only"
            code = "no_new_units"
        if category is None:
            continue
        key = (category, code)
        entry = grouped.setdefault(
            key,
            {
                "category": category,
                "code": code,
                "count": 0,
                "steps": [],
                "messages": [],
            },
        )
        entry["count"] += 1
        entry["steps"].append(int(row["step"]))
        if message and message not in entry["messages"]:
            entry["messages"].append(message)
    return [grouped[key] for key in sorted(grouped)]


def _retrieved_units(episode: EpisodeResult) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    selected = {item.id for item in episode.evidence_refs}
    for step in episode.trajectory:
        if not _execution(step)["progress_evaluable"]:
            continue
        action_type = _action_type(step)
        if action_type == "FINISH":
            continue
        units = _result_units(step)
        visible_after = _visible_ids(step.state_after)
        for stable_id in sorted(set(units) & visible_after):
            unit = units[stable_id]
            existing = records.get(stable_id)
            if existing is None:
                existing = _unit_for_output(stable_id, unit, state=step.state_after)
                existing.update(
                    {
                        "first_seen_step": step.step,
                        "acquisition_steps": [],
                        "acquisition_paths": [],
                        "eligible_since_step": None,
                        "read_since_step": None,
                        "selected_by_finish": stable_id in selected,
                        "final_evidence": stable_id in selected,
                    }
                )
                records[stable_id] = existing
            else:
                _merge_unit_content(existing, unit, step.state_after)
            existing["acquisition_steps"].append(step.step)
            existing["acquisition_paths"].append(
                _acquisition_path(step, stable_id, unit)
            )
            if (
                stable_id in step.state_after.eligible_sentence_ids
                and existing["eligible_since_step"] is None
            ):
                existing["eligible_since_step"] = step.step
            if (
                stable_id in step.state_after.read_chunk_ids
                and existing["read_since_step"] is None
            ):
                existing["read_since_step"] = step.step
    return sorted(
        records.values(), key=lambda unit: (unit["first_seen_step"], unit["stable_id"])
    )


def _acquisition_path(
    step: StepRecord,
    stable_id: str,
    unit: Mapping[str, Any],
) -> dict[str, Any]:
    action_type = _action_type(step)
    mechanism = {
        "SEARCH": "DIRECT_SEARCH",
        "EXPAND": "EXPAND",
        "READ": "READ",
    }.get(action_type, action_type or "UNKNOWN")
    resolved_action = (
        step.resolved_decision.action if step.resolved_decision is not None else None
    )
    parent_id = (
        resolved_action.source_id
        if isinstance(resolved_action, ResolvedExpandAction)
        else (
            resolved_action.chunk_id
            if action_type == "READ" and resolved_action is not None
            else None
        )
    )
    return {
        "step": step.step,
        "mechanism": mechanism,
        "action_type": action_type,
        "parent_ref": (
            _typed_ref(step.state_before, parent_id) if parent_id is not None else None
        ),
        "parent_stable_id": parent_id,
        "relation": (
            resolved_action.kind.value
            if isinstance(resolved_action, ResolvedExpandAction)
            else None
        ),
        "newly_visible": stable_id
        not in _visible_ids(step.state_before),
        "became_eligible": stable_id
        in (
            step.state_after.eligible_sentence_ids
            - step.state_before.eligible_sentence_ids
        ),
        "became_read": stable_id
        in (step.state_after.read_chunk_ids - step.state_before.read_chunk_ids),
        "result_paths": unit.get("paths", []),
    }


def _support_diagnostics(
    trajectory: Sequence[StepRecord],
    *,
    gold_facts: Sequence[Mapping[str, Any]],
    sentence_provenance: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not gold_facts:
        seen_sources: set[str] = set()
        diagnostics: list[dict[str, Any]] = []
        for index, step in enumerate(trajectory):
            execution = _execution(step)
            diagnostics.append(
                {
                    "step": step.step,
                    "next_policy_saw_result": index + 1 < len(trajectory),
                    "execution": execution,
                    "information_progress": (
                        _information_progress(step, seen_sources=seen_sources)
                        if execution["progress_evaluable"]
                        else None
                    ),
                    "supporting_fact_progress": None,
                    "direct_expand_utility": _expand_utility_unlabelled(
                        step,
                        seen_sources=seen_sources,
                    ),
                }
            )
            if execution["progress_evaluable"]:
                seen_sources.update(_source_keys(step))
        return (
            diagnostics,
            {
                "available": False,
                "reason": "training supporting_facts were not supplied",
                "matching_priority": [
                    "sentence_provenance(title, original_sentence_id)",
                    "legacy-only unique exact normalized supporting-fact text",
                ],
                "fuzzy_matching_used": False,
                "unmatched_visible_sentence_count": None,
            },
        )

    facts_by_key = {
        (_normalize_title(item["title"]), int(item["sentence_id"])): str(
            item["fact_id"]
        )
        for item in gold_facts
    }
    facts_by_text: dict[str, list[str]] = defaultdict(list)
    for item in gold_facts:
        facts_by_text[_normalize_text(item["text"])].append(str(item["fact_id"]))

    all_units: dict[str, dict[str, Any]] = {}
    for step in trajectory:
        for stable_id, unit in _result_units(step).items():
            current = all_units.get(stable_id)
            if current is None or _unit_quality(unit) > _unit_quality(current):
                all_units[stable_id] = dict(unit)

    stable_matches: dict[str, list[str]] = {}
    match_basis: dict[str, str] = {}
    visible_sentence_ids = sorted(
        {
            sentence_id
            for step in trajectory
            for sentence_id in step.state_after.visible_sentence_ids
        }
    )
    strict_provenance = sentence_provenance is not None
    provenance = sentence_provenance or {}
    unresolved_sentence_ids: list[str] = []
    mapped_non_gold_ids: list[str] = []
    for stable_id in visible_sentence_ids:
        if strict_provenance:
            mapped = provenance.get(stable_id)
            if not isinstance(mapped, Mapping):
                unresolved_sentence_ids.append(stable_id)
                continue
            source_pos = mapped.get(
                "original_sentence_id", mapped.get("sentence_id")
            )
            title = mapped.get("title", mapped.get("original_title"))
            if (
                not isinstance(source_pos, int)
                or isinstance(source_pos, bool)
                or not isinstance(title, str)
                or not title.strip()
            ):
                unresolved_sentence_ids.append(stable_id)
                continue
            fact_id = facts_by_key.get((_normalize_title(title), source_pos))
            if fact_id is not None:
                stable_matches[stable_id] = [fact_id]
                match_basis[stable_id] = "title_and_original_sentence_id"
            else:
                mapped_non_gold_ids.append(stable_id)
            # Supplying provenance selects strict mode: never silently fall
            # back to text matching for an unmapped or non-gold sentence.
            continue

        # Backward compatibility for legacy substrates that have no stable
        # sentence provenance at all.  This mode is explicitly audited below.
        unit = all_units.get(stable_id, {})
        text_key = _normalize_text(unit.get("text"))
        exact = facts_by_text.get(text_key, []) if text_key else []
        if len(exact) == 1:
            stable_matches[stable_id] = list(exact)
            match_basis[stable_id] = "legacy_unique_exact_normalized_text"

    diagnostics: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for index, step in enumerate(trajectory):
        next_policy_saw_result = index + 1 < len(trajectory)
        execution = _execution(step)
        if not execution["progress_evaluable"]:
            diagnostics.append(
                {
                    "step": step.step,
                    "next_policy_saw_result": next_policy_saw_result,
                    "execution": execution,
                    "information_progress": None,
                    "supporting_fact_progress": None,
                    "direct_expand_utility": _expand_utility_na(step),
                }
            )
            continue

        visible_before = _facts_for_sentences(
            step.state_before.visible_sentence_ids, stable_matches
        )
        visible_after = _facts_for_sentences(
            step.state_after.visible_sentence_ids, stable_matches
        )
        eligible_before = _facts_for_sentences(
            step.state_before.eligible_sentence_ids, stable_matches
        )
        eligible_after = _facts_for_sentences(
            step.state_after.eligible_sentence_ids, stable_matches
        )
        new_environment_visible = visible_after - visible_before
        new_environment_eligible = eligible_after - eligible_before
        # A terminal retrieval can mutate state_after, but no Target Policy
        # call consumed that observation.  Do not credit it as agent progress.
        policy_visible = visible_after if next_policy_saw_result else visible_before
        policy_eligible = (
            eligible_after if next_policy_saw_result else eligible_before
        )
        new_policy_visible = policy_visible - visible_before
        new_policy_eligible = policy_eligible - eligible_before
        support = {
            "provenance_complete": not unresolved_sentence_ids,
            "environment_visible": {
                "new_gold_fact_ids": sorted(new_environment_visible),
                "coverage": _coverage(visible_after, len(gold_facts)),
            },
            "environment_eligible": {
                "new_gold_fact_ids": sorted(new_environment_eligible),
                "coverage": _coverage(eligible_after, len(gold_facts)),
            },
            "policy_visible": {
                "new_gold_fact_ids": sorted(new_policy_visible),
                "coverage": _coverage(policy_visible, len(gold_facts)),
            },
            "policy_eligible": {
                "new_gold_fact_ids": sorted(new_policy_eligible),
                "coverage": _coverage(policy_eligible, len(gold_facts)),
            },
        }
        diagnostics.append(
            {
                "step": step.step,
                "next_policy_saw_result": next_policy_saw_result,
                "execution": execution,
                "information_progress": _information_progress(
                    step,
                    seen_sources=seen_sources,
                ),
                "supporting_fact_progress": support,
                "direct_expand_utility": _expand_utility(
                    step,
                    stable_matches=stable_matches,
                    new_environment_visible_facts=new_environment_visible,
                    new_environment_eligible_facts=new_environment_eligible,
                    new_policy_visible_facts=new_policy_visible,
                    new_policy_eligible_facts=new_policy_eligible,
                    next_policy_saw_result=next_policy_saw_result,
                    seen_sources=seen_sources,
                ),
            }
        )
        seen_sources.update(_source_keys(step))

    return (
        diagnostics,
        {
            "available": True,
            "gold_fact_count": len(gold_facts),
            "matched_stable_sentence_count": len(stable_matches),
            "visible_sentence_count": len(visible_sentence_ids),
            "unmatched_visible_sentence_count": len(unresolved_sentence_ids),
            "unmatched_visible_sentence_ids": unresolved_sentence_ids,
            "mapped_non_gold_sentence_count": len(mapped_non_gold_ids),
            "provenance_mode": (
                "strict_title_and_original_sentence_id"
                if strict_provenance
                else "legacy_exact_text_fallback"
            ),
            "provenance_complete": not unresolved_sentence_ids,
            "match_basis_counts": {
                basis: list(match_basis.values()).count(basis)
                for basis in sorted(set(match_basis.values()))
            },
            "matching_priority": [
                "sentence_provenance(title, original_sentence_id)",
                "legacy-only unique exact normalized supporting-fact text",
            ],
            "fuzzy_matching_used": False,
            "environment_visible_means": (
                "present in state_after even if no later Policy call consumed it"
            ),
            "policy_visible_means": (
                "present in a subsequent Target Policy input; previews count"
            ),
            "eligible_means": (
                "complete Sentence evidence or a Sentence exposed by READ"
            ),
        },
    )


def _information_progress(
    step: StepRecord,
    *,
    seen_sources: set[str],
) -> dict[str, Any]:
    before_visible = _visible_ids(step.state_before)
    after_visible = _visible_ids(step.state_after)
    returned = _returned_ids(step)
    new_ids = after_visible - before_visible
    duplicate_ids = returned & before_visible
    returned_sources = _source_keys(step)
    new_sources = returned_sources - seen_sources
    upgrades: list[dict[str, Any]] = []
    for stable_id in sorted(
        step.state_after.eligible_sentence_ids
        - step.state_before.eligible_sentence_ids
    ):
        if stable_id in step.state_before.visible_sentence_ids:
            upgrades.append(
                {
                    "ref": _typed_ref(step.state_after, stable_id),
                    "stable_id": stable_id,
                    "type": "sentence_preview_to_eligible",
                }
            )
    for stable_id in sorted(
        step.state_after.read_chunk_ids - step.state_before.read_chunk_ids
    ):
        if stable_id in step.state_before.visible_chunk_ids:
            upgrades.append(
                {
                    "ref": _typed_ref(step.state_after, stable_id),
                    "stable_id": stable_id,
                    "type": "chunk_handle_to_read",
                }
            )
    return {
        "top_level_result_count": _top_level_result_count(step),
        "returned_unit_count": len(returned),
        "new_unit_count": len(new_ids),
        "duplicate_unit_count": len(duplicate_ids),
        "returned_source_count": len(returned_sources),
        "new_source_count": len(new_sources),
        "new_refs": [
            _typed_ref(step.state_after, stable_id) for stable_id in sorted(new_ids)
        ],
        "duplicate_refs": [
            _typed_ref(step.state_after, stable_id)
            for stable_id in sorted(duplicate_ids)
        ],
        "visibility_upgrades": upgrades,
    }


def _expand_utility(
    step: StepRecord,
    *,
    stable_matches: Mapping[str, Sequence[str]],
    new_environment_visible_facts: set[str],
    new_environment_eligible_facts: set[str],
    new_policy_visible_facts: set[str],
    new_policy_eligible_facts: set[str],
    next_policy_saw_result: bool,
    seen_sources: set[str],
) -> dict[str, Any] | None:
    if _action_type(step) != "EXPAND":
        return None
    direct_children = _top_level_result_ids(step)
    returned_facts = _facts_for_sentences(
        _returned_ids(step), stable_matches
    )
    resolved = (
        step.resolved_decision.action if step.resolved_decision is not None else None
    )
    parent_id = resolved.source_id if isinstance(resolved, ResolvedExpandAction) else None
    return {
        "evaluable": True,
        "parent_ref": (
            _typed_ref(step.state_before, parent_id) if parent_id is not None else None
        ),
        "parent_stable_id": parent_id,
        "relation": (
            resolved.kind.value if isinstance(resolved, ResolvedExpandAction) else None
        ),
        "direct_child_count": len(direct_children),
        "direct_new_child_count": len(
            direct_children - _visible_ids(step.state_before)
        ),
        "direct_new_source_count": len(_source_keys(step) - seen_sources),
        "gold_fact_ids_returned": sorted(returned_facts),
        "next_policy_saw_result": next_policy_saw_result,
        "new_environment_visible_gold_fact_ids": sorted(
            new_environment_visible_facts
        ),
        "new_environment_eligible_gold_fact_ids": sorted(
            new_environment_eligible_facts
        ),
        "new_policy_visible_gold_fact_ids": sorted(new_policy_visible_facts),
        "new_policy_eligible_gold_fact_ids": sorted(new_policy_eligible_facts),
        "attribution_scope": "direct EXPAND observation only; not causal",
    }


def _expand_utility_na(step: StepRecord) -> dict[str, Any] | None:
    if _action_type(step) != "EXPAND":
        return None
    return {
        "evaluable": False,
        "reason": _execution(step)["state"],
        "direct_child_count": None,
        "direct_new_child_count": None,
        "direct_new_source_count": None,
        "gold_fact_ids_returned": None,
        "next_policy_saw_result": None,
        "new_environment_visible_gold_fact_ids": None,
        "new_environment_eligible_gold_fact_ids": None,
        "new_policy_visible_gold_fact_ids": None,
        "new_policy_eligible_gold_fact_ids": None,
        "attribution_scope": "direct EXPAND observation only; not causal",
    }


def _expand_utility_unlabelled(
    step: StepRecord,
    *,
    seen_sources: set[str],
) -> dict[str, Any] | None:
    if _action_type(step) != "EXPAND":
        return None
    execution = _execution(step)
    if not execution["progress_evaluable"]:
        return _expand_utility_na(step)
    direct_children = _top_level_result_ids(step)
    resolved = (
        step.resolved_decision.action if step.resolved_decision is not None else None
    )
    parent_id = resolved.source_id if isinstance(resolved, ResolvedExpandAction) else None
    return {
        "evaluable": True,
        "parent_ref": (
            _typed_ref(step.state_before, parent_id) if parent_id is not None else None
        ),
        "parent_stable_id": parent_id,
        "relation": (
            resolved.kind.value if isinstance(resolved, ResolvedExpandAction) else None
        ),
        "direct_child_count": len(direct_children),
        "direct_new_child_count": len(
            direct_children - _visible_ids(step.state_before)
        ),
        "direct_new_source_count": len(_source_keys(step) - seen_sources),
        "gold_fact_ids_returned": None,
        "next_policy_saw_result": None,
        "new_environment_visible_gold_fact_ids": None,
        "new_environment_eligible_gold_fact_ids": None,
        "new_policy_visible_gold_fact_ids": None,
        "new_policy_eligible_gold_fact_ids": None,
        "attribution_scope": "direct EXPAND observation only; GT unavailable",
    }


def _execution(step: StepRecord) -> dict[str, Any]:
    observation = step.observation
    if observation is None:
        return {
            "state": ExecutionState.NOT_EXECUTED.value,
            "outcome": "policy_error",
            "progress_evaluable": False,
        }
    if observation.status in {
        ObservationStatus.INVALID_ACTION,
        ObservationStatus.DUPLICATE_ACTION,
    }:
        return {
            "state": ExecutionState.NOT_EXECUTED.value,
            "outcome": _observation_outcome(observation),
            "progress_evaluable": False,
        }
    if observation.status is ObservationStatus.ERROR:
        return {
            "state": ExecutionState.EXECUTION_FAILED.value,
            "outcome": _observation_outcome(observation),
            "progress_evaluable": False,
        }
    return {
        "state": ExecutionState.EXECUTED.value,
        "outcome": _observation_outcome(observation),
        "progress_evaluable": True,
    }


def _observation_outcome(observation: Observation) -> str:
    if observation.status is ObservationStatus.DUPLICATE_ACTION:
        return ObservationOutcome.DUPLICATE_ACTION.value
    if observation.status is ObservationStatus.INVALID_ACTION:
        return ObservationOutcome.INVALID_ACTION.value
    if observation.status is ObservationStatus.ERROR:
        if observation.error_code and "budget" in observation.error_code:
            return ObservationOutcome.BUDGET_REJECTED.value
        return ObservationOutcome.TOOL_ERROR.value
    if observation.results:
        return ObservationOutcome.SUCCESS.value
    return ObservationOutcome.EMPTY.value


def _result_units(step: StepRecord) -> dict[str, dict[str, Any]]:
    if step.observation is None or _action_type(step) == "FINISH":
        return {}
    units: dict[str, dict[str, Any]] = {}
    for result in step.observation.results:
        _collect_result_units(result, units=units)
    visible = _visible_ids(step.state_after)
    return {key: value for key, value in units.items() if key in visible}


def _collect_result_units(
    value: Any,
    *,
    units: dict[str, dict[str, Any]],
    inherited_navigation: bool = False,
    inherited_chunk_id: str | None = None,
    inherited_document_id: str | None = None,
    inherited_title: str | None = None,
    field_name: str | None = None,
) -> None:
    if isinstance(value, list):
        navigation = inherited_navigation or field_name in {
            "previews",
            "sentence_previews",
            "trigger_sentences",
            "bridge_previews",
        }
        for item in value:
            _collect_result_units(
                item,
                units=units,
                inherited_navigation=navigation,
                inherited_chunk_id=inherited_chunk_id,
                inherited_document_id=inherited_document_id,
                inherited_title=inherited_title,
                field_name=field_name,
            )
        return
    if not isinstance(value, Mapping):
        return

    navigation = inherited_navigation or bool(value.get("navigation_only", False))
    title_value = value.get("title")
    title = title_value if isinstance(title_value, str) else inherited_title
    document_id_value = value.get("document_id", value.get("doc_id"))
    document_id = (
        document_id_value
        if isinstance(document_id_value, str)
        else inherited_document_id
    )
    paths = value.get("paths") if isinstance(value.get("paths"), list) else []

    entity_id = value.get("entity_id", value.get("target_entity_id"))
    if isinstance(entity_id, str):
        _merge_extracted_unit(
            units,
            entity_id,
            {
                "node_type": "ENTITY",
                "document_id": document_id,
                "canonical_name": value.get(
                    "canonical_name", value.get("target_canonical_name")
                ),
                "entity_type": value.get("entity_type"),
                "paths": paths,
            },
        )

    chunk_id = value.get(
        "parent_chunk_id",
        value.get("chunk_id", value.get("bridge_chunk_id")),
    )
    if not isinstance(chunk_id, str):
        chunk_id = inherited_chunk_id
    if isinstance(chunk_id, str):
        chunk_text = (
            value.get("text")
            if isinstance(value.get("sentences"), list)
            and isinstance(value.get("text"), str)
            else None
        )
        _merge_extracted_unit(
            units,
            chunk_id,
            {
                "node_type": "CHUNK",
                "document_id": document_id,
                "title": title,
                "text": chunk_text,
                "chunk_position": value.get("chunk_pos"),
                "paths": paths,
            },
        )

    sentence_id = value.get("sentence_id", value.get("bridge_sentence_id"))
    if isinstance(sentence_id, str):
        text = value.get("text", value.get("bridge_text"))
        _merge_extracted_unit(
            units,
            sentence_id,
            {
                "node_type": "SENTENCE",
                "document_id": document_id,
                "title": title,
                "text": text if isinstance(text, str) else None,
                "parent_chunk_id": chunk_id,
                "presentation": "preview" if navigation else "complete",
                "paths": paths,
            },
        )

    for key, child in value.items():
        if key in {
            "entity_id",
            "target_entity_id",
            "chunk_id",
            "parent_chunk_id",
            "bridge_chunk_id",
            "sentence_id",
            "bridge_sentence_id",
            "paths",
        }:
            continue
        _collect_result_units(
            child,
            units=units,
            inherited_navigation=navigation,
            inherited_chunk_id=chunk_id,
            inherited_document_id=document_id,
            inherited_title=title,
            field_name=str(key),
        )


def _merge_extracted_unit(
    units: dict[str, dict[str, Any]],
    stable_id: str,
    candidate: dict[str, Any],
) -> None:
    existing = units.get(stable_id)
    if existing is None:
        units[stable_id] = candidate
        return
    if _unit_quality(candidate) > _unit_quality(existing):
        paths = [*existing.get("paths", []), *candidate.get("paths", [])]
        units[stable_id] = {**existing, **candidate, "paths": paths}
    else:
        existing["paths"] = [
            *existing.get("paths", []),
            *candidate.get("paths", []),
        ]
        for key, item in candidate.items():
            if existing.get(key) is None and item is not None:
                existing[key] = item


def _unit_quality(unit: Mapping[str, Any]) -> tuple[int, int]:
    presentation = str(unit.get("presentation") or "")
    return (
        {"complete": 2, "preview": 1}.get(presentation, 0),
        int(bool(unit.get("text") or unit.get("canonical_name"))),
    )


def _unit_for_output(
    stable_id: str,
    unit: Mapping[str, Any] | None,
    *,
    state: Any,
) -> dict[str, Any]:
    value = dict(unit or {})
    node_type = str(value.get("node_type") or _node_type_from_ref(state, stable_id))
    return {
        "ref": _typed_ref(state, stable_id),
        "stable_id": stable_id,
        "unit_type": node_type,
        "source_document_id": value.get("document_id"),
        "title": value.get("title"),
        "canonical_text": (
            value.get("canonical_name")
            if node_type == "ENTITY"
            else value.get("text")
        ),
        "presentation": (
            "read"
            if node_type == "CHUNK" and stable_id in state.read_chunk_ids
            else value.get("presentation", "handle")
        ),
        "evidence_eligible": (
            stable_id in state.eligible_sentence_ids
            if node_type == "SENTENCE"
            else node_type == "CHUNK" and stable_id in state.read_chunk_ids
        ),
    }


def _merge_unit_content(
    target: dict[str, Any],
    candidate: Mapping[str, Any],
    state: Any,
) -> None:
    rendered = _unit_for_output(target["stable_id"], candidate, state=state)
    if rendered.get("canonical_text") is not None:
        target["canonical_text"] = rendered["canonical_text"]
    if rendered.get("title") is not None:
        target["title"] = rendered["title"]
    if rendered.get("source_document_id") is not None:
        target["source_document_id"] = rendered["source_document_id"]
    if rendered["presentation"] in {"complete", "read"}:
        target["presentation"] = rendered["presentation"]
    target["evidence_eligible"] = bool(
        target.get("evidence_eligible") or rendered["evidence_eligible"]
    )


def _returned_ids(step: StepRecord) -> set[str]:
    return set(_result_units(step))


def _top_level_result_count(step: StepRecord) -> int:
    if (
        step.observation is None
        or not _execution(step)["progress_evaluable"]
        or _action_type(step) == "FINISH"
    ):
        return 0
    return len(step.observation.results)


def _top_level_result_ids(step: StepRecord) -> set[str]:
    if step.observation is None:
        return set()
    result: set[str] = set()
    for item in step.observation.results:
        if not isinstance(item, Mapping):
            continue
        for key in ("entity_id", "sentence_id", "chunk_id", "target_entity_id"):
            value = item.get(key)
            if isinstance(value, str):
                result.add(value)
                break
    return result


def _source_keys(step: StepRecord) -> set[str]:
    """Return stable document IDs actually present in this observation."""

    return {
        document_id
        for unit in _result_units(step).values()
        if isinstance((document_id := unit.get("document_id")), str)
        and document_id
    }


def _action_type(step: StepRecord) -> str | None:
    action = (
        step.resolved_decision.action
        if step.resolved_decision is not None
        else (step.decision.action if step.decision is not None else None)
    )
    value = getattr(action, "type", None)
    return str(getattr(value, "value", value)) if value is not None else None


def _visible_ids(state: Any) -> set[str]:
    return set(state.visible_entity_ids) | set(state.visible_sentence_ids) | set(
        state.visible_chunk_ids
    )


def _typed_ref(state: Any, stable_id: str | None) -> str | None:
    if stable_id is None:
        return None
    return state.reference_registry.stable_id_to_ref.get(stable_id)


def _node_type_from_ref(state: Any, stable_id: str) -> str:
    ref = _typed_ref(state, stable_id)
    if not ref:
        return "UNKNOWN"
    return {"E": "ENTITY", "S": "SENTENCE", "C": "CHUNK"}.get(
        ref[0], "UNKNOWN"
    )


def _budget(state: Any) -> dict[str, int]:
    return {
        "steps": state.remaining_step_budget,
        "policy_attempts": state.remaining_policy_attempt_budget,
        "retrieval_tokens": state.remaining_retrieved_token_budget,
    }


def _normalize_gold_facts(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("supporting_facts must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    pending: list[tuple[str, int, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"supporting_facts[{index}] must be an object")
        title = item.get("title")
        sentence_id = item.get("sentence_id")
        text = item.get("text")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"supporting_facts[{index}].title must be non-empty")
        if (
            not isinstance(sentence_id, int)
            or isinstance(sentence_id, bool)
            or sentence_id < 0
        ):
            raise ValueError(
                f"supporting_facts[{index}].sentence_id must be a 0-based integer"
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"supporting_facts[{index}].text must be non-empty")
        key = (_normalize_title(title), sentence_id)
        if key in seen:
            raise ValueError(f"duplicate supporting fact: {title!r}, {sentence_id}")
        seen.add(key)
        pending.append((title, sentence_id, text))
    pending.sort(key=lambda item: (_normalize_title(item[0]), item[1], item[2]))
    for ordinal, (title, sentence_id, text) in enumerate(pending, start=1):
        normalized.append(
            {
                "fact_id": f"G{ordinal}",
                "title": title,
                "sentence_id": sentence_id,
                "text": text,
            }
        )
    return normalized


def _facts_for_sentences(
    sentence_ids: Sequence[str] | set[str],
    stable_matches: Mapping[str, Sequence[str]],
) -> set[str]:
    return {
        fact_id
        for sentence_id in sentence_ids
        for fact_id in stable_matches.get(sentence_id, ())
    }


def _coverage(fact_ids: set[str], total: int) -> dict[str, Any]:
    return {
        "fact_ids": sorted(fact_ids),
        "count": len(fact_ids),
        "total": total,
        "recall": (len(fact_ids) / total if total else None),
    }


def _normalize_title(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RE.sub(
        " ", unicodedata.normalize("NFKC", str(value)).casefold()
    ).strip()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RE.sub(
        " ", unicodedata.normalize("NFKC", str(value)).casefold()
    ).strip()


def _jsonable_mapping(
    value: Mapping[str, Any] | BaseModel | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return json.loads(json.dumps(dict(value), ensure_ascii=False, default=str))


def _safe_evaluation(
    value: Mapping[str, Any] | BaseModel | None,
) -> dict[str, Any] | None:
    """Return only episode metrics that cannot disclose a reference label.

    EvaluationResult also stores ``gold_answer``, normalized gold, judge
    prompts, and raw judge output.  None of those fields may cross the
    validation/test reflection boundary.
    """

    raw = _jsonable_mapping(value)
    if raw is None:
        return None
    allowed = (
        "status",
        "llm_acc",
        "contain_acc",
        "hard",
        "soft",
        "dataset",
        "reported_metrics",
        "hard_metric",
        "soft_metric",
    )
    return {key: raw[key] for key in allowed if key in raw}


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_object(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


__all__ = [
    "REFLECTION_MANIFEST_SCHEMA_VERSION",
    "REFLECTION_SCHEMA_VERSION",
    "TRAJECTORY_REPRESENTATIONS",
    "TrajectoryRepresentation",
    "build_reflection_input",
    "build_training_reference",
    "build_training_reference_text",
    "normalize_trajectory_representation",
]
