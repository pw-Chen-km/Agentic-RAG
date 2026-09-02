"""Repository-local safety bridge for SkillOpt v0.2.0 patch application.

Optimizer models sometimes reproduce a Markdown target with line wrapping
collapsed to spaces.  Native SkillOpt correctly refuses that non-exact target,
but the otherwise valid edit is then lost.  This module resolves only a unique
whitespace-normalized target back to the exact substring in the current skill.
All actual edits and reports still come from SkillOpt's native implementation.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator

from agentic_rag.agent.skill import (
    FIXED_ANSWER_CONTRACT_HEADING,
    SkillDocument,
)


NativePatchApplier = Callable[[str, Any], tuple[str, list[dict[str, Any]]]]
_NORMALIZABLE_OPERATIONS = frozenset({"replace", "delete", "insert_after"})
_NATIVE_PROTECTED_MARKERS = (
    ("<!-- SLOW_UPDATE_START -->", "<!-- SLOW_UPDATE_END -->"),
    ("<!-- APPENDIX_START -->", "<!-- APPENDIX_END -->"),
)


def build_compatible_patch_applier(
    native_apply_patch_with_report: NativePatchApplier,
    *,
    fixed_answer_contract: str | None,
) -> NativePatchApplier:
    """Wrap native patch application with unique whitespace resolution.

    Edits are still applied one-by-one by the native function so a later edit
    resolves against the exact output of earlier edits.  A supplied fixed
    answer contract is restored after every native application, preventing the
    trainer's current/best skill from ever drifting away from the seed block.
    """

    if fixed_answer_contract is not None and not fixed_answer_contract.startswith(
        FIXED_ANSWER_CONTRACT_HEADING
    ):
        raise ValueError("fixed answer contract has an invalid heading")

    def compatible_apply(
        skill: str,
        patch: Any,
    ) -> tuple[str, list[dict[str, Any]]]:
        edits = _patch_edits(patch)
        reports: list[dict[str, Any]] = []
        current = skill
        for index, edit in enumerate(edits, start=1):
            requested_target = str(_edit_field(edit, "target") or "")
            operation = str(_edit_field(edit, "op") or "")
            resolved_edit = edit
            resolution: dict[str, Any] | None = None
            protected_skip_status: str | None = None

            if (
                operation in _NORMALIZABLE_OPERATIONS
                and requested_target
                and requested_target in current
            ):
                exact_start = current.find(requested_target)
                protected_skip_status = _protected_region_status(
                    current,
                    exact_start,
                    exact_start + len(requested_target),
                    protect_fixed=fixed_answer_contract is not None,
                )
            elif (
                operation in _NORMALIZABLE_OPERATIONS
                and requested_target
            ):
                matches = _whitespace_normalized_matches(
                    current, requested_target
                )
                resolution = {
                    "used": False,
                    "match_count": len(matches),
                    "requested_target": requested_target[:500],
                }
                if len(matches) == 1:
                    start, end = matches[0]
                    exact_target = current[start:end]
                    protected_skip_status = _protected_region_status(
                        current,
                        start,
                        end,
                        protect_fixed=fixed_answer_contract is not None,
                    )
                    if protected_skip_status is not None:
                        resolution["reason"] = "unique_match_is_protected"
                    else:
                        resolved_edit = _edit_as_dict(
                            edit,
                            target=exact_target,
                        )
                        resolution.update(
                            {
                                "used": True,
                                "reason": "unique_whitespace_normalized_match",
                                "resolved_target": exact_target[:500],
                            }
                        )
                elif not matches:
                    resolution["reason"] = "no_normalized_match"
                else:
                    resolution["reason"] = "ambiguous_normalized_match"

            before = current
            if protected_skip_status is not None:
                native_candidate = current
                report = {
                    "op": operation,
                    "target": requested_target[:200],
                    "content_preview": str(
                        _edit_field(edit, "content") or ""
                    )[:200],
                    "status": protected_skip_status,
                }
            else:
                native_candidate, native_reports = native_apply_patch_with_report(
                    current,
                    {"edits": [resolved_edit]},
                )
                report = (
                    dict(native_reports[0])
                    if native_reports
                    else {
                        "op": operation,
                        "target": requested_target[:200],
                        "content_preview": str(
                            _edit_field(edit, "content") or ""
                        )[:200],
                        "status": "error",
                        "error": "native patch applier returned no report",
                    }
                )
            report["index"] = index
            if resolution is not None:
                report["normalized_resolution"] = resolution

            if fixed_answer_contract is not None:
                try:
                    native_candidate, moved_tail = _move_native_append_before_fixed(
                        native_candidate,
                        fixed_answer_contract=fixed_answer_contract,
                        native_status=str(report.get("status") or ""),
                    )
                    current = SkillDocument.freeze_answer_contract(
                        native_candidate,
                        fixed_answer_contract,
                    )
                except ValueError as exc:
                    # An edit that removes/spans the heading must not make the
                    # entire training run fail or corrupt the fixed block.
                    current = before
                    report["native_status"] = report.get("status")
                    report["status"] = "skipped_fixed_answer_contract_guard"
                    report["fixed_answer_contract_error"] = str(exc)
                report["fixed_answer_contract_restored"] = True
                if moved_tail:
                    report["fixed_answer_contract_tail_relocated"] = True
            else:
                current = native_candidate
            reports.append(report)
        return current, reports

    return compatible_apply


def _move_native_append_before_fixed(
    candidate: str,
    *,
    fixed_answer_contract: str,
    native_status: str,
) -> tuple[str, bool]:
    """Keep native append/fallback content trainable while fixed stays last."""

    if native_status not in {
        "applied_append",
        "applied_insert_after_fallback_append",
    }:
        return candidate, False
    fixed_start = candidate.find(FIXED_ANSWER_CONTRACT_HEADING)
    if fixed_start < 0:
        return candidate, False
    fixed_end = fixed_start + len(fixed_answer_contract)
    if candidate[fixed_start:fixed_end] != fixed_answer_contract:
        return candidate, False
    appended_tail = candidate[fixed_end:].strip()
    if not appended_tail:
        return candidate, False
    workflow = candidate[:fixed_start].rstrip()
    return (
        f"{workflow}\n\n{appended_tail}\n\n{fixed_answer_contract}\n",
        True,
    )


@contextmanager
def use_skillopt_patch_compatibility(
    *,
    fixed_answer_contract: str | None,
) -> Iterator[None]:
    """Temporarily patch both SkillOpt references used during ``train()``."""

    try:
        import skillopt.engine.trainer as skillopt_trainer
        import skillopt.optimizer.skill as skillopt_skill
    except ImportError:
        # Fake-trainer tests intentionally run without the optional dependency.
        yield
        return

    original_optimizer_apply = skillopt_skill.apply_patch_with_report
    original_trainer_apply = skillopt_trainer.apply_patch_with_report
    compatible_apply = build_compatible_patch_applier(
        original_optimizer_apply,
        fixed_answer_contract=fixed_answer_contract,
    )
    skillopt_skill.apply_patch_with_report = compatible_apply
    skillopt_trainer.apply_patch_with_report = compatible_apply
    try:
        yield
    finally:
        skillopt_skill.apply_patch_with_report = original_optimizer_apply
        skillopt_trainer.apply_patch_with_report = original_trainer_apply


def _patch_edits(patch: Any) -> list[Any]:
    value = patch.edits if hasattr(patch, "edits") else patch.get("edits", [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return list(value)


def _edit_field(edit: Any, field: str) -> Any:
    if hasattr(edit, field):
        return getattr(edit, field)
    if isinstance(edit, Mapping):
        return edit.get(field)
    return None


def _edit_as_dict(edit: Any, *, target: str) -> dict[str, Any]:
    return {
        "op": _edit_field(edit, "op") or "",
        "content": _edit_field(edit, "content") or "",
        "target": target,
    }


def _whitespace_normalized_matches(
    source: str,
    target: str,
) -> list[tuple[int, int]]:
    normalized_source, spans = _normalize_with_spans(source)
    normalized_target = re.sub(r"\s+", " ", target).strip()
    if not normalized_target:
        return []
    matches: list[tuple[int, int]] = []
    search_from = 0
    while True:
        position = normalized_source.find(normalized_target, search_from)
        if position < 0:
            break
        end_position = position + len(normalized_target)
        matches.append((spans[position][0], spans[end_position - 1][1]))
        search_from = position + 1
    return matches


def _normalize_with_spans(value: str) -> tuple[str, list[tuple[int, int]]]:
    characters: list[str] = []
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(value):
        if value[index].isspace():
            end = index + 1
            while end < len(value) and value[end].isspace():
                end += 1
            characters.append(" ")
            spans.append((index, end))
            index = end
            continue
        characters.append(value[index])
        spans.append((index, index + 1))
        index += 1
    return "".join(characters), spans


def _protected_region_status(
    skill: str,
    start: int,
    end: int,
    *,
    protect_fixed: bool,
) -> str | None:
    if protect_fixed:
        fixed_start = skill.find(FIXED_ANSWER_CONTRACT_HEADING)
        if fixed_start >= 0 and start < len(skill) and end > fixed_start:
            return "skipped_fixed_answer_contract_guard"
    for start_marker, end_marker in _NATIVE_PROTECTED_MARKERS:
        region_start = skill.find(start_marker)
        if region_start < 0:
            continue
        marker_end = skill.find(end_marker, region_start + len(start_marker))
        region_end = (
            len(skill) if marker_end < 0 else marker_end + len(end_marker)
        )
        if start < region_end and end > region_start:
            return "skipped_protected_region"
    return None


__all__ = [
    "build_compatible_patch_applier",
    "use_skillopt_patch_compatibility",
]
