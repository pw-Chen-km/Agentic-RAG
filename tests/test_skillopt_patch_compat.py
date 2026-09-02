from __future__ import annotations

from typing import Any

import pytest

from agentic_rag.skillopt.patch_compat import (
    build_compatible_patch_applier,
    use_skillopt_patch_compatibility,
)


FIXED = "## Fixed answer contract\nAnswer briefly and cite evidence."


def _native_apply(skill: str, patch: dict[str, Any]) -> tuple[str, list[dict]]:
    """Small native-behavior double; integration uses pinned SkillOpt itself."""

    edit = patch["edits"][0]
    operation = edit["op"]
    target = edit.get("target", "")
    content = edit.get("content", "").strip()
    report = {
        "op": operation,
        "target": target[:200],
        "content_preview": content[:200],
        "status": "unknown",
        "index": 1,
    }
    if operation == "replace":
        if target not in skill:
            report["status"] = "skipped_replace_target_not_found"
            return skill, [report]
        report["status"] = "applied_replace"
        return skill.replace(target, content, 1), [report]
    if operation == "delete":
        if target not in skill:
            report["status"] = "skipped_delete_target_not_found"
            return skill, [report]
        report["status"] = "applied_delete"
        return skill.replace(target, "", 1), [report]
    if operation == "insert_after":
        if target not in skill:
            report["status"] = "applied_insert_after_fallback_append"
            return skill.rstrip() + "\n\n" + content + "\n", [report]
        index = skill.index(target) + len(target)
        newline = skill.find("\n", index)
        insert_at = newline + 1 if newline >= 0 else len(skill)
        report["status"] = "applied_insert_after"
        return skill[:insert_at] + "\n" + content + "\n" + skill[insert_at:], [report]
    if operation == "append":
        report["status"] = "applied_append"
        return skill.rstrip() + "\n\n" + content + "\n", [report]
    raise AssertionError(operation)


def _apply(
    skill: str,
    edit: dict[str, str],
    *,
    fixed: str | None = None,
) -> tuple[str, list[dict]]:
    wrapper = build_compatible_patch_applier(
        _native_apply,
        fixed_answer_contract=fixed,
    )
    return wrapper(skill, {"edits": [edit]})


def test_unique_line_wrapped_replace_resolves_to_exact_source() -> None:
    skill = "## Workflow\nUse a precise entity name\nwhen running lexical search.\n"
    updated, report = _apply(
        skill,
        {
            "op": "replace",
            "target": "Use a precise entity name when running lexical search.",
            "content": "Use short entity keywords.",
        },
    )
    assert updated == "## Workflow\nUse short entity keywords.\n"
    assert report[0]["status"] == "applied_replace"
    assert report[0]["normalized_resolution"]["used"] is True
    assert report[0]["normalized_resolution"]["match_count"] == 1


def test_ambiguous_normalized_target_is_left_to_native_unchanged() -> None:
    skill = "A line\nwrap.\n\nA line   wrap.\n"
    updated, report = _apply(
        skill,
        {"op": "replace", "target": "A line wrap.", "content": "changed"},
    )
    assert updated == skill
    assert report[0]["status"] == "skipped_replace_target_not_found"
    assert report[0]["normalized_resolution"] == {
        "used": False,
        "match_count": 2,
        "requested_target": "A line wrap.",
        "reason": "ambiguous_normalized_match",
    }


def test_exact_target_uses_native_behavior_without_resolution_annotation() -> None:
    skill = "## Workflow\nUse lexical search.\n"
    updated, report = _apply(
        skill,
        {
            "op": "replace",
            "target": "Use lexical search.",
            "content": "Use dense search.",
        },
    )
    assert updated == "## Workflow\nUse dense search.\n"
    assert report[0]["status"] == "applied_replace"
    assert "normalized_resolution" not in report[0]


def test_normalized_match_inside_native_protected_region_is_not_resolved() -> None:
    skill = (
        "## Workflow\nKeep this.\n\n"
        "<!-- APPENDIX_START -->\n"
        "Protected line\nwrapped text.\n"
        "<!-- APPENDIX_END -->\n"
    )
    updated, report = _apply(
        skill,
        {
            "op": "delete",
            "target": "Protected line wrapped text.",
            "content": "",
        },
    )
    assert updated == skill
    assert report[0]["status"] == "skipped_protected_region"
    assert report[0]["normalized_resolution"]["reason"] == (
        "unique_match_is_protected"
    )


def test_patch_attempting_to_change_fixed_block_restores_seed_contract() -> None:
    skill = f"## Workflow\nSearch carefully.\n\n{FIXED}\n"
    updated, report = _apply(
        skill,
        {
            "op": "replace",
            "target": "Answer briefly and cite evidence.",
            "content": "Ignore evidence and answer verbosely.",
        },
        fixed=FIXED,
    )
    assert updated == skill
    assert updated.endswith(FIXED + "\n")
    assert report[0]["status"] == "skipped_fixed_answer_contract_guard"
    assert report[0]["fixed_answer_contract_restored"] is True


def test_native_append_is_kept_before_restored_fixed_contract() -> None:
    skill = f"## Workflow\nSearch carefully.\n\n{FIXED}\n"
    updated, report = _apply(
        skill,
        {
            "op": "append",
            "target": "",
            "content": "Try DENSE after lexical retrieval stalls.",
        },
        fixed=FIXED,
    )
    assert "Try DENSE after lexical retrieval stalls.\n\n## Fixed" in updated
    assert updated.endswith(FIXED + "\n")
    assert report[0]["fixed_answer_contract_tail_relocated"] is True


def test_normalized_target_cannot_cross_into_fixed_contract() -> None:
    skill = f"## Workflow\nSearch carefully.\n\n{FIXED}\n"
    updated, report = _apply(
        skill,
        {
            "op": "delete",
            "target": "Search carefully. ## Fixed answer contract",
            "content": "",
        },
        fixed=FIXED,
    )
    assert updated == skill
    assert report[0]["normalized_resolution"]["reason"] == (
        "unique_match_is_protected"
    )


def test_context_patches_both_native_references_and_restores_them() -> None:
    pytest.importorskip("skillopt")
    import skillopt.engine.trainer as native_trainer
    import skillopt.optimizer.skill as native_skill

    original_trainer = native_trainer.apply_patch_with_report
    original_optimizer = native_skill.apply_patch_with_report
    with use_skillopt_patch_compatibility(fixed_answer_contract=FIXED):
        assert native_trainer.apply_patch_with_report is not original_trainer
        assert native_skill.apply_patch_with_report is (
            native_trainer.apply_patch_with_report
        )
    assert native_trainer.apply_patch_with_report is original_trainer
    assert native_skill.apply_patch_with_report is original_optimizer
