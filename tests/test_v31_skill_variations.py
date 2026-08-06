from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = PROJECT_ROOT / "skills"
SKILLS = {
    "a": SKILL_DIR / "hotpotqa_v31_a_baseline_clean.md",
    "b": SKILL_DIR / "hotpotqa_v31_b_chunk_entity_expert.md",
    "c": SKILL_DIR / "hotpotqa_v31_c_posthoc_guarded.md",
}


def test_v31_skills_use_typed_refs_without_old_indices() -> None:
    for path in SKILLS.values():
        content = path.read_text(encoding="utf-8")
        assert content.strip()
        assert "E#" in content
        assert "S#" in content
        assert "C#" in content
        assert "citation index" not in content.casefold()
        assert "memory index" not in content.casefold()


def test_v31_chunk_entity_skill_preserves_procedure() -> None:
    content = SKILLS["b"].read_text(encoding="utf-8")
    assert "first action for every question" in content
    assert "`method=BM25`" in content
    assert "`target=CHUNK`" in content
    assert "`LEXICAL -> ENTITY`" in content
    assert "`ENTITY_MENTIONED_IN_SENTENCE`" in content


def test_v31_guarded_skill_names_new_recovery_codes() -> None:
    content = SKILLS["c"].read_text(encoding="utf-8")
    for error_code in (
        "duplicate_action",
        "reference_not_available",
        "reference_type_mismatch",
        "reference_not_evidence",
        "chunk_not_readable",
    ):
        assert f"`{error_code}`" in content
