from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = PROJECT_ROOT / "skills"
BASELINE_SOURCE = SKILL_DIR / "hotpotqa_single_agent_v3.md"
SKILLS = {
    "a": SKILL_DIR / "hotpotqa_v3_a_baseline_clean.md",
    "b": SKILL_DIR / "hotpotqa_v3_b_chunk_entity_expert.md",
    "c": SKILL_DIR / "hotpotqa_v3_c_posthoc_guarded.md",
}

TEST_SPECIFIC_PHRASES = (
    "more than 330 million people",
    "american schoolteacher and publisher",
    "matt groening",
    "livin' la vida loca",
    "midnight madness",
    "tori amos",
    "hussein ali mahfouz",
    "semitic languages",
    "moses harman",
    "futurama",
    "coal chamber",
    "return to oz",
    "tim armstrong",
)


def _content(key: str) -> str:
    return SKILLS[key].read_text(encoding="utf-8")


def test_all_variations_are_nonempty_and_free_of_test_specific_facts() -> None:
    for path in SKILLS.values():
        content = path.read_text(encoding="utf-8")
        assert content.strip()
        lowered = content.casefold()
        assert not any(phrase in lowered for phrase in TEST_SPECIFIC_PHRASES)


def test_clean_baseline_changes_only_the_leaking_compound_example() -> None:
    original = BASELINE_SOURCE.read_text(encoding="utf-8")
    expected = original.replace(
        "Copy\ncompound wording completely: if the evidence says `American "
        "schoolteacher and\npublisher`, do not shorten it to one occupation.",
        "When\nevidence states multiple coordinated roles or qualifiers, "
        "preserve all of them\ninstead of returning only one.",
    )
    assert _content("a") == expected


def test_chunk_entity_skill_encodes_the_fixed_first_action() -> None:
    content = _content("b")
    assert "first action for every question" in content
    assert "`method=BM25`" in content
    assert "`target=CHUNK`" in content
    assert "`LEXICAL -> ENTITY`" in content
    assert "`ENTITY_MENTIONED_IN_SENTENCE`" in content


def test_guarded_skill_covers_observed_failure_classes() -> None:
    content = _content("c")
    for error_code in (
        "duplicate_action",
        "memory_node_type_mismatch",
        "chunk_not_readable",
        "citation_index_out_of_range",
    ):
        assert f"`{error_code}`" in content
    for gate in (
        "Evidence gate",
        "Progress gate",
        "Novelty gate",
        "Reference gate",
    ):
        assert gate in content
