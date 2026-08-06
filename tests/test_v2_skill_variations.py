from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"


def _read(name: str) -> str:
    return (SKILLS / name).read_text(encoding="utf-8")


def test_v2_variations_are_nonempty_and_do_not_embed_test_answers() -> None:
    names = (
        "hotpotqa_v2_a_evidence_adaptive.md",
        "hotpotqa_v2_b_chunk_entity_expert.md",
        "hotpotqa_v2_c_posthoc_guarded.md",
    )
    forbidden = (
        "more than 330 million people",
        "american schoolteacher and publisher",
        "matt groening",
        "livin' la vida loca",
        "midnight madness",
        "tori amos",
    )
    for name in names:
        text = _read(name)
        assert text.strip()
        lowered = text.casefold()
        assert all(answer not in lowered for answer in forbidden)


def test_v2_a_is_adaptive_and_uses_handle_evidence_contract() -> None:
    text = _read("hotpotqa_v2_a_evidence_adaptive.md")
    assert "exact remaining fact" in text
    assert "SEARCH remains available" in text
    assert "selected_evidence_refs" in text
    assert "allowed_expansions" in text


def test_v2_b_requires_chunk_first_and_entity_drilldown() -> None:
    text = _read("hotpotqa_v2_b_chunk_entity_expert.md")
    assert "first action for every question must be `SEARCH`" in text
    assert "`target=CHUNK`" in text
    assert "ENTITY_MENTIONED_IN_SENTENCE" in text
    assert "visible E#" in text


def test_v2_c_has_four_gates_and_v2_error_recovery() -> None:
    text = _read("hotpotqa_v2_c_posthoc_guarded.md")
    for gate in ("Evidence gate", "Progress gate", "Novelty gate", "Handle gate"):
        assert gate in text
    for error in ("duplicate_action", "handle_type_mismatch", "unknown_handle"):
        assert error in text


def test_v2_variations_do_not_use_v3_context_indices() -> None:
    for name in (
        "hotpotqa_v2_a_evidence_adaptive.md",
        "hotpotqa_v2_b_chunk_entity_expert.md",
        "hotpotqa_v2_c_posthoc_guarded.md",
    ):
        text = _read(name)
        assert "source_context_index" not in text
        assert "chunk_context_index" not in text
        assert "citation_index" not in text
