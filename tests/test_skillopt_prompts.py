from __future__ import annotations

from pathlib import Path

from agentic_rag.skillopt import trainer


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "src" / "agentic_rag" / "skillopt" / "prompts"


def _prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def test_full_agent_policy_bundle_has_truthful_provenance() -> None:
    assert trainer._PROMPT_BUNDLE_ID == "full-agent-policy-v1"
    assert trainer._CUSTOMIZED_PROMPT_FILES == (
        "analyst_error.md",
        "analyst_success.md",
        "ranking.md",
    )
    source = (PROMPTS / "SOURCE.md").read_text(encoding="utf-8")
    assert "full-agent-policy-v1" in source
    assert "not verbatim upstream prompts" in source
    for name in trainer._PROMPT_NAMES:
        assert (PROMPTS / f"{name}.md").is_file()


def test_failure_analyst_covers_the_complete_trainable_decision_scope() -> None:
    prompt = _prompt("analyst_error")
    for label in (
        "action_family_selection",
        "search_configuration",
        "expand_configuration",
        "read_selection",
        "finish_timing",
        "evidence_selection",
        "answer_formulation",
        "retrieval_tool_or_corpus",
        "interface_or_infrastructure",
    ):
        assert f"`{label}`" in prompt
    assert "representation legend" in prompt
    assert "Count trajectories, not actions, retries, or steps" in prompt
    assert "between 1 and batch_size" in prompt
    assert "diagnose answer formulation rather than retrieval" in prompt
    assert '"failure_summary"' in prompt
    assert '"patch"' in prompt
    assert '"edits"' in prompt


def test_success_analyst_learns_actions_stopping_evidence_and_answers() -> None:
    prompt = _prompt("analyst_success")
    for phrase in (
        "SEARCH query wording, method, and target",
        "EXPAND kind, parent/source choice, direction",
        "stopping only when all answer obligations are supported",
        "selection of eligible evidence",
        "answer extraction and formulation",
        "representation legend",
    ):
        assert phrase in prompt
    assert "accidental match" in prompt
    assert '"success_patterns"' in prompt
    assert '"edits"' in prompt


def test_analysts_and_ranker_protect_only_the_evidence_safety_block() -> None:
    for name in ("analyst_error", "analyst_success", "ranking"):
        prompt = _prompt(name)
        assert "`## Fixed answer contract`" in prompt
        assert "Protocol" in prompt
        assert "schema" in prompt
        assert "Controller" in prompt
        assert "validator" in prompt
    ranking = _prompt("ranking")
    assert '"selected_indices"' in ranking
    assert "may be an empty list" in ranking
    assert "answer extraction and formulation" in ranking
