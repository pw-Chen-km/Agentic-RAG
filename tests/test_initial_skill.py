from pathlib import Path

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.skill import SkillDocument


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INITIAL_SKILL = PROJECT_ROOT / "skills" / "hotpotqa_skillopt_initial.md"
HOTPOTQA_CONFIG = (
    PROJECT_ROOT / "configs" / "agentic_hotpotqa_skillopt_smoke.yaml"
)


def test_hotpotqa_initial_skill_covers_runtime_action_contract() -> None:
    skill = SkillDocument.load(INITIAL_SKILL)
    content = skill.content

    for action_type in ("SEARCH", "EXPAND", "READ", "FINISH"):
        assert f"`{action_type}`" in content

    config = AgentConfig.from_yaml(HOTPOTQA_CONFIG)
    for kind in config.enabled_expansions:
        assert f"`{kind.value}`" in content

    for handle in ("`E#`", "`S#`", "`C#`"):
        assert handle in content

    assert "selected_evidence_refs" in content
    assert "allowed_expansions" in content
