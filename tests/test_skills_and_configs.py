from __future__ import annotations

from pathlib import Path

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.skill import SkillDocument


ROOT = Path(__file__).resolve().parents[1]


def test_all_research_skills_share_one_contract() -> None:
    for name in (
        "baseline.md",
        "chunk_entity_expert.md",
        "guarded_procedure.md",
        "skillopt_seed.md",
    ):
        skill = SkillDocument.load(ROOT / "skills" / name)
        assert skill.content.strip()
        assert ("selected_evidence" + "_refs") not in skill.content
        assert "context_index" not in skill.content


def test_luna_qwen_and_smoke_configs_load_without_architecture_switch() -> None:
    configs = {
        name: AgentConfig.from_yaml(ROOT / "configs" / name)
        for name in (
            "hotpotqa_luna.yaml",
            "hotpotqa_qwen.yaml",
            "hotpotqa_qwen_thinking.yaml",
            "hotpotqa_smoke.yaml",
        )
    }
    assert configs["hotpotqa_luna.yaml"].policy.provider == "openai"
    assert configs["hotpotqa_qwen.yaml"].policy.provider == "ollama"
    thinking = configs["hotpotqa_qwen_thinking.yaml"].policy
    assert thinking.provider == "ollama"
    assert thinking.think == "high"
    for config in configs.values():
        assert config.max_steps == 10
        assert config.max_policy_attempts == 12
        assert config.max_retrieved_tokens == 12_000
