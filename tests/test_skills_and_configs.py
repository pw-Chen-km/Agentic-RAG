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


def test_baseline_has_a_fixed_answer_contract() -> None:
    skill = SkillDocument.load(ROOT / "skills" / "baseline.md")
    fixed = skill.fixed_answer_contract
    assert fixed is not None
    assert fixed.startswith("## Fixed answer contract")
    assert skill.content.count("## Trainable action and answer workflow") == 1
    assert skill.content.count("## Fixed answer contract") == 1
    assert skill.content.index("## Trainable action and answer workflow") < (
        skill.content.index("## Fixed answer contract")
    )

    candidate = skill.content.replace(
        "The final answer must be supported by evidence that is legal for FINISH.",
        "The final answer may ignore the evidence.",
    )
    frozen = SkillDocument.freeze_answer_contract(candidate, fixed)
    assert "The final answer may ignore the evidence." not in frozen
    trainable_candidate = skill.content.replace(
        "Return a direct, concise, but complete answer.",
        "Return the requested value in a complete sentence.",
    )
    trainable_frozen = SkillDocument.freeze_answer_contract(
        trainable_candidate,
        fixed,
    )
    assert "Return the requested value in a complete sentence." in (
        trainable_frozen
    )
    assert "query must contain that name alone" in skill.content
    assert "Eric A. Sykes nationality country" in skill.content
    assert "preview appears to contain the answer" in skill.content
    assert "inspect `latest_attempt`" in skill.content
    assert "choosing FINISH is itself the decision to answer" in skill.content
    assert frozen.endswith(f"{fixed}\n")


def test_skill_without_fixed_contract_remains_backward_compatible() -> None:
    content = "# Retrieval skill\n\nSearch, then answer.\n"
    skill = SkillDocument.from_text(content)
    assert skill.fixed_answer_contract is None
    assert SkillDocument.freeze_answer_contract(content, None) == content


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
        assert config.show_available_action_options is True
        assert config.use_state_conditioned_schema is True


def test_action_option_ablation_configs_differ_only_by_prompt_flag() -> None:
    enabled = AgentConfig.from_yaml(
        ROOT / "configs" / "hotpotqa_qwen36_amd_nothink_options_on.yaml"
    )
    disabled = AgentConfig.from_yaml(
        ROOT / "configs" / "hotpotqa_qwen36_amd_nothink_options_off.yaml"
    )
    assert enabled.show_available_action_options is True
    assert disabled.show_available_action_options is False
    enabled_payload = enabled.model_dump(mode="json")
    disabled_payload = disabled.model_dump(mode="json")
    enabled_payload.pop("show_available_action_options")
    disabled_payload.pop("show_available_action_options")
    assert enabled_payload == disabled_payload


def test_prompt_and_schema_ablation_configs_cover_all_four_combinations() -> None:
    names = {
        (True, True): "hotpotqa_qwen36_amd_nothink_options_on.yaml",
        (False, True): "hotpotqa_qwen36_amd_nothink_options_off.yaml",
        (True, False): "hotpotqa_qwen36_amd_nothink_options_on_schema_off.yaml",
        (False, False): "hotpotqa_qwen36_amd_nothink_options_off_schema_off.yaml",
    }
    normalized = []
    for expected, name in names.items():
        config = AgentConfig.from_yaml(ROOT / "configs" / name)
        assert (
            config.show_available_action_options,
            config.use_state_conditioned_schema,
        ) == expected
        payload = config.model_dump(mode="json")
        payload.pop("show_available_action_options")
        payload.pop("use_state_conditioned_schema")
        normalized.append(payload)
    assert all(payload == normalized[0] for payload in normalized[1:])
