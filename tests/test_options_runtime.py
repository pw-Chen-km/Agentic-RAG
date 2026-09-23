from __future__ import annotations

from pathlib import Path

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.models import EpisodeState
from agentic_rag.agent.policy import ScriptedPolicy
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.options.catalog import OptionCatalog
from agentic_rag.options.harness import OptionsAgentHarness
from agentic_rag.options.context import OptionContextBuilder
from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.substrate.storage import Substrate
from conftest import FakeEmbeddingBackend


def test_options_runtime_returns_to_selector_and_only_answer_can_finish(
    built_substrate: Path, fake_embedder: FakeEmbeddingBackend, tmp_path: Path
) -> None:
    policy = ScriptedPolicy([
        {"option_id": "O1_START_SEARCH"},
        {
            "assessment": {"supported_facts": [], "missing_information": ["birthplace"]},
            "option_status": "CONTINUE",
            "action": {"type": "SEARCH", "query": "Where was Marie Curie born?", "method": "BM25", "target": "SENTENCE", "top_k": 5},
        },
        {"assessment": {"supported_facts": ["birthplace found"], "missing_information": []}, "option_status": "COMPLETE"},
        {"option_id": "O5_ANSWER"},
        {
            "assessment": {"supported_facts": ["Marie Curie was born in Warsaw."], "missing_information": []},
            "option_status": "CONTINUE",
            "action": {"type": "FINISH", "answer": "Warsaw", "evidence_refs": ["S1"]},
        },
    ])
    harness = OptionsAgentHarness(
        substrate=Substrate.open(built_substrate),
        config=AgentConfig(max_steps=10, max_policy_attempts=12),
        skill=SkillDocument.from_text("Use the selected option."),
        catalog=OptionCatalog.load(Path(__file__).parents[1] / "skills/options_v1.json"), policy=policy,
        output_root=tmp_path / "runs", embedding_backend=fake_embedder,
    )
    result = harness.run("Where was Marie Curie born?", "q1", episode_id="options-e2e")
    print(result.error_code, result.error_message, result.option_trace)
    assert result.answer == "Warsaw"
    assert result.termination_reason.value == "finish"
    assert result.option_trace is not None
    assert result.option_trace["selector_calls"] == 2
    assert result.option_trace["policy_calls"] == 3
    assert [event["event"] for event in result.option_trace["events"] if event["event"] == "selected"] == ["selected", "selected"]
    assert all(step.option_id for step in result.trajectory)
    assert all(step.option_id != "O5_ANSWER" for step in result.trajectory[:-1])


def test_progressive_disclosure_hides_full_catalog_from_selector(
    built_substrate: Path, fake_embedder: FakeEmbeddingBackend
) -> None:
    substrate = Substrate.open(built_substrate)
    catalog = OptionCatalog.load(Path(__file__).parents[1] / "skills/options_v1.json")
    base = PolicyContextBuilder(substrate)
    builder = OptionContextBuilder(substrate, base, catalog)
    state = EpisodeState.initial()
    _, selector_messages, _, _, _ = builder.selector(
        "Who is the author?", "ignored", state, [], scope_id="q1", last_option_id=None
    )
    selector_text = "\n".join(message.content for message in selector_messages)
    assert "Observation-driven policy" not in selector_text
    assert "Start search" in selector_text
    assert "Primitive fallback" in selector_text
    assert "Resolve bridge" not in selector_text

    policy_context = builder.policy(
        "Who is the author?", "ignored", state, [], scope_id="q1",
        option_id="O1_START_SEARCH", last_option_id=None,
    )
    policy_text = "\n".join(message.content for message in policy_context.messages)
    assert "Policy after each observation" in policy_text
    assert "Resolve bridge" not in policy_text
    assert "Fixed answer contract:" not in policy_text
