from __future__ import annotations

from pathlib import Path

from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.models import EpisodeState
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.substrate.storage import Substrate


def test_empty_context_is_neutral_and_has_compact_budget(built_substrate: Path) -> None:
    substrate = Substrate.open(built_substrate)
    state = EpisodeState.initial(
        max_steps=10,
        max_policy_attempts=12,
        max_retrieved_tokens=12_000,
    )
    built = PolicyContextBuilder(substrate).build(
        "Where was Marie Curie born?",
        SkillDocument.from_text("Use evidence."),
        state,
        [],
        scope_id="q1",
    )
    prompt = "\n".join(message.content for message in built.messages)
    assert "Original question" in prompt
    assert "Action meanings" in prompt
    assert "Current trainable retrieval skill" in prompt
    assert '"last_assessment"' in prompt
    assert '"semantic_memory"' in prompt
    assert '"attempted_actions"' in prompt
    assert "Budget: 10 steps, 12 attempts, 12000 retrieval tokens left" in prompt
    assert "Latest event" not in prompt
    assert "Action Targets" not in prompt
    assert "Allowed actions" not in prompt
    assert built.reference_map.typed_refs == {}
