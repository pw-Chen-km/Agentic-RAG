from pathlib import Path

import pytest

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (Assessment, FinishAction, PolicyDecision, SearchAction,
                                      SearchMethod, SearchTarget, Usage)
from agentic_rag.agent.policy import PolicyResponseError
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.tool_calling import decision_from_tool_call
from agentic_rag.substrate.storage import Substrate


@pytest.mark.parametrize("condition", ["C0", "C1", "C2", "C3", "C5", "C4", "A1"])
@pytest.mark.parametrize("invalid_after_search", [False, True])
def test_spans_describe_input_not_new_retrieval(
    built_substrate: Path, fake_embedder, tmp_path: Path,
    condition: str, invalid_after_search: bool,
):
    class Policy:
        last_usage = Usage(policy_calls=1, input_tokens=10, output_tokens=2, total_tokens=12)
        last_usage_metadata = {}
        count = 0

        def decide(self, messages, *, tools=None, **kwargs):
            assert tools
            self.count += 1
            if self.count == 1:
                decision = PolicyDecision(
                    assessment=Assessment(missing_information=["Where Marie Curie was born"]),
                    action=SearchAction(query="Marie Curie", method=SearchMethod.DENSE, target=SearchTarget.CHUNK),
                )
            elif invalid_after_search and self.count == 2:
                raise PolicyResponseError("test malformed constrained decision")
            else:
                decision = PolicyDecision(
                    assessment=Assessment(supported_facts=["Marie Curie was born in Warsaw"]),
                    action=FinishAction(answer="Warsaw", evidence_refs=[]),
                )
            self.last_usage_metadata = {"constrained_single_decision": True, "decision_count": 1}
            return decision

    harness = AgentHarness(substrate=Substrate.open(built_substrate),
        config=AgentConfig(interface=condition, max_steps=15, max_policy_attempts=15),
        skill=SkillDocument.from_text("Answer using available evidence."),
        policy=Policy(), output_root=tmp_path / "run", embedding_backend=fake_embedder)
    result = harness.run("Where was Marie Curie born?", "q1")
    assert result.termination_reason.value == "finish"
    first, *later = result.trajectory
    assert first.visible_source_spans == []  # retrieval hasn't reached the model yet
    assert first.observation.metadata["projected_source_spans"]
    assert first.decision.assessment.missing_information == ["Where Marie Curie was born"]
    for step in later:
        assert step.visible_source_spans
        assert step.tool_definitions
        content = "\n".join(m.content or "" for m in step.messages)
        assert "PREVIOUS ASSESSMENT" in content
        assert "Where Marie Curie was born" in content
        for span in step.visible_source_spans:
            assert span["span_type"] == "sentence"
            assert span["complete"] and span["seen_by_policy"]
            assert span["text"] in content
        assert step.telemetry["tool_schema_token_estimate"] > 0
