from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.models import (
    Assessment,
    AssessmentStatus,
    ContextNodeReference,
    ContextReferenceMap,
    ExpandAction,
    FinishAction,
    PolicyDecision,
    ReadAction,
)
from agentic_rag.agent.policy import policy_decision_model
from agentic_rag.agent.references import ReferenceResolutionError, resolve_decision


def test_schema_exposes_all_four_action_families() -> None:
    schema = policy_decision_model().model_json_schema()
    serialized = str(schema)
    for action in ("SEARCH", "EXPAND", "READ", "FINISH"):
        assert action in serialized
    assert "source_ref" in serialized
    assert "chunk_ref" in serialized
    assert "evidence_refs" in serialized
    assert "source_id" not in serialized
    assert "chunk_context_index" not in serialized


def test_config_has_one_architecture_and_rejects_removed_fields(tmp_path: Path) -> None:
    config = AgentConfig()
    assert config.max_steps == 10
    assert config.max_policy_attempts == 12
    assert config.max_retrieved_tokens == 12_000

    invalid = tmp_path / "invalid.yaml"
    removed_selector = "workflow" + "_mode"
    invalid.write_text(
        f"agent:\n  {removed_selector}: old\npolicy:\n  provider: openai\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        AgentConfig.from_yaml(invalid)


def test_refs_normalize_case_but_never_fuzzy_match() -> None:
    references = ContextReferenceMap(
        typed_refs={
            "E1": ContextNodeReference(
                stable_id="entity:1", node_type="ENTITY"
            ),
            "S1": ContextNodeReference(
                stable_id="sentence:1",
                node_type="SENTENCE",
                can_use_as_evidence=True,
            ),
            "C1": ContextNodeReference(
                stable_id="chunk:1",
                node_type="CHUNK",
                can_read=True,
            ),
        }
    )
    assessment = Assessment(
        status=AssessmentStatus.INSUFFICIENT,
        missing_information=["next fact"],
    )
    expanded = resolve_decision(
        PolicyDecision(
            assessment=assessment,
            action=ExpandAction(
                kind="ENTITY_MENTIONED_IN_SENTENCE",
                source_ref=" e1 ",
                query="next fact",
            ),
        ),
        references,
    )
    assert expanded.action.source_id == "entity:1"

    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_decision(
            PolicyDecision(
                assessment=assessment,
                action=ReadAction(chunk_ref="C2"),
            ),
            references,
        )
    assert exc.value.code == "reference_not_available"

    finish = resolve_decision(
        PolicyDecision(
            assessment=Assessment(status=AssessmentStatus.SUFFICIENT),
            action=FinishAction(answer="answer", evidence_refs=["s1"]),
        ),
        references,
    )
    assert finish.action.evidence_refs[0].id == "sentence:1"


def test_reference_namespaces_are_type_checked() -> None:
    references = ContextReferenceMap(
        typed_refs={
            "E1": ContextNodeReference(
                stable_id="entity:1", node_type="ENTITY"
            )
        }
    )
    decision = PolicyDecision(
        assessment=Assessment(
            status=AssessmentStatus.INSUFFICIENT,
            missing_information=["context"],
        ),
        action=ReadAction(chunk_ref="E1"),
    )
    with pytest.raises(ReferenceResolutionError) as exc:
        resolve_decision(decision, references)
    assert exc.value.code == "reference_type_mismatch"
