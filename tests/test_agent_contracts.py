from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agentic_rag.agent.answer import (
    OpenAIResponsesAnswerGenerator,
    ScriptedAnswerGenerator,
)
from agentic_rag.agent.config import (
    AgentConfig,
    AnswerConfig,
    OllamaAnswerConfig,
    OllamaPolicyConfig,
    PolicyConfig,
)
from agentic_rag.agent.context import AppendOnlyContextBuilder
from agentic_rag.agent.models import (
    DEFAULT_ENABLED_EXPANSIONS,
    AssessmentStatus,
    ContextMode,
    ControllerState,
    EvidenceAssessment,
    ExpandAction,
    ExpansionKind,
    FinishAction,
    Message,
    Observation,
    ObservationStatus,
    PolicyDecision,
    ResolvedEvidence,
    SearchAction,
    SentenceRef,
    StepRecord,
    ValidationStatus,
)
from agentic_rag.agent.policy import (
    OpenAIResponsesPolicy,
    PolicyResponseError,
    PolicyTransportError,
    ScriptedPolicy,
    policy_decision_model,
)
from agentic_rag.agent.skill import SkillDocument


def _assessment(status: AssessmentStatus = AssessmentStatus.INSUFFICIENT):
    return EvidenceAssessment(
        status=status,
        missing_information=["The birthplace is unknown."]
        if status is AssessmentStatus.INSUFFICIENT
        else [],
    )


def _search_decision() -> PolicyDecision:
    return PolicyDecision(
        assessment=_assessment(),
        action=SearchAction(
            query="Where was Marie Curie born?",
            method="BM25",
            target="SENTENCE",
        ),
    )


def test_action_contracts_are_discriminated_and_fixed_top_k() -> None:
    decision = PolicyDecision.model_validate(
        {
            "assessment": {
                "status": "INSUFFICIENT",
                "supported_facts": [],
                "missing_information": ["birthplace"],
                "selected_evidence_refs": [],
            },
            "action": {
                "type": "SEARCH",
                "query": "Where was Marie Curie born?",
                "method": "BM25",
                "target": "SENTENCE",
                "top_k": 5,
            },
        }
    )
    assert isinstance(decision.action, SearchAction)
    assert decision.action.query == "Where was Marie Curie born?"
    provider_schema = json.dumps(
        policy_decision_model().model_json_schema(),
        sort_keys=True,
    )
    assert "selected_evidence_refs" in provider_schema
    assert "candidate_evidence_refs" not in provider_schema
    assert '"maxItems": 20' in provider_schema

    with pytest.raises(ValidationError):
        SearchAction(
            query="Where was Marie Curie born?",
            method="BM25",
            target="SENTENCE",
            top_k=3,
        )
    with pytest.raises(ValidationError):
        SearchAction(
            query="Marie Curie",
            method="LEXICAL",
            target="CHUNK",
        )


def test_expansion_contract_has_eight_kinds_and_four_defaults() -> None:
    assert len(ExpansionKind) == 8
    assert len(DEFAULT_ENABLED_EXPANSIONS) == 4
    assert "SENTENCE_PART_OF_CHUNK" not in {
        item.value for item in ExpansionKind
    }
    assert set(DEFAULT_ENABLED_EXPANSIONS) == {
        ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
        ExpansionKind.SENTENCE_MENTIONS_ENTITY,
        ExpansionKind.ENTITY_CO_OCCURS_ENTITY_SENTENCE,
        ExpansionKind.CHUNK_ADJACENT_CHUNK,
    }
    adjacent = ExpandAction(
        kind="CHUNK_ADJACENT_CHUNK",
        source_id="chunk:C4",
        direction="NEXT",
    )
    assert adjacent.direction == "NEXT"
    with pytest.raises(ValidationError):
        ExpandAction(kind="CHUNK_ADJACENT_CHUNK", source_id="chunk:C4")
    with pytest.raises(ValidationError):
        ExpandAction(
            kind="SENTENCE_MENTIONS_ENTITY",
            source_id="sentence:S12",
            direction="NEXT",
        )


def test_finish_requires_typed_evidence() -> None:
    action = FinishAction(
        evidence_refs=[{"unit": "SENTENCE", "id": "sentence:S12"}]
    )
    assert isinstance(action.evidence_refs[0], SentenceRef)
    with pytest.raises(ValidationError):
        FinishAction(evidence_refs=[])
    with pytest.raises(ValidationError):
        FinishAction(
            evidence_refs=[{"unit": "ENTITY", "id": "entity:Marie_Curie"}]
        )
    duplicate = {"unit": "SENTENCE", "id": "sentence:S12"}
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        FinishAction(evidence_refs=[duplicate, duplicate])
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        EvidenceAssessment(
            status="INSUFFICIENT",
            selected_evidence_refs=[duplicate, duplicate],
        )


def test_agent_config_loads_nested_yaml_and_rejects_removed_kind(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        """
agent:
  max_steps: 7
  max_policy_attempts: 9
  max_consecutive_invalid_attempts: 3
  max_retrieved_tokens: 9000
  enabled_expansions:
    - ENTITY_MENTIONED_IN_SENTENCE
    - SENTENCE_MENTIONS_ENTITY
policy:
  model: gpt-5.6-terra
answer:
  model: gpt-5.6-luna
""".lstrip(),
        encoding="utf-8",
    )
    config = AgentConfig.from_yaml(config_path)
    assert config.max_steps == 7
    assert config.max_policy_attempts == 9
    assert config.max_consecutive_invalid_attempts == 3
    assert config.context_mode is ContextMode.COMPACT_EVIDENCE
    assert config.policy.model == "gpt-5.6-terra"
    assert config.enabled_expansions == (
        ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,
        ExpansionKind.SENTENCE_MENTIONS_ENTITY,
    )

    config_path.write_text(
        """
agent:
  enabled_expansions:
    - SENTENCE_PART_OF_CHUNK
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        AgentConfig.from_yaml(config_path)
    with pytest.raises(ValidationError):
        AgentConfig(policy={"provider": "not-implemented"})
    with pytest.raises(ValidationError):
        AgentConfig(context_mode="latest_only")


def test_agent_config_rejects_unknown_root_and_duplicate_sections(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "invalid-agent.yaml"
    config_path.write_text(
        """
agent:
  max_steps: 4
polciy:
  model: gpt-5.6-terra
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"unknown root key.*polciy"):
        AgentConfig.from_yaml(config_path)

    config_path.write_text(
        """
agent:
  max_steps: 4
  policy:
    model: gpt-5.6-terra
policy:
  model: gpt-5.6-luna
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="policy.*not both"):
        AgentConfig.from_yaml(config_path)


def test_agent_config_discriminates_local_ollama_providers() -> None:
    backwards_compatible = AgentConfig(
        policy=PolicyConfig(),
        answer=AnswerConfig(),
    )
    assert backwards_compatible.policy.provider == "openai"
    assert backwards_compatible.answer.provider == "openai"

    config = AgentConfig(
        policy={"provider": "ollama", "model": "qwen3:8b"},
        answer={
            "provider": "ollama",
            "model": "qwen3:8b",
            "think": "low",
            "keep_alive": "10m",
        },
    )
    assert isinstance(config.policy, OllamaPolicyConfig)
    assert isinstance(config.answer, OllamaAnswerConfig)
    assert config.policy.host == "http://localhost:11434"
    assert config.policy.temperature == 0
    assert config.policy.think is False
    assert config.policy.timeout_seconds == 300
    assert config.policy.max_retries == 2
    assert config.policy.num_ctx == 32_768
    assert config.answer.think == "low"
    assert config.effective_dict()["policy"]["provider"] == "ollama"


def test_ollama_config_requires_model_and_rejects_cloud() -> None:
    with pytest.raises(ValidationError):
        AgentConfig(policy={"provider": "ollama"})

    with pytest.raises(ValidationError) as error:
        AgentConfig(
            policy={
                "provider": "ollama",
                "model": "qwen3:8b",
                "host": "https://api.ollama.com",
            }
        )
    assert error.value.errors()[0]["type"] == "ollama_cloud_unsupported"

    assert OllamaPolicyConfig(model="qwen3:8b", think=True).think is True
    assert OllamaPolicyConfig(model="qwen3:8b", think=None).think is None
    with pytest.raises(ValidationError):
        OllamaAnswerConfig(model="qwen3:8b", num_ctx=1_024)


@pytest.mark.parametrize(
    "host",
    [
        "ftp://localhost:11434",
        "localhost:11434",
        "http://",
        "http://:11434",
        "http://localhost:not-a-port",
        "http://localhost:70000",
        "http://local host:11434",
        "http://user:secret@localhost:11434",
        "http://localhost:11434?token=secret",
        "http://localhost:11434#fragment",
        "://localhost:11434",
    ],
)
def test_ollama_config_rejects_non_http_or_malformed_hosts(
    host: str,
) -> None:
    with pytest.raises(ValidationError) as error:
        OllamaPolicyConfig(model="qwen3:8b", host=host)
    assert error.value.errors()[0]["type"] == "ollama_host_invalid"


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("http://localhost:11434/", "http://localhost:11434"),
        ("https://ollama.internal:443", "https://ollama.internal:443"),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_ollama_config_accepts_http_hosts(
    host: str, expected: str
) -> None:
    config = OllamaAnswerConfig(model="qwen3:8b", host=host)
    assert config.host == expected


@pytest.mark.parametrize(
    "keep_alive",
    [True, False, float("nan"), float("inf"), float("-inf")],
)
def test_ollama_config_rejects_invalid_numeric_keep_alive(
    keep_alive: bool | float,
) -> None:
    with pytest.raises(ValidationError):
        OllamaAnswerConfig(
            model="qwen3:8b", keep_alive=keep_alive
        )


def test_provider_policy_schema_contains_only_enabled_expansions() -> None:
    provider_model = policy_decision_model(DEFAULT_ENABLED_EXPANSIONS)
    schema = provider_model.model_json_schema()
    assert len(provider_model.__name__) <= 64
    assert _schema_literal_values(schema) & {
        item.value for item in ExpansionKind
    } == {
        item.value for item in DEFAULT_ENABLED_EXPANSIONS
    }
    assert "SENTENCE_PART_OF_CHUNK" not in str(schema)
    assert not _schema_keys(schema) & {
        "oneOf",
        "discriminator",
        "default",
        "allOf",
        "if",
        "then",
        "else",
    }
    _assert_all_object_fields_required(schema)

    for enabled in (
        (ExpansionKind.CHUNK_ADJACENT_CHUNK,),
        (ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,),
        tuple(ExpansionKind),
    ):
        dynamic_schema = policy_decision_model(enabled).model_json_schema()
        assert _schema_literal_values(dynamic_schema) & {
            item.value for item in ExpansionKind
        } == {item.value for item in enabled}
        assert not _schema_keys(dynamic_schema) & {
            "oneOf",
            "discriminator",
            "default",
            "allOf",
            "if",
            "then",
            "else",
        }
        _assert_all_object_fields_required(dynamic_schema)

    no_expand_schema = policy_decision_model(()).model_json_schema()
    action_schema = no_expand_schema["properties"]["action"]
    action_variants = action_schema["anyOf"]
    variant_refs = {item["$ref"] for item in action_variants}
    assert all("ExpandAction" not in ref for ref in variant_refs)
    assert not _schema_keys(no_expand_schema) & {
        "oneOf",
        "discriminator",
        "default",
    }
    _assert_all_object_fields_required(no_expand_schema)


def test_provider_expand_variants_enforce_direction_by_kind() -> None:
    provider_model = policy_decision_model(DEFAULT_ENABLED_EXPANSIONS)
    common = {
        "assessment": {
            "status": "INSUFFICIENT",
            "supported_facts": [],
            "missing_information": ["birthplace"],
            "selected_evidence_refs": [],
        }
    }

    adjacent = {
        **common,
        "action": {
            "type": "EXPAND",
            "kind": "CHUNK_ADJACENT_CHUNK",
            "source_id": "chunk:C4",
            "direction": "NEXT",
            "query": None,
            "top_k": 5,
        },
    }
    assert provider_model.model_validate(adjacent).action.direction == "NEXT"
    with pytest.raises(ValidationError):
        provider_model.model_validate(
            {
                **common,
                "action": {**adjacent["action"], "direction": None},
            }
        )

    non_adjacent = {
        **common,
        "action": {
            "type": "EXPAND",
            "kind": "ENTITY_MENTIONED_IN_SENTENCE",
            "source_id": "entity:E1",
            "direction": None,
            "query": "Where was Marie Curie born?",
            "top_k": 5,
        },
    }
    assert provider_model.model_validate(non_adjacent).action.direction is None
    with pytest.raises(ValidationError):
        provider_model.model_validate(
            {
                **common,
                "action": {**non_adjacent["action"], "direction": "NEXT"},
            }
        )


def _schema_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_schema_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_schema_keys(child))
        return keys
    return set()


def _schema_literal_values(value: object) -> set[str]:
    values: set[str] = set()
    if isinstance(value, dict):
        enum_values = value.get("enum")
        if isinstance(enum_values, list):
            values.update(item for item in enum_values if isinstance(item, str))
        const_value = value.get("const")
        if isinstance(const_value, str):
            values.add(const_value)
        for child in value.values():
            values.update(_schema_literal_values(child))
    elif isinstance(value, list):
        for child in value:
            values.update(_schema_literal_values(child))
    return values


def _assert_all_object_fields_required(value: object) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if value.get("type") == "object" and isinstance(properties, dict):
            assert set(properties) == set(value.get("required", []))
            assert value.get("additionalProperties") is False
        for child in value.values():
            _assert_all_object_fields_required(child)
    elif isinstance(value, list):
        for child in value:
            _assert_all_object_fields_required(child)


def test_provider_specific_expand_decision_normalizes_to_stable_contract() -> None:
    responses = _FakeResponses(
        policy_decision_model(DEFAULT_ENABLED_EXPANSIONS).model_validate(
            {
                "assessment": {
                    "status": "INSUFFICIENT",
                    "supported_facts": [],
                    "missing_information": ["country"],
                    "selected_evidence_refs": [],
                },
                "action": {
                    "type": "EXPAND",
                    "kind": "CHUNK_ADJACENT_CHUNK",
                    "source_id": "chunk:C4",
                    "direction": "NEXT",
                    "query": "What country was she born in?",
                    "top_k": 5,
                },
            }
        )
    )
    policy = OpenAIResponsesPolicy(
        client=SimpleNamespace(responses=responses),
        retry_backoff_seconds=0,
    )
    decision = policy.decide([Message(role="user", content="question")])
    assert isinstance(decision, PolicyDecision)
    assert isinstance(decision.action, ExpandAction)
    assert decision.action.kind is ExpansionKind.CHUNK_ADJACENT_CHUNK


def test_skill_document_hashes_exact_markdown_bytes(tmp_path: Path) -> None:
    content = "# Retrieval policy\n\nSearch with the full question.\n"
    path = tmp_path / "initial.md"
    path.write_bytes(content.encode("utf-8"))
    skill = SkillDocument.load(path)
    assert skill.content == content
    assert skill.sha256 == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert skill.version == skill.sha256

    snapshot = tmp_path / "snapshot.md"
    skill.write_snapshot(snapshot)
    assert snapshot.read_bytes() == content.encode("utf-8")


def test_context_replays_complete_trajectory_append_only() -> None:
    builder = AppendOnlyContextBuilder()
    skill = SkillDocument.from_text("# Strategy\nSearch first.")
    initial = ControllerState.initial(max_steps=3, max_retrieved_tokens=100)
    first_messages = builder.build(
        "Where was Marie Curie born?", skill, initial, []
    )

    decision = _search_decision()
    after = initial.model_copy(
        update={
            "step": 1,
            "remaining_step_budget": 2,
            "visible_sentence_ids": {"sentence:S12"},
            "eligible_sentence_ids": {"sentence:S12"},
            "visible_chunk_ids": {"chunk:C4"},
        },
        deep=True,
    )
    observation = Observation(
        status=ObservationStatus.OK,
        action=decision.action,
        results=[
            {
                "sentence_id": "sentence:S12",
                "text": "Marie Curie was born in Warsaw.",
                "parent_chunk_id": "chunk:C4",
                "document_id": "document:D1",
                "title": "Marie Curie",
            }
        ],
        retrieved_tokens=8,
    )
    record = StepRecord(
        step=1,
        decision=decision,
        validation_status=ValidationStatus.VALID,
        observation=observation,
        state_before=initial,
        state_after=after,
    )
    replayed = builder.build(
        "Where was Marie Curie born?", skill, after, [record]
    )

    assert [item.content for item in replayed[:3]] == [
        item.content for item in first_messages[:3]
    ]
    assert any("Marie Curie was born in Warsaw." in item.content for item in replayed)
    assert any('"remaining_step_budget":2' in item.content for item in replayed)


def test_context_lists_only_enabled_expansion_semantics() -> None:
    builder = AppendOnlyContextBuilder(
        (ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,)
    )
    messages = builder.build(
        "Where was Marie Curie born?",
        SkillDocument.from_text("# Strategy\nSearch first."),
        ControllerState.initial(),
        [],
    )
    protocol = messages[0].content
    assert "ENTITY_MENTIONED_IN_SENTENCE: visible Entity" in protocol
    assert "Returned Sentences are eligible evidence" in protocol
    assert "direction=null" in protocol
    assert "SENTENCE_MENTIONS_ENTITY" not in protocol
    assert "CHUNK_ADJACENT_CHUNK" not in protocol


def test_context_can_describe_every_ablation_expansion_individually() -> None:
    for enabled_kind in ExpansionKind:
        messages = AppendOnlyContextBuilder((enabled_kind,)).build(
            "Where was Marie Curie born?",
            SkillDocument.from_text("# Strategy\nUse local graph traversal."),
            ControllerState.initial(),
            [],
        )
        protocol = messages[0].content
        assert f"{enabled_kind.value}:" in protocol
        for disabled_kind in set(ExpansionKind) - {enabled_kind}:
            assert f"{disabled_kind.value}:" not in protocol


def test_scripted_policy_returns_typed_decisions_and_records_calls() -> None:
    expected = _search_decision()
    policy = ScriptedPolicy([expected])
    decision = policy.decide([Message(role="user", content="question")])
    assert decision == expected
    assert len(policy.calls) == 1
    assert policy.last_usage.policy_calls == 1


def test_invalid_scripted_policy_output_is_a_consumable_response_error() -> None:
    policy = ScriptedPolicy([{"not": "a decision"}])
    with pytest.raises(PolicyResponseError):
        policy.decide([Message(role="user", content="question")])


class _FakeResponses:
    def __init__(self, parsed: object) -> None:
        self.parsed = parsed
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=self.parsed,
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=4,
                total_tokens=14,
                output_tokens_details=SimpleNamespace(reasoning_tokens=2),
            ),
        )


class _InvalidStructuredResponses:
    def parse(self, **kwargs):
        return PolicyDecision.model_validate({"not": "a decision"})


def test_openai_policy_uses_responses_parse_and_store_false() -> None:
    responses = _FakeResponses(_search_decision())
    client = SimpleNamespace(responses=responses)
    policy = OpenAIResponsesPolicy(client=client, retry_backoff_seconds=0)
    result = policy.decide(
        [Message(role="user", content="Where was Marie Curie born?")]
    )
    assert result.action.type == "SEARCH"
    call = responses.calls[0]
    assert call["text_format"] is policy.decision_format
    assert call["reasoning"] == {"context": "current_turn"}
    assert call["store"] is False
    assert policy.last_usage.total_tokens == 14
    assert policy.last_usage.policy_input_tokens == 10
    assert policy.last_usage.policy_output_tokens == 4
    assert policy.last_usage.policy_reasoning_tokens == 2
    assert policy.last_usage.answer_input_tokens == 0


def test_openai_structured_parse_error_is_a_consumable_response_error() -> None:
    policy = OpenAIResponsesPolicy(
        client=SimpleNamespace(responses=_InvalidStructuredResponses()),
        retry_backoff_seconds=0,
    )
    with pytest.raises(PolicyResponseError):
        policy.decide([Message(role="user", content="question")])


def test_failed_openai_call_does_not_reuse_previous_usage() -> None:
    responses = _FakeResponses(_search_decision())
    policy = OpenAIResponsesPolicy(
        client=SimpleNamespace(responses=responses),
        max_retries=0,
        retry_backoff_seconds=0,
    )
    policy.decide([Message(role="user", content="first")])
    assert policy.last_usage.total_tokens == 14

    def fail(**kwargs):
        raise ConnectionError("offline")

    responses.parse = fail
    with pytest.raises(PolicyTransportError):
        policy.decide([Message(role="user", content="second")])
    assert policy.last_usage.total_tokens == 0
    assert policy.last_usage.policy_calls == 0


def test_answer_generators_receive_only_question_and_resolved_evidence() -> None:
    evidence = [
        ResolvedEvidence(
            ref=SentenceRef(id="sentence:S12"),
            text="Marie Curie was born in Warsaw.",
            document_id="document:D1",
            parent_chunk_id="chunk:C4",
            title="Marie Curie",
        )
    ]
    scripted = ScriptedAnswerGenerator("Warsaw")
    assert scripted.generate("Where was Marie Curie born?", evidence) == "Warsaw"
    assert scripted.calls[0][1] == evidence

    responses = _FakeResponses({"answer": "Warsaw"})
    generator = OpenAIResponsesAnswerGenerator(
        client=SimpleNamespace(responses=responses),
        retry_backoff_seconds=0,
    )
    assert generator.generate("Where was Marie Curie born?", evidence) == "Warsaw"
    call = responses.calls[0]
    assert call["store"] is False
    assert "Marie Curie was born in Warsaw." in call["input"][1]["content"]
    assert generator.last_usage.answer_input_tokens == 10
    assert generator.last_usage.answer_output_tokens == 4
    assert generator.last_usage.answer_reasoning_tokens == 2
    assert generator.last_usage.policy_input_tokens == 0
