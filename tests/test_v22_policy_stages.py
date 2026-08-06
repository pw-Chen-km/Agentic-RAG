from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agentic_rag.agent.models import (
    ActionSelection,
    AssessmentStatus,
    ExpandParameters,
    FinishAction,
    FinishParameters,
    Message,
    PolicyStageRecord,
    SearchAction,
    SearchParameters,
    assemble_policy_decision,
)
from agentic_rag.agent.ollama import OllamaChatPolicy
from agentic_rag.agent.policy import (
    OpenAIResponsesPolicy,
    ScriptedPolicy,
    action_parameters_model,
)


def _selection(action_type: str = "SEARCH") -> ActionSelection:
    return ActionSelection(
        action_type=action_type,
        action_intent=(
            "Find a sentence that states Marie Curie's birthplace."
        ),
        selected_evidence_refs=[],
    )


def test_action_selection_is_high_level_and_evidence_is_unique() -> None:
    selection = _selection()
    assert selection.action_type == "SEARCH"
    assert set(ActionSelection.model_json_schema()["required"]) == {
        "action_type",
        "action_intent",
        "selected_evidence_refs",
    }

    with pytest.raises(ValidationError):
        ActionSelection.model_validate(
            {
                **selection.model_dump(mode="json"),
                "query": "this belongs to stage two",
            }
        )
    duplicate = {"unit": "SENTENCE", "id": "S4"}
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        ActionSelection(
            action_type="FINISH",
            action_intent="The accumulated evidence supports the answer.",
            selected_evidence_refs=[duplicate, duplicate],
        )


def test_action_parameter_schemas_omit_type_and_limit_expansions() -> None:
    search_schema = action_parameters_model("SEARCH").model_json_schema()
    assert "type" not in search_schema["properties"]
    assert set(search_schema["required"]) == {
        "query",
        "method",
        "target",
        "top_k",
    }

    expand_model = action_parameters_model(
        ["SENTENCE_MENTIONS_ENTITY"],
        "EXPAND",
    )
    schema_text = json.dumps(expand_model.model_json_schema())
    assert "SENTENCE_MENTIONS_ENTITY" in schema_text
    assert "ENTITY_MENTIONED_IN_SENTENCE" not in schema_text
    parameters = expand_model.model_validate(
        {
            "kind": "SENTENCE_MENTIONS_ENTITY",
            "source_id": "S4",
            "direction": None,
            "query": "Marie Curie",
            "top_k": 5,
        }
    )
    assert isinstance(parameters, ExpandParameters)
    with pytest.raises(ValidationError):
        expand_model.model_validate(
            {
                "kind": "ENTITY_MENTIONED_IN_SENTENCE",
                "source_id": "E2",
                "direction": None,
                "query": "Marie Curie",
                "top_k": 5,
            }
        )


def test_assemble_policy_decision_synthesizes_assessment() -> None:
    search = assemble_policy_decision(
        _selection(),
        SearchParameters(
            query="Marie Curie birthplace",
            method="BM25",
            target="SENTENCE",
            top_k=5,
        ),
    )
    assert isinstance(search.action, SearchAction)
    assert search.assessment.status is AssessmentStatus.INSUFFICIENT
    assert search.assessment.missing_information == [
        "Find a sentence that states Marie Curie's birthplace."
    ]

    ref = {"unit": "SENTENCE", "id": "S4"}
    finish_selection = ActionSelection(
        action_type="FINISH",
        action_intent="The evidence now supports the answer.",
        selected_evidence_refs=[ref],
    )
    finish = assemble_policy_decision(
        finish_selection,
        FinishParameters(answer="Warsaw", evidence_refs=[ref]),
    )
    assert isinstance(finish.action, FinishAction)
    assert finish.assessment.status is AssessmentStatus.SUFFICIENT
    assert finish.assessment.missing_information == []
    assert finish.assessment.selected_evidence_refs == (
        finish_selection.selected_evidence_refs
    )

    with pytest.raises(ValueError, match="SEARCH selection requires"):
        assemble_policy_decision(
            _selection(),
            FinishParameters(answer="Warsaw", evidence_refs=[ref]),
        )


def test_scripted_policy_generates_arbitrary_structured_stages() -> None:
    policy = ScriptedPolicy(
        [
            _selection().model_dump(mode="json"),
            {
                "query": "Marie Curie birthplace",
                "method": "BM25",
                "target": "SENTENCE",
                "top_k": 5,
            },
        ]
    )
    messages = [Message(role="user", content="Where was she born?")]

    selection = policy.generate_structured(messages, ActionSelection)
    assert isinstance(selection, ActionSelection)
    assert policy.last_usage.policy_calls == 1
    parameters = policy.generate_structured(messages, SearchParameters)
    assert isinstance(parameters, SearchParameters)
    assert policy.last_usage.policy_calls == 1
    assert len(policy.calls) == 2


class _FakeResponses:
    def __init__(self, parsed: object) -> None:
        self.parsed = parsed
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=self.parsed,
            usage=SimpleNamespace(
                input_tokens=8,
                output_tokens=3,
                total_tokens=11,
                output_tokens_details=SimpleNamespace(reasoning_tokens=1),
            ),
        )


def test_openai_policy_generic_structured_call_uses_requested_model() -> None:
    responses = _FakeResponses(_selection())
    policy = OpenAIResponsesPolicy(
        client=SimpleNamespace(responses=responses),
        retry_backoff_seconds=0,
    )

    result = policy.generate_structured(
        [Message(role="user", content="Where was she born?")],
        ActionSelection,
    )

    assert isinstance(result, ActionSelection)
    wire_model = responses.calls[0]["text_format"]
    assert wire_model is not ActionSelection
    schema = wire_model.model_json_schema()
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
    assert policy.last_usage.policy_calls == 1
    assert policy.last_usage.total_tokens == 11

    finish_responses = _FakeResponses(
        {
            "answer": "Warsaw",
            "evidence_refs": [{"unit": "SENTENCE", "id": "S1"}],
        }
    )
    finish_policy = OpenAIResponsesPolicy(
        client=SimpleNamespace(responses=finish_responses),
        retry_backoff_seconds=0,
    )
    finish = finish_policy.generate_structured(
        [Message(role="user", content="Finish")], FinishParameters
    )
    assert finish.answer == "Warsaw"
    finish_schema = finish_responses.calls[0]["text_format"].model_json_schema()
    assert not _schema_keys(finish_schema) & {
        "oneOf",
        "discriminator",
        "default",
    }
    _assert_all_object_fields_required(finish_schema)


class _FakeOllamaClient:
    def __init__(self, *responses: object) -> None:
        self.responses = deque(responses)
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.responses.popleft()


def test_ollama_policy_generic_structured_call_uses_requested_schema() -> None:
    response = {
        "message": {
            "role": "assistant",
            "content": _selection().model_dump_json(),
        },
        "prompt_eval_count": 6,
        "eval_count": 2,
    }
    client = _FakeOllamaClient(response)
    policy = OllamaChatPolicy(
        model="qwen3.5:9b",
        client=client,
        think=False,
        retry_backoff_seconds=0,
    )

    result = policy.generate_structured(
        [Message(role="user", content="Where was she born?")],
        ActionSelection,
    )

    assert isinstance(result, ActionSelection)
    assert client.calls[0]["format"] == ActionSelection.model_json_schema()
    assert client.calls[0]["think"] is False
    assert policy.last_usage.policy_calls == 1
    assert policy.last_usage.total_tokens == 8


def test_policy_stage_record_is_backward_compatible_and_auditable() -> None:
    record = PolicyStageRecord(
        phase="action_selection",
        messages=[Message(role="user", content="question")],
        output=_selection().model_dump(mode="json"),
        action_type="SEARCH",
        disclosed_skill_paths=["SKILL.md"],
        disclosed_skill_hashes={"SKILL.md": "abc123"},
    )
    assert record.phase == "action_selection"
    assert record.usage.policy_calls == 0
    assert record.disclosed_skill_hashes == {"SKILL.md": "abc123"}


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
