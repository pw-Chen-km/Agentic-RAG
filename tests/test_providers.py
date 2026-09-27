from __future__ import annotations

import json
from types import SimpleNamespace

from agentic_rag.agent.models import Message
from agentic_rag.agent.action_schema import ActionSchemaBuilder, native_action_tools
from agentic_rag.agent.action_space import AvailableActionSpaceBuilder
from agentic_rag.agent.models import ContextReferenceMap, EpisodeState
from agentic_rag.agent.providers.ollama import OllamaChatPolicy
from agentic_rag.agent.providers.openai import OpenAIResponsesPolicy


PAYLOAD = {
    "assessment": {
        "supported_facts": [],
        "missing_information": ["birthplace"],
    },
    "action": {
        "type": "SEARCH",
        "query": "Where was Marie Curie born?",
        "method": "BM25",
        "target": "SENTENCE",
        "top_k": 5,
    },
}


class FakeResponses:
    def __init__(self) -> None:
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        parsed = kwargs["text_format"].model_validate(PAYLOAD)
        return SimpleNamespace(output_parsed=parsed, usage=None, output=[])


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class FakeOllamaClient:
    def __init__(self) -> None:
        self.kwargs = None

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return {
            "message": {"content": json.dumps(PAYLOAD)},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }


class FakeNativeOllamaClient:
    def __init__(self, *, arguments: dict | None = None) -> None:
        self.kwargs = None
        self.arguments = arguments or {
            "assessment": {
                "supported_facts": [],
                "missing_information": ["birthplace"],
            },
            "action": PAYLOAD["action"],
        }

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": json.dumps(self.arguments),
                        }
                    }
                ]
            },
            "prompt_eval_count": 10,
            "eval_count": 5,
        }


def test_openai_and_ollama_use_the_same_action_contract() -> None:
    messages = [Message(role="user", content="question")]
    openai_client = FakeOpenAIClient()
    ollama_client = FakeOllamaClient()
    openai = OpenAIResponsesPolicy(client=openai_client, max_retries=0)
    ollama = OllamaChatPolicy(
        model="qwen3.5:9b", client=ollama_client, max_retries=0
    )

    assert openai.decide(messages) == ollama.decide(messages)
    openai_schema = openai_client.responses.kwargs["text_format"].model_json_schema()
    ollama_schema = ollama_client.kwargs["format"]
    assert openai_schema == ollama_schema
    serialized = json.dumps(ollama_schema)
    for action in ("SEARCH", "EXPAND", "READ", "FINISH"):
        assert action in serialized


def test_ollama_receives_the_exact_state_conditioned_schema() -> None:
    client = FakeOllamaClient()
    policy = OllamaChatPolicy(
        model="qwen3.5:9b", client=client, max_retries=0
    )
    action_space = AvailableActionSpaceBuilder(()).build(
        EpisodeState.initial(), ContextReferenceMap()
    )
    decision_format = ActionSchemaBuilder().build(action_space)

    policy.decide(
        [Message(role="user", content="question")],
        decision_format=decision_format,
    )

    assert client.kwargs["format"] == decision_format.model_json_schema()
    serialized = json.dumps(client.kwargs["format"])
    assert "SEARCH" in serialized
    assert all(item not in serialized for item in ("EXPAND", "READ", "FINISH"))


def test_ollama_native_tools_omit_structured_format() -> None:
    client = FakeNativeOllamaClient()
    policy = OllamaChatPolicy(
        model="qwen3.5:9b",
        client=client,
        output_mode="native_tools",
        max_retries=0,
    )
    action_space = AvailableActionSpaceBuilder(()).build(
        EpisodeState.initial(), ContextReferenceMap()
    )
    tools, tool_models = native_action_tools(action_space)

    decision = policy.decide(
        [Message(role="user", content="question")],
        decision_format=ActionSchemaBuilder().build(action_space),
        tools=tools,
        tool_models=tool_models,
    )

    assert decision.action.type == "SEARCH"
    assert client.kwargs["tools"] == tools
    assert "format" not in client.kwargs
    assert client.kwargs["options"] == {"temperature": 0.0, "num_ctx": 32_768}


def test_ollama_native_tools_decode_nested_json_arguments() -> None:
    client = FakeNativeOllamaClient(
        arguments={
            "assessment": json.dumps(
                {"supported_facts": [], "missing_information": ["birthplace"]}
            ),
            "action": json.dumps(PAYLOAD["action"]),
        }
    )
    policy = OllamaChatPolicy(
        model="qwen3.5:9b",
        client=client,
        output_mode="native_tools",
        max_retries=0,
    )
    action_space = AvailableActionSpaceBuilder(()).build(
        EpisodeState.initial(), ContextReferenceMap()
    )
    tools, tool_models = native_action_tools(action_space)

    decision = policy.decide(
        [Message(role="user", content="question")],
        tools=tools,
        tool_models=tool_models,
    )

    assert decision.action.query == PAYLOAD["action"]["query"]
