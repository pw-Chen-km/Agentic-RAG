from __future__ import annotations

from agentic_rag.agent.models import Message
from agentic_rag.agent.providers.ollama import OllamaChatPolicy
from agentic_rag.agent.providers.openai_compatible import OpenAICompatibleChatPolicy
from agentic_rag.agent.tool_calling import build_tool_definitions
from agentic_rag.agent.models import AvailableActionSpace, SearchActionOption, SearchMethod, SearchTarget
from agentic_rag.agent.action_schema import InterfaceDecisionSchemaBuilder


def _tools() -> list[dict]:
    return build_tool_definitions(
        AvailableActionSpace(
            search_options=(
                SearchActionOption(method=SearchMethod.DENSE, target=SearchTarget.CHUNK),
            ),
            finish_available=True,
        )
    )


def _single_decision_model():
    return InterfaceDecisionSchemaBuilder().build(
        AvailableActionSpace(
            search_options=(SearchActionOption(method=SearchMethod.DENSE, target=SearchTarget.CHUNK),),
            finish_available=True,
        )
    )


class _FakeOllama:
    def __init__(self) -> None:
        self.request: dict = {}

    def chat(self, **kwargs):
        self.request = kwargs
        return {
            "message": {
                "tool_calls": [
                    {
                        "id": "ollama-call-1",
                        "function": {
                            "name": "find_passages",
                            "arguments": {"assessment": {"supported_facts": [], "missing_information": ["Where Marie Curie was born"]}, "query": "Marie Curie"},
                        },
                    }
                ]
            },
            "prompt_eval_count": 12,
            "eval_count": 4,
        }


def test_ollama_uses_native_tools_without_json_format() -> None:
    client = _FakeOllama()
    policy = OllamaChatPolicy(model="qwen3.5:4b", client=client)
    decision = policy.decide([Message(role="user", content="Question")], tools=_tools())
    assert decision.action.type == "SEARCH"
    assert "tools" in client.request
    assert "format" not in client.request
    assert client.request["tools"][0]["function"]["name"] == "find_passages"
    assert policy.last_usage_metadata["native_tool_calling"] is True


def test_openai_compatible_uses_single_native_tool_call() -> None:
    policy = OpenAICompatibleChatPolicy(
        model="Qwen/Qwen3.8-27B-FP8", base_url="http://localhost:8000/v1"
    )
    captured: dict = {}

    def fake_request(payload):
        captured.update(payload)
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "vllm-call-1",
                                "type": "function",
                                "function": {
                                    "name": "find_passages",
                                    "arguments": '{"assessment":{"supported_facts":[],"missing_information":["Where Marie Curie was born"]},"query":"Marie Curie"}',
                                },
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }

    policy._request = fake_request  # type: ignore[method-assign]
    decision = policy.decide([Message(role="user", content="Question")], tools=_tools())
    assert decision.action.type == "SEARCH"
    assert "tools" in captured and captured["tool_choice"] == "required"
    assert "response_format" not in captured
    assert captured["parallel_tool_calls"] is False


def test_ollama_constrains_one_flattened_decision_without_native_tools() -> None:
    class Client:
        request = None
        def chat(self, **kwargs):
            self.request = kwargs
            return {"message": {"content": '{"supported_facts":[],"missing_information":["birthplace"],"action":{"name":"find_passages","query":"Marie Curie birthplace"}}'},
                    "prompt_eval_count": 12, "eval_count": 8}
    client = Client()
    policy = OllamaChatPolicy(model="qwen3.5:4b", client=client)
    decision = policy.decide([Message(role="user", content="Question")],
                             decision_format=_single_decision_model(), tools=None)
    assert decision.action.type == "SEARCH"
    assert decision.assessment.missing_information == ["birthplace"]
    assert "format" in client.request and "tools" not in client.request
    assert policy.last_usage_metadata["constrained_single_decision"] is True
    assert policy.last_usage_metadata["decision_count"] == 1


def test_openai_compatible_constrains_one_flattened_decision() -> None:
    policy = OpenAICompatibleChatPolicy(model="Qwen/Qwen3.8-27B-FP8", base_url="http://localhost:8000/v1")
    captured = {}
    def fake_request(payload):
        captured.update(payload)
        return {"choices": [{"message": {"content": '{"supported_facts":[],"missing_information":[],"action":{"name":"finish","answer":"Unknown","evidence_refs":[]}}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    policy._request = fake_request  # type: ignore[method-assign]
    decision = policy.decide([Message(role="user", content="Question")],
                             decision_format=_single_decision_model(), tools=None)
    assert decision.action.type == "FINISH"
    assert "response_format" in captured and "tools" not in captured
    assert policy.last_usage_metadata["decision_count"] == 1
