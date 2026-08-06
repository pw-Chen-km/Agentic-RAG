from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_rag.agent.answer import AnswerGenerationError
from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (
    Message,
    PolicyDecision,
    ResolvedEvidence,
    SentenceRef,
    V31PolicyDecision,
    V3PolicyDecision,
)
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.storage import Substrate
from agentic_rag.agent.ollama import (
    OllamaChatAnswerGenerator,
    OllamaChatPolicy,
)
from agentic_rag.agent.policy import (
    PolicyConfigurationError,
    PolicyResponseError,
    PolicyTransportError,
)


@pytest.fixture(autouse=True)
def _isolate_ollama_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


class _FakeClient:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = deque(outcomes)
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if not self.outcomes:
            raise AssertionError("fake Ollama client has no response")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _ResponseError(Exception):
    def __init__(
        self, status_code: int, error: str | None = None
    ) -> None:
        detail = error or f"HTTP {status_code}"
        super().__init__(detail)
        self.status_code = status_code
        self.error = detail


def _decision_payload() -> dict[str, object]:
    return {
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


def _finish_payload(sentence_id: str) -> dict[str, object]:
    evidence_ref = {"unit": "SENTENCE", "id": sentence_id}
    return {
        "assessment": {
            "status": "SUFFICIENT",
            "supported_facts": [
                "Marie Curie was born in Warsaw."
            ],
            "missing_information": [],
            "selected_evidence_refs": [evidence_ref],
        },
        "action": {
            "type": "FINISH",
            "evidence_refs": [evidence_ref],
        },
    }


def _chat_response(
    content: str,
    *,
    prompt_tokens: int = 12,
    output_tokens: int = 7,
    thinking: str | None = None,
) -> dict[str, object]:
    message: dict[str, object] = {
        "role": "assistant",
        "content": content,
    }
    if thinking is not None:
        message["thinking"] = thinking
    return {
        "model": "qwen3:8b",
        "message": message,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": prompt_tokens,
        "eval_count": output_tokens,
    }


def test_ollama_policy_sends_native_structured_chat_and_maps_usage() -> None:
    client = _FakeClient(
        _chat_response(
            json.dumps(_decision_payload()), thinking="private trace"
        )
    )
    policy = OllamaChatPolicy(
        model="qwen3:8b",
        client=client,
        temperature=0.25,
        num_ctx=32_768,
        think=False,
        keep_alive="10m",
        retry_backoff_seconds=0,
    )
    messages = [
        Message(role="system", content="Follow the action protocol."),
        {"role": "user", "content": "Where was Marie Curie born?"},
    ]

    decision = policy.decide(messages)

    assert isinstance(decision, PolicyDecision)
    assert decision.action.type == "SEARCH"
    assert decision.action.query == "Where was Marie Curie born?"
    assert client.calls == [
        {
            "model": "qwen3:8b",
            "messages": [
                {
                    "role": "system",
                    "content": "Follow the action protocol.",
                },
                {
                    "role": "user",
                    "content": "Where was Marie Curie born?",
                },
            ],
            "stream": False,
            "format": policy.decision_format.model_json_schema(),
            "options": {"temperature": 0.25, "num_ctx": 32_768},
            "think": False,
            "keep_alive": "10m",
        }
    ]
    # There is no separate reasoning-token count in the native API.
    assert policy.last_usage.model_dump() == {
        "policy_calls": 1,
        "answer_calls": 0,
        "input_tokens": 12,
        "output_tokens": 7,
        "reasoning_tokens": 0,
        "total_tokens": 19,
        "retrieved_tokens": 0,
        "policy_input_tokens": 12,
        "policy_output_tokens": 7,
        "policy_reasoning_tokens": 0,
        "answer_input_tokens": 0,
        "answer_output_tokens": 0,
        "answer_reasoning_tokens": 0,
    }


def test_ollama_policy_parses_v3_semantic_memory_decision() -> None:
    client = _FakeClient(
        _chat_response(json.dumps(_v3_decision_payload()))
    )
    policy = OllamaChatPolicy(
        model="qwen3:8b",
        client=client,
        semantic_memory_v3=True,
        direct_answer=True,
    )

    decision = policy.decide(
        [Message(role="user", content="Where was Marie Curie born?")]
    )

    assert isinstance(decision, V3PolicyDecision)
    assert decision.action.type == "SEARCH"
    schema = json.dumps(policy.decision_format.model_json_schema())
    assert "source_context_index" in schema
    assert "chunk_context_index" in schema
    assert "citations" in schema
    assert "selected_evidence_refs" not in schema


def test_ollama_policy_parses_v31_typed_reference_decision() -> None:
    client = _FakeClient(_chat_response(json.dumps(_v3_decision_payload())))
    policy = OllamaChatPolicy(
        model="qwen3:8b",
        client=client,
        semantic_memory_v31=True,
        direct_answer=True,
    )

    decision = policy.decide(
        [Message(role="user", content="Where was Marie Curie born?")]
    )

    assert isinstance(decision, V31PolicyDecision)
    schema = json.dumps(policy.decision_format.model_json_schema())
    assert "source_ref" in schema
    assert "chunk_ref" in schema
    assert "evidence_refs" in schema
    assert "source_context_index" not in schema
    assert "citations" not in schema


def test_ollama_policy_forwards_explicit_thinking_level() -> None:
    client = _FakeClient(_chat_response(json.dumps(_decision_payload())))
    policy = OllamaChatPolicy(
        model="qwen3:8b",
        client=client,
        think="low",
        retry_backoff_seconds=0,
    )

    policy.decide([Message(role="user", content="question")])

    assert client.calls[0]["think"] == "low"
    assert client.calls[0]["options"] == {
        "temperature": 0.0,
        "num_ctx": 32_768,
    }


def test_ollama_policy_omits_thinking_only_when_none() -> None:
    client = _FakeClient(_chat_response(json.dumps(_decision_payload())))
    policy = OllamaChatPolicy(
        model="qwen3:8b",
        client=client,
        think=None,
        retry_backoff_seconds=0,
    )

    policy.decide([Message(role="user", content="question")])

    assert "think" not in client.calls[0]


def test_ollama_policy_invalid_json_is_a_response_error_with_usage() -> None:
    client = _FakeClient(
        _chat_response("not-json", prompt_tokens=4, output_tokens=2)
    )
    policy = OllamaChatPolicy(
        model="qwen3:8b",
        client=client,
        retry_backoff_seconds=0,
    )

    with pytest.raises(PolicyResponseError):
        policy.decide([Message(role="user", content="question")])

    assert len(client.calls) == 1
    assert policy.last_usage.policy_calls == 1
    assert policy.last_usage.total_tokens == 6


def test_ollama_policy_does_not_retry_missing_model() -> None:
    client = _FakeClient(
        _ResponseError(404, "model 'missing' not found"),
        _chat_response(json.dumps(_decision_payload())),
    )
    policy = OllamaChatPolicy(
        model="missing",
        client=client,
        max_retries=2,
        retry_backoff_seconds=0,
    )

    with pytest.raises(PolicyTransportError) as raised:
        policy.decide([Message(role="user", content="question")])

    assert len(client.calls) == 1
    assert policy.last_usage == policy.last_usage.__class__()
    assert "model 'missing' not found" in str(raised.value)
    assert "ollama pull missing" in str(raised.value)


@pytest.mark.parametrize(
    "first_failure",
    [
        _ResponseError(429),
        _ResponseError(500),
        _ResponseError(502),
        ConnectionError("Ollama is offline"),
    ],
)
def test_ollama_policy_retries_transient_failures(
    first_failure: Exception,
) -> None:
    client = _FakeClient(
        first_failure,
        _chat_response(json.dumps(_decision_payload())),
    )
    policy = OllamaChatPolicy(
        model="qwen3:8b",
        client=client,
        max_retries=1,
        retry_backoff_seconds=0,
    )

    assert policy.decide(
        [Message(role="user", content="question")]
    ).action.type == "SEARCH"
    assert len(client.calls) == 2
    assert policy.last_usage.policy_calls == 1


def test_ollama_policy_bounds_transient_retries() -> None:
    client = _FakeClient(
        ConnectionError("offline"),
        ConnectionError("offline"),
        ConnectionError("offline"),
        _chat_response(json.dumps(_decision_payload())),
    )
    policy = OllamaChatPolicy(
        model="qwen3:8b",
        client=client,
        max_retries=2,
        retry_backoff_seconds=0,
    )

    with pytest.raises(PolicyTransportError):
        policy.decide([Message(role="user", content="question")])

    assert len(client.calls) == 3


def test_ollama_answer_is_structured_and_receives_only_evidence() -> None:
    client = _FakeClient(
        SimpleNamespace(
            message=SimpleNamespace(
                role="assistant", content='{"answer":"Warsaw"}'
            ),
            prompt_eval_count=20,
            eval_count=3,
        )
    )
    generator = OllamaChatAnswerGenerator(
        model="qwen3:8b",
        client=client,
        num_ctx=16_384,
        think=True,
        keep_alive=0,
        retry_backoff_seconds=0,
    )
    evidence = [
        ResolvedEvidence(
            ref=SentenceRef(id="sentence:S12"),
            text="Marie Curie was born in Warsaw.",
            document_id="document:D1",
            parent_chunk_id="chunk:C4",
            title="Marie Curie",
        )
    ]

    answer = generator.generate(
        "Where was Marie Curie born?", evidence
    )

    assert answer == "Warsaw"
    call = client.calls[0]
    assert call["stream"] is False
    assert call["think"] is True
    assert call["keep_alive"] == 0
    assert call["options"] == {
        "temperature": 0.0,
        "num_ctx": 16_384,
    }
    assert call["format"]["properties"]["answer"]["type"] == "string"
    wire_user_payload = json.loads(call["messages"][1]["content"])
    assert set(wire_user_payload) == {"question", "evidence"}
    assert wire_user_payload["question"] == "Where was Marie Curie born?"
    assert wire_user_payload["evidence"] == [
        item.model_dump(mode="json") for item in evidence
    ]
    assert generator.last_usage.answer_calls == 1
    assert generator.last_usage.total_tokens == 23
    assert generator.last_usage.answer_input_tokens == 20
    assert generator.last_usage.answer_output_tokens == 3
    assert generator.last_usage.answer_reasoning_tokens == 0
    assert generator.last_usage.policy_input_tokens == 0


def test_ollama_answer_rejects_invalid_or_empty_outputs() -> None:
    evidence = [
        ResolvedEvidence(
            ref=SentenceRef(id="sentence:S12"),
            text="Marie Curie was born in Warsaw.",
            document_id="document:D1",
        )
    ]
    invalid = OllamaChatAnswerGenerator(
        model="qwen3:8b",
        client=_FakeClient(_chat_response("not-json")),
        retry_backoff_seconds=0,
    )
    empty = OllamaChatAnswerGenerator(
        model="qwen3:8b",
        client=_FakeClient(_chat_response("")),
        retry_backoff_seconds=0,
    )

    with pytest.raises(AnswerGenerationError):
        invalid.generate("Where was Marie Curie born?", evidence)
    with pytest.raises(AnswerGenerationError):
        empty.generate("Where was Marie Curie born?", evidence)


def test_ollama_answer_retries_transient_failures() -> None:
    client = _FakeClient(
        _ResponseError(500),
        _chat_response('{"answer":"Warsaw"}'),
    )
    generator = OllamaChatAnswerGenerator(
        model="qwen3:8b",
        client=client,
        max_retries=1,
        retry_backoff_seconds=0,
    )
    evidence = [
        ResolvedEvidence(
            ref=SentenceRef(id="sentence:S12"),
            text="Marie Curie was born in Warsaw.",
            document_id="document:D1",
        )
    ]

    assert (
        generator.generate("Where was Marie Curie born?", evidence)
        == "Warsaw"
    )
    assert len(client.calls) == 2
    assert generator.last_usage.answer_calls == 1


def test_ollama_answer_does_not_retry_missing_model() -> None:
    client = _FakeClient(
        _ResponseError(404, "provider says model is missing"),
        _chat_response('{"answer":"Warsaw"}'),
    )
    generator = OllamaChatAnswerGenerator(
        model="missing",
        client=client,
        max_retries=2,
        retry_backoff_seconds=0,
    )
    evidence = [
        ResolvedEvidence(
            ref=SentenceRef(id="sentence:S12"),
            text="Marie Curie was born in Warsaw.",
            document_id="document:D1",
        )
    ]

    with pytest.raises(AnswerGenerationError) as raised:
        generator.generate("Where was Marie Curie born?", evidence)

    assert len(client.calls) == 1
    assert generator.last_usage.answer_calls == 0
    assert "provider says model is missing" in str(raised.value)
    assert "ollama pull missing" in str(raised.value)


def test_ollama_adapter_normalizes_model_and_explicit_host() -> None:
    policy = OllamaChatPolicy(
        model="  qwen3:8b  ",
        host="  localhost:11434/  ",
        client=_FakeClient(),
    )

    assert policy.model == "qwen3:8b"
    assert policy.host == "http://localhost:11434"


@pytest.mark.parametrize(
    "host",
    [
        "https://ollama.com",
        "https://api.ollama.com/",
        "https://private.api.ollama.com",
    ],
)
def test_ollama_adapter_rejects_explicit_cloud_hosts(
    host: str,
) -> None:
    with pytest.raises(
        PolicyConfigurationError,
        match="Cloud does not support.*structured outputs",
    ):
        OllamaChatPolicy(
            model="qwen3:8b",
            host=host,
            client=_FakeClient(),
        )


def test_ollama_adapter_rejects_cloud_host_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OLLAMA_HOST", "  https://cloud.api.ollama.com/  "
    )

    with pytest.raises(
        AnswerGenerationError,
        match="Cloud does not support.*structured outputs",
    ):
        OllamaChatAnswerGenerator(
            model="qwen3:8b",
            client=_FakeClient(),
        )


@pytest.mark.parametrize(
    "host",
    [
        "ftp://localhost:11434",
        "file:///tmp/ollama.sock",
        "ssh://localhost",
    ],
)
def test_ollama_adapter_rejects_non_http_host_scheme(
    host: str,
) -> None:
    with pytest.raises(
        PolicyConfigurationError,
        match="scheme must be http or https",
    ):
        OllamaChatPolicy(
            model="qwen3:8b",
            host=host,
            client=_FakeClient(),
        )


@pytest.mark.parametrize(
    "host",
    [
        "http://local host:11434",
        "http://user:secret@localhost:11434",
        "http://localhost:11434?token=secret",
        "http://localhost:11434#fragment",
    ],
)
def test_ollama_adapter_rejects_unsafe_host_components(
    host: str,
) -> None:
    with pytest.raises(
        PolicyConfigurationError,
        match="must not contain whitespace, credentials",
    ):
        OllamaChatPolicy(
            model="qwen3:8b",
            host=host,
            client=_FakeClient(),
        )


def test_ollama_transport_error_preserves_provider_detail() -> None:
    policy = OllamaChatPolicy(
        model="qwen3:8b",
        client=_FakeClient(
            _ResponseError(500, "runner process exited unexpectedly")
        ),
        max_retries=0,
        retry_backoff_seconds=0,
    )

    with pytest.raises(
        PolicyTransportError,
        match="runner process exited unexpectedly",
    ):
        policy.decide([Message(role="user", content="question")])


def test_ollama_adapters_run_real_controller_and_write_artifacts(
    built_substrate: Path,
    fake_embedder: Any,
    tmp_path: Path,
) -> None:
    question = "Where was Marie Curie born?"
    substrate = Substrate.open(built_substrate)
    sentence_id = next(
        sentence.sentence_id
        for sentence in substrate.sentences
        if sentence.text == "Marie Curie was born in Warsaw."
    )
    policy_client = _FakeClient(
        _chat_response(
            json.dumps(_decision_payload()),
            prompt_tokens=10,
            output_tokens=4,
        ),
        _chat_response(
            json.dumps(_finish_payload(sentence_id)),
            prompt_tokens=20,
            output_tokens=5,
        ),
    )
    answer_client = _FakeClient(
        _chat_response(
            '{"answer":"Warsaw"}',
            prompt_tokens=8,
            output_tokens=2,
        )
    )
    policy = OllamaChatPolicy(
        model="qwen3:8b",
        client=policy_client,
        max_retries=0,
    )
    answer_generator = OllamaChatAnswerGenerator(
        model="qwen3:8b",
        client=answer_client,
        max_retries=0,
    )
    output_root = tmp_path / "runs"
    harness = AgentHarness(
        substrate=substrate,
        config=AgentConfig(
            policy={
                "provider": "ollama",
                "model": "qwen3:8b",
            },
            answer={
                "provider": "ollama",
                "model": "qwen3:8b",
            },
        ),
        skill=SkillDocument.from_text(
            "# Initial retrieval strategy\n\n"
            "Search sentences with the complete question.\n"
        ),
        policy=policy,
        answer_generator=answer_generator,
        output_root=output_root,
        embedding_backend=fake_embedder,
    )

    result = harness.run(
        question,
        "q1",
        episode_id="ollama-controller-e2e",
    )

    assert result.termination_reason == "finish"
    assert result.answer == "Warsaw"
    assert [
        record.decision.action.type for record in result.trajectory
    ] == ["SEARCH", "FINISH"]
    assert (
        result.trajectory[0].decision.action.query
        == "Where was Marie Curie born?"
    )
    assert question in policy_client.calls[0]["messages"][2]["content"]
    assert result.usage.policy_calls == 2
    assert result.usage.answer_calls == 1
    assert result.usage.input_tokens == 38
    assert result.usage.output_tokens == 11
    assert result.usage.total_tokens == 49
    assert result.usage.policy_input_tokens == 30
    assert result.usage.policy_output_tokens == 9
    assert result.usage.policy_reasoning_tokens == 0
    assert result.usage.answer_input_tokens == 8
    assert result.usage.answer_output_tokens == 2
    assert result.usage.answer_reasoning_tokens == 0

    run_dir = output_root / "ollama-controller-e2e"
    assert {path.name for path in run_dir.iterdir()} == {
        "episode.json",
        "conversation.json",
        "target_system_prompt.txt",
        "target_user_prompt.txt",
        "skill.md",
        "effective_config.json",
    }
    effective = json.loads(
        (run_dir / "effective_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert effective["runtime_components"] == {
        "policy_client": "OllamaChatPolicy",
        "answer_generator": "OllamaChatAnswerGenerator",
        "state_manager": "EpisodeStateManager",
        "controller_role": "stateless_loop_orchestrator",
    }


def _v3_decision_payload() -> dict[str, object]:
    return {
        "assessment": {
            "status": "INSUFFICIENT",
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
    persisted_episode = json.loads(
        (run_dir / "episode.json").read_text(encoding="utf-8")
    )
    assert persisted_episode["query"] == question
    assert persisted_episode["answer"] == "Warsaw"
    assert persisted_episode["usage"]["total_tokens"] == 49


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"model": ""}, PolicyConfigurationError),
        ({"model": "qwen3", "num_ctx": 0}, PolicyConfigurationError),
        ({"model": "qwen3", "think": "max"}, PolicyConfigurationError),
        (
            {"model": "qwen3", "keep_alive": True},
            PolicyConfigurationError,
        ),
        (
            {"model": "qwen3", "keep_alive": float("inf")},
            PolicyConfigurationError,
        ),
        (
            {"model": "qwen3", "keep_alive": float("nan")},
            PolicyConfigurationError,
        ),
    ],
)
def test_ollama_policy_rejects_invalid_configuration(
    kwargs: dict[str, object], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        OllamaChatPolicy(client=_FakeClient(), **kwargs)
