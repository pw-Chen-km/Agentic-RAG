from __future__ import annotations

import os

import pytest

from agentic_rag.agent.models import Message, ResolvedEvidence, SentenceRef
from agentic_rag.agent.ollama import (
    OllamaChatAnswerGenerator,
    OllamaChatPolicy,
)


pytestmark = pytest.mark.ollama_smoke


@pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_SMOKE") != "1",
    reason="Set RUN_OLLAMA_SMOKE=1 to call a running local Ollama server",
)
def test_local_ollama_policy_and_answer_structured_outputs_roundtrip() -> None:
    model = os.environ.get("OLLAMA_MODEL")
    if not model:
        pytest.skip("Set OLLAMA_MODEL to an already installed local model")

    from ollama import Client

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    client = Client(host=host, timeout=30)
    try:
        installed = client.list()
    except ConnectionError:
        pytest.skip(f"Ollama is not reachable at {host}")

    installed_names = {
        name
        for item in (_value(installed, "models") or [])
        for name in (
            _value(item, "model"),
            _value(item, "name"),
        )
        if isinstance(name, str)
    }
    if model not in installed_names:
        pytest.skip(
            f"Ollama model {model!r} is not installed; this test never pulls models"
        )

    policy = OllamaChatPolicy(
        model=model,
        client=client,
        max_retries=0,
        num_ctx=32_768,
    )
    decision = policy.decide(
        [
            Message(
                role="system",
                content=(
                    "Return one PolicyDecision. Choose SEARCH with method=BM25, "
                    "target=SENTENCE, top_k=5, and query exactly "
                    '"Where was Marie Curie born?". Use assessment status '
                    "INSUFFICIENT and populate every required list."
                ),
            ),
            Message(
                role="user",
                content="Where was Marie Curie born?",
            ),
        ]
    )
    assert decision.action.type == "SEARCH"
    assert decision.action.query == "Where was Marie Curie born?"

    evidence = [
        ResolvedEvidence(
            ref=SentenceRef(id="sentence:S12"),
            text="Marie Curie was born in Warsaw.",
            document_id="document:D1",
            parent_chunk_id="chunk:C4",
            title="Marie Curie",
        )
    ]
    answer = OllamaChatAnswerGenerator(
        model=model,
        client=client,
        max_retries=0,
        num_ctx=32_768,
    ).generate("Where was Marie Curie born?", evidence)
    assert "warsaw" in answer.casefold()


def _value(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
