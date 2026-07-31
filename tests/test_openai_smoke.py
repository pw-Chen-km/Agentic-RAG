from __future__ import annotations

import os

import pytest

from agentic_rag.agent.answer import (
    AnswerGenerationError,
    OpenAIResponsesAnswerGenerator,
)
from agentic_rag.agent.models import Message, ResolvedEvidence, SentenceRef
from agentic_rag.agent.policy import (
    OpenAIResponsesPolicy,
    PolicyTransportError,
)


pytestmark = pytest.mark.openai_smoke


@pytest.mark.skipif(
    os.environ.get("RUN_OPENAI_SMOKE") != "1",
    reason="Set RUN_OPENAI_SMOKE=1 to call the OpenAI API",
)
def test_luna_policy_and_answer_structured_outputs_roundtrip() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")

    policy = OpenAIResponsesPolicy(
        model="gpt-5.6-luna",
        max_retries=0,
    )
    try:
        decision = policy.decide(
            [
                Message(
                    role="system",
                    content=(
                        "Return one PolicyDecision. For this schema smoke test, "
                        "choose SEARCH with method=BM25, target=SENTENCE, top_k=5, "
                        'and query="Where was Marie Curie born?". Assess the '
                        "evidence as INSUFFICIENT and provide all required fields."
                    ),
                ),
                Message(
                    role="user",
                    content="Where was Marie Curie born?",
                ),
            ]
        )
    except PolicyTransportError as exc:
        if _is_network_unavailable(exc):
            pytest.skip(f"OpenAI network is unavailable: {exc}")
        raise

    assert decision.action.type == "SEARCH"
    assert decision.action.query == "Where was Marie Curie born?"
    assert decision.action.method.value == "BM25"
    assert decision.action.target.value == "SENTENCE"

    evidence = [
        ResolvedEvidence(
            ref=SentenceRef(id="sentence:S12"),
            text="Marie Curie was born in Warsaw.",
            document_id="document:D1",
            parent_chunk_id="chunk:C4",
            title="Marie Curie",
        )
    ]
    generator = OpenAIResponsesAnswerGenerator(
        model="gpt-5.6-luna",
        max_retries=0,
    )
    try:
        answer = generator.generate(
            "Where was Marie Curie born?",
            evidence,
        )
    except AnswerGenerationError as exc:
        if _is_network_unavailable(exc):
            pytest.skip(f"OpenAI network is unavailable: {exc}")
        raise

    assert "warsaw" in answer.casefold()


def _is_network_unavailable(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, (ConnectionError, TimeoutError)):
            return True
        if type(current).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
        }:
            return True
        current = current.__cause__
    return False
