import pickle
from typing import ClassVar

import numpy as np
import pytest

from arag.tools import semantic_search as module


class FakeEmbedding:
    calls: ClassVar[list] = []

    def __init__(self, name, device=None):
        self.prompts = {"query": "retrieval instruction"}

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, texts, **options):
        self.calls.append((texts, options))
        return np.array([[1.0, 0.0]])


def tool(tmp_path, monkeypatch, index_model="qwen", index_prompt="query"):
    monkeypatch.setattr(module, "HAS_SENTENCE_TRANSFORMERS", True)
    monkeypatch.setattr(module, "SentenceTransformer", FakeEmbedding, raising=False)
    data = {
        "model_name": index_model,
        "query_prompt_name": index_prompt,
        "sentences": ["Alice was born in York"],
        "embeddings": np.array([[1.0, 0.0]]),
        "sentence_to_chunk": ["31"],
        "chunks": {"31": {"id": "31", "text": "Alice was born in York."}},
    }
    (tmp_path / "sentence_index.pkl").write_bytes(pickle.dumps(data))
    return module.SemanticSearchTool(
        "unused", str(tmp_path), "qwen", query_prompt_name="query"
    )


def test_qwen_query_uses_prompt_but_keeps_original_tool_format(tmp_path, monkeypatch):
    from arag.core.context import AgentContext

    search = tool(tmp_path, monkeypatch)
    result, metadata = search.execute(AgentContext(), "Where was Alice born?")
    assert "Chunk ID: 31" in result and "Alice was born in York" in result
    assert metadata["chunks_found"] == 1
    assert FakeEmbedding.calls[-1][1] == {
        "normalize_embeddings": True,
        "prompt_name": "query",
    }


def test_wrong_embedding_index_or_prompt_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="different model"):
        tool(tmp_path, monkeypatch, index_model="minilm")
    with pytest.raises(ValueError, match="Query prompt"):
        tool(tmp_path, monkeypatch, index_prompt=None)
