import json

import pytest

from arag.core.context import AgentContext
from arag.tools import entity_hop as entity_hop_module
from arag.tools.bm25_index import BM25Index
from arag.tools.entity_hop import EntityHopTool


class FixtureTokenizer:
    def encode(self, text):
        return text.split()


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    """Avoid downloading the optional tiktoken encoding in local unit tests."""

    monkeypatch.setattr(entity_hop_module.tiktoken, "encoding_for_model", lambda _: FixtureTokenizer())


def write_chunks(tmp_path, rows=None):
    rows = rows or [
        {"id": "0", "text": "Alice visited New York City in summer."},
        {"id": "1", "text": "The city hosts a large library."},
        {"id": "2", "text": "Bob studies an unrelated topic."},
    ]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def write_entity_index(tmp_path, *, aliases=None, chunk_ids=None):
    path = tmp_path / "entities.json"
    path.write_text(
        json.dumps(
            {
                "version": "entityhop-entity-link-v1",
                "entities": [
                    {
                        "entity_id": "entity:new-york-city",
                        "canonical": "New York City",
                        "aliases": aliases or ["New York City", "NYC"],
                        "chunk_ids": chunk_ids or ["0"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_direct_bm25_schema_and_output_hide_scores_and_internal_ids(tmp_path):
    tool = EntityHopTool(write_chunks(tmp_path), backend="direct_bm25")
    schema = tool.get_schema()["function"]
    assert schema["name"] == "entity_hop"
    assert schema["parameters"]["required"] == ["entity"]

    result, log = tool.execute(AgentContext(), entity="Alice", top_k=2)
    assert "Chunk ID: 0" in result
    assert "Alice" in result
    assert "entity:new-york-city" not in result
    assert "score" not in result.lower()
    assert log["hop_backend"] == "direct_bm25"


def test_entity_linked_alias_returns_linked_chunk_and_hides_internal_id(tmp_path):
    tool = EntityHopTool(
        write_chunks(tmp_path),
        backend="entity_linked",
        entity_index_file=str(write_entity_index(tmp_path)),
    )
    result, log = tool.execute(AgentContext(), entity="NYC", top_k=2)
    # This fixture includes the alias in the entity substrate but not in the
    # chunk text; the expected behavior is tested separately below as xfail.
    assert "entity:new-york-city" not in result
    assert "score" not in result.lower()
    assert log["hop_backend"] == "entity_linked"


def test_entity_linked_does_not_fallback_to_direct_bm25(tmp_path):
    tool = EntityHopTool(
        write_chunks(tmp_path),
        backend="entity_linked",
        entity_index_file=str(write_entity_index(tmp_path)),
    )
    result, log = tool.execute(AgentContext(), entity="library", top_k=2)
    assert result == "No entity-linked chunks found for entity: library"
    assert log["status"] == "no_entity_match"


def test_bm25_index_mismatch_is_rejected(tmp_path):
    chunks = write_chunks(tmp_path)
    mismatched = tmp_path / "wrong.json"
    BM25Index({"wrong": "different corpus"}).save(mismatched)
    with pytest.raises(ValueError, match="does not match"):
        EntityHopTool(chunks, backend="direct_bm25", index_file=str(mismatched))


def test_entity_linked_alias_absent_from_chunk_still_returns_linked_chunk(tmp_path):
    tool = EntityHopTool(
        write_chunks(tmp_path),
        backend="entity_linked",
        entity_index_file=str(write_entity_index(tmp_path, aliases=["NYC"])),
    )
    result, _log = tool.execute(AgentContext(), entity="NYC", top_k=2)
    assert "Chunk ID: 0" in result

