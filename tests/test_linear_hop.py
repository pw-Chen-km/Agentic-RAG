import json
import pickle

from arag.core.context import AgentContext
from arag.tools.linear_graph_index import build_linear_graph, write_linear_graph
from arag.tools.linear_hop import LinearHopTool
from arag_skillopt.entityhop_metrics import episode_metrics
from arag_skillopt.observations import result_units


def _artifacts(tmp_path):
    chunks = tmp_path / "chunks.json"
    chunks.write_text(json.dumps([
        {"id": "c1", "text": "Alice met Bob."},
        {"id": "c2", "text": "Bob works with Carol."},
        {"id": "c3", "text": "Carol visited town."},
    ]), encoding="utf-8")
    sentence_index = tmp_path / "sentence_index.pkl"
    with sentence_index.open("wb") as stream:
        pickle.dump({
            "sentences": ["Alice met Bob", "Bob works with Carol", "Carol visited town"],
            "embeddings": None,
            "sentence_to_chunk": ["c1", "c2", "c3"],
            "model_name": "test-model",
        }, stream)
    entity_index = tmp_path / "entity_linked.json"
    entity_index.write_text(json.dumps({"entities": [
        {"entity_id": "e1", "canonical": "Alice", "aliases": ["Alice"], "chunk_ids": ["c1"]},
        {"entity_id": "e2", "canonical": "Bob", "aliases": ["Bob"], "chunk_ids": ["c1", "c2"]},
        {"entity_id": "e3", "canonical": "Carol", "aliases": ["Carol"], "chunk_ids": ["c2", "c3"]},
    ]}), encoding="utf-8")
    graph = tmp_path / "linear_trigraph.json"
    write_linear_graph(build_linear_graph(chunks, sentence_index, entity_index), graph)
    return chunks, graph


def test_linear_graph_hop_returns_snippets_and_activates_second_entity(tmp_path):
    _, graph = _artifacts(tmp_path)

    def ranker(query, sentence_ids):
        return {sentence_id: 1.0 for sentence_id in sentence_ids}

    tool = LinearHopTool(str(graph), sentence_ranker=ranker, max_hops=2, default_top_k=5)
    context = AgentContext(query="Who did Alice meet and where did that person work?")
    text, metadata = tool.execute(context, entity="Alice")

    assert metadata["status"] == "ok"
    assert metadata["max_hops"] == 2
    assert metadata["chunk_ids"] == ["c1", "c2"]
    assert "Alice met Bob" in text
    assert "Bob works with Carol" in text
    assert "e2" in metadata["activated_entities"]
    assert "Full content" not in text


def test_linear_hop_schema_exposes_bounded_defaults(tmp_path):
    _, graph = _artifacts(tmp_path)
    schema = LinearHopTool(str(graph), sentence_ranker=lambda q, ids: {i: 0.1 for i in ids}).get_schema()
    properties = schema["function"]["parameters"]["properties"]
    assert properties["max_hops"]["default"] == 2
    assert properties["top_k"]["default"] == 5
    assert "read_chunk" in schema["function"]["description"]


def test_linear_hop_result_is_parsed_as_snippet_and_metrics(tmp_path):
    _, graph = _artifacts(tmp_path)
    tool = LinearHopTool(str(graph), sentence_ranker=lambda q, ids: {i: 1.0 for i in ids})
    text, metadata = tool.execute(AgentContext(query="Alice"), entity="Alice", max_hops=1, top_k=1)
    units, status = result_units("linear_hop", text)
    assert status is None
    assert units[0]["chunk_id"] == "c1"
    metrics = episode_metrics({"trajectory": [{
        "loop": 1,
        "tool_name": "linear_hop",
        "arguments": {"entity": "Alice"},
        "tool_result": text,
        **metadata,
    }]})
    assert metrics["linear_hop_calls"] == 1
    assert metrics["entity_hop_calls"] == 0
    assert metrics["all_hop_calls"] == 1
