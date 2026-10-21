"""Build and load a LinearRAG-style corpus graph.

The graph is an offline corpus artifact.  It keeps the existing A-RAG chunk
corpus and sentence embedding index as the source of truth, then adds explicit
sentence/entity and entity/entity links.  Questions, answers and gold evidence
are intentionally not accepted by the builder.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


VERSION = "linear-rag-trigraph-v1"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(value.split()).strip(" \t\r\n.,;:()[]{}")


def _contains(surface: str, text: str) -> bool:
    needle = _norm(surface)
    haystack = _norm(text)
    if len(needle) < 2 or not haystack:
        return False
    # Word boundaries avoid linking an entity such as ``Art`` to ``article``.
    pattern = rf"(?<!\w){re.escape(needle)}(?!\w)"
    return re.search(pattern, haystack, flags=re.UNICODE) is not None


def _load_chunks(path: str | Path) -> list[dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("chunks file must contain a JSON list")
    rows = []
    for index, row in enumerate(payload):
        if isinstance(row, Mapping):
            rows.append({"id": str(row.get("id", index)), "text": str(row.get("text", ""))})
        else:
            value = str(row)
            chunk_id, separator, text = value.partition(":")
            rows.append({"id": chunk_id if separator else str(index), "text": text if separator else value})
    return rows


def _load_sentence_index(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    required = {"sentences", "sentence_to_chunk"}
    if not required.issubset(data):
        raise ValueError(f"sentence index missing keys: {sorted(required - set(data))}")
    if len(data["sentences"]) != len(data["sentence_to_chunk"]):
        raise ValueError("sentence index sentence/mapping lengths differ")
    return data


def _load_entity_index(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or not isinstance(data.get("entities"), list):
        raise ValueError("entity index must contain an entities list")
    return data


def build_linear_graph(
    chunks_file: str | Path,
    sentence_index_file: str | Path,
    entity_index_file: str | Path,
) -> dict[str, Any]:
    """Build a deterministic sentence/entity graph from corpus-only artifacts."""

    chunks = _load_chunks(chunks_file)
    chunk_ids = {row["id"] for row in chunks}
    sentence_index = _load_sentence_index(sentence_index_file)
    entity_index = _load_entity_index(entity_index_file)
    sentences = [str(item) for item in sentence_index["sentences"]]
    sentence_to_chunk = [str(item) for item in sentence_index["sentence_to_chunk"]]
    if any(chunk_id not in chunk_ids for chunk_id in sentence_to_chunk):
        raise ValueError("sentence index contains a chunk absent from chunks_file")

    entities: list[dict[str, Any]] = []
    chunk_to_entities: dict[str, list[str]] = defaultdict(list)
    entity_docs: dict[str, str] = {}
    for record in entity_index["entities"]:
        entity_id = str(record["entity_id"])
        aliases = sorted({str(x) for x in record.get("aliases", []) if str(x).strip()})
        canonical = str(record.get("canonical", ""))
        if not aliases and canonical:
            aliases = [canonical]
        attached_chunks = sorted({str(x) for x in record.get("chunk_ids", []) if str(x) in chunk_ids})
        if not attached_chunks or not aliases:
            continue
        clean = {
            "entity_id": entity_id,
            "canonical": canonical,
            "aliases": aliases,
            "entity_type": record.get("entity_type"),
            "chunk_ids": attached_chunks,
        }
        entities.append(clean)
        entity_docs[entity_id] = " ".join([canonical, *aliases])
        for chunk_id in attached_chunks:
            chunk_to_entities[chunk_id].append(entity_id)

    entities.sort(key=lambda item: item["entity_id"])
    entities_by_id = {item["entity_id"]: item for item in entities}
    sentence_to_entities: list[list[str]] = [[] for _ in sentences]
    entity_to_sentences: dict[str, list[str]] = {item["entity_id"]: [] for item in entities}
    for sentence_number, (sentence, chunk_id) in enumerate(zip(sentences, sentence_to_chunk)):
        for entity_id in chunk_to_entities.get(chunk_id, []):
            entity = entities_by_id[entity_id]
            if any(_contains(alias, sentence) for alias in entity["aliases"]):
                sentence_id = str(sentence_number)
                sentence_to_entities[sentence_number].append(entity_id)
                entity_to_sentences[entity_id].append(sentence_id)
        sentence_to_entities[sentence_number].sort()

    # Passage-local adjacency is retained as a deterministic fallback bridge.
    sentence_neighbors: dict[str, list[str]] = {str(i): [] for i in range(len(sentences))}
    chunk_sentence_ids: dict[str, list[str]] = defaultdict(list)
    for sentence_number, chunk_id in enumerate(sentence_to_chunk):
        chunk_sentence_ids[chunk_id].append(str(sentence_number))
    for ids in chunk_sentence_ids.values():
        for position, sentence_id in enumerate(ids):
            neighbors = []
            if position:
                neighbors.append(ids[position - 1])
            if position + 1 < len(ids):
                neighbors.append(ids[position + 1])
            sentence_neighbors[sentence_id] = neighbors

    entity_neighbors: dict[str, dict[str, int]] = {item["entity_id"]: {} for item in entities}
    for ids in sentence_to_entities:
        for left in ids:
            for right in ids:
                if left == right:
                    continue
                entity_neighbors[left][right] = entity_neighbors[left].get(right, 0) + 1

    graph = {
        "version": VERSION,
        "normalization": "NFKC + casefold + whitespace + surrounding punctuation",
        "source": {
            "chunks_file_sha256": _sha256(chunks_file),
            "sentence_index_sha256": _sha256(sentence_index_file),
            "entity_index_sha256": _sha256(entity_index_file),
            "chunk_count": len(chunks),
            "sentence_count": len(sentences),
            "entity_count": len(entities),
            "questions_used": False,
            "answers_used": False,
            "gold_evidence_used": False,
        },
        "embedding": {
            "model_name": sentence_index.get("model_name"),
            "query_prompt_name": sentence_index.get("query_prompt_name"),
            "sentence_index_file": str(Path(sentence_index_file).resolve()),
        },
        "sentences": sentences,
        "sentence_to_chunk": sentence_to_chunk,
        "sentence_to_entities": sentence_to_entities,
        "entity_to_sentences": entity_to_sentences,
        "sentence_neighbors": sentence_neighbors,
        "entity_neighbors": entity_neighbors,
        "entities": entities,
        "entity_docs": entity_docs,
    }
    canonical = json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    graph["graph_sha256"] = hashlib.sha256(canonical).hexdigest()
    return graph


def write_linear_graph(graph: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_linear_graph(path: str | Path) -> dict[str, Any]:
    graph = json.loads(Path(path).read_text(encoding="utf-8"))
    if graph.get("version") != VERSION:
        raise ValueError(f"Unsupported LinearRAG graph version: {graph.get('version')!r}")
    required = {"sentences", "sentence_to_chunk", "sentence_to_entities", "entity_to_sentences", "entities"}
    if not required.issubset(graph):
        raise ValueError(f"LinearRAG graph missing keys: {sorted(required - set(graph))}")
    count = len(graph["sentences"])
    if any(len(graph[key]) != count for key in ("sentence_to_chunk", "sentence_to_entities")):
        raise ValueError("LinearRAG graph sentence arrays have inconsistent lengths")
    return graph
