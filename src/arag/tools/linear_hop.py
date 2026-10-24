"""LinearRAG-style entity graph hopping for the A-RAG action interface.

The Agent chooses an entity and bounded hop parameters.  This tool performs
the graph traversal, ranks bridge sentences with the original question, and
returns only sentence snippets plus chunk IDs.  ``read_chunk`` remains the
separate action that exposes full passage text.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool
from arag.tools.bm25_index import BM25Index
from arag.tools.linear_graph_index import load_linear_graph

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
except ImportError:  # pragma: no cover - regular runtime requires it
    tiktoken = None


SentenceRanker = Callable[[str, Iterable[str]], Dict[str, float]]


class LinearHopTool(BaseTool):
    """Bounded question-sentence similarity traversal over a corpus graph."""

    def __init__(
        self,
        graph_index_file: str,
        *,
        sentence_index_file: str | None = None,
        sentence_ranker: SentenceRanker | None = None,
        top_entities: int = 3,
        default_top_k: int = 5,
        max_top_k: int = 20,
        default_max_hops: int = 2,
        max_hops: int = 2,
        entity_source: str = "question_or_visible_evidence",
    ) -> None:
        if entity_source not in {"visible_evidence", "question_or_visible_evidence"}:
            raise ValueError(f"Unsupported LinearHop entity source: {entity_source}")
        self.graph_index_file = graph_index_file
        self.graph = load_linear_graph(graph_index_file)
        if sentence_index_file:
            actual_hash = hashlib.sha256(Path(sentence_index_file).read_bytes()).hexdigest()
            expected_hash = self.graph.get("source", {}).get("sentence_index_sha256")
            if expected_hash and actual_hash != expected_hash:
                raise ValueError("LinearHop graph was built from a different sentence index")
        self.sentences = [str(value) for value in self.graph["sentences"]]
        self.sentence_to_chunk = [str(value) for value in self.graph["sentence_to_chunk"]]
        self.sentence_to_entities = [
            [str(value) for value in values]
            for values in self.graph["sentence_to_entities"]
        ]
        self.entity_to_sentences = {
            str(key): [str(value) for value in values]
            for key, values in self.graph["entity_to_sentences"].items()
        }
        self.entity_records = {
            str(record["entity_id"]): record
            for record in self.graph.get("entities", [])
        }
        self.entity_index = BM25Index(self.graph.get("entity_docs", {}))
        self.sentence_lexical_index = BM25Index(
            {str(index): sentence for index, sentence in enumerate(self.sentences)}
        )
        self.sentence_ranker = sentence_ranker
        self.top_entities = max(1, int(top_entities))
        self.default_top_k = max(1, int(default_top_k))
        self.max_top_k = max(self.default_top_k, int(max_top_k))
        self.default_max_hops = max(1, int(default_max_hops))
        self.max_hops = max(self.default_max_hops, int(max_hops))
        self.entity_source = entity_source
        if tiktoken is None:
            raise ImportError("tiktoken required for LinearHop")
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    @property
    def name(self) -> str:
        return "linear_hop"

    def get_schema(self) -> Dict[str, Any]:
        source = "the question or visible evidence" if self.entity_source == "question_or_visible_evidence" else "visible evidence"
        timing = (
            "This can be the first retrieval or a follow-up on a newly visible entity; no prior search or read is required. "
            if self.entity_source == "question_or_visible_evidence" else
            "Use this only after a search or read has exposed the entity. "
        )
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    f"Find passages connected to a concrete entity from {source}. "
                    + timing +
                    "It finds sentences mentioning the entity, ranks them by relevance to the original question, "
                    "then follows other entities in those sentences to related passages. "
                    "This may find passages that do not mention the starting entity; you do not need to know the connecting entity in advance. "
                    "The result contains abbreviated snippets and Chunk IDs; always use read_chunk on useful results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": f"A concrete entity from {source} to expand through the corpus graph.",
                        },
                        "max_hops": {
                            "type": "integer",
                            "description": f"Maximum graph expansion depth (default {self.default_max_hops}, maximum {self.max_hops}).",
                            "default": self.default_max_hops,
                        },
                        "top_k": {
                            "type": "integer",
                            "description": f"Maximum chunks/snippets to return (default {self.default_top_k}, maximum {self.max_top_k}).",
                            "default": self.default_top_k,
                        },
                    },
                    "required": ["entity"],
                },
            },
        }

    def _rank_sentences(self, query: str, sentence_ids: Iterable[str]) -> dict[str, float]:
        ids = sorted({str(value) for value in sentence_ids})
        if not ids:
            return {}
        if self.sentence_ranker is not None:
            scores = self.sentence_ranker(query, ids)
            return {sid: float(scores.get(sid, 0.0)) for sid in ids}
        # A deterministic fallback keeps synthetic/offline tests usable.  A
        # production config always wires the existing semantic index ranker.
        return {sid: float(score) for sid, score in self.sentence_lexical_index.search(query, len(ids), allowed_ids=ids)}

    def _seed_entities(self, query: str, entity: str) -> tuple[dict[str, float], list[str]]:
        matches = self.entity_index.search(entity, self.top_entities)
        seed_scores = {entity_id: max(float(score), 1.0) for entity_id, score in matches}
        if seed_scores:
            return seed_scores, []
        # A question can contain an entity surface that the corpus NER did not
        # index exactly.  Similarity-ranked seed sentences still activate the
        # same graph entities without inventing a direct entity/entity edge.
        scores = self._rank_sentences(query, [str(i) for i in range(len(self.sentences))])
        seed_sentences = [sid for sid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: self.top_k_seed]]
        for sid in seed_sentences:
            for entity_id in self.sentence_to_entities[int(sid)]:
                seed_scores[entity_id] = max(seed_scores.get(entity_id, 0.0), float(scores.get(sid, 0.0)))
        return seed_scores, seed_sentences

    @property
    def top_k_seed(self) -> int:
        return max(10, self.default_top_k * 2)

    def _snippet(self, sentence: str) -> str:
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if len(sentence) > 600:
            sentence = sentence[:597].rstrip() + "..."
        return f"... {sentence} ..." if sentence else "(empty sentence)"

    def execute(
        self,
        context: "AgentContext",
        entity: str,
        max_hops: int | None = None,
        top_k: int | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        entity = str(entity or "").strip()
        if not entity:
            return "Error: entity is required", {"error": "entity_required", "retrieved_tokens": 0}
        hop_limit = min(max(1, int(max_hops or self.default_max_hops)), self.max_hops)
        result_limit = min(max(1, int(top_k or self.default_top_k)), self.max_top_k)
        query = str(getattr(context, "query", None) or entity)
        seed_entities, seed_sentences = self._seed_entities(query, entity)
        metadata: dict[str, Any] = {
            "entity": entity,
            "query": query,
            "hop_backend": "linear_rag_trigraph",
            "graph_version": self.graph["version"],
            "max_hops": hop_limit,
            "top_k": result_limit,
            "top_entities": self.top_entities,
            "seed_entities": sorted(seed_entities),
            "seed_sentences": seed_sentences,
            "similarity_mode": "question_sentence" if self.sentence_ranker is not None else "bm25_fallback",
        }
        if not seed_entities:
            result = f"No graph entities found for entity: {entity}"
            metadata.update({"status": "no_entity_match", "chunks_found": 0})
            context.add_retrieval_log(self.name, 0, metadata)
            return result, {"retrieved_tokens": 0, **metadata}

        active_entities = seed_entities
        seen_sentences: set[str] = set()
        seen_entities: set[str] = set(seed_entities)
        selected: list[dict[str, Any]] = []
        for hop in range(1, hop_limit + 1):
            candidate_sources: dict[str, list[str]] = {}
            for source_entity, source_score in active_entities.items():
                for sentence_id in self.entity_to_sentences.get(source_entity, []):
                    if sentence_id in seen_sentences:
                        continue
                    candidate_sources.setdefault(sentence_id, []).append(source_entity)
            if not candidate_sources:
                break
            sentence_scores = self._rank_sentences(query, candidate_sources)
            ranked = sorted(
                candidate_sources,
                key=lambda sid: (-sentence_scores.get(sid, 0.0) * max(active_entities.get(source, 0.0) for source in candidate_sources[sid]), sid),
            )[:result_limit]
            next_entities: dict[str, float] = {}
            for sentence_id in ranked:
                source_ids = candidate_sources[sentence_id]
                similarity = float(sentence_scores.get(sentence_id, 0.0))
                path_score = similarity * max(active_entities.get(source, 0.0) for source in source_ids)
                target_ids = self.sentence_to_entities[int(sentence_id)]
                selected.append({
                    "sentence_id": sentence_id,
                    "chunk_id": self.sentence_to_chunk[int(sentence_id)],
                    "sentence": self.sentences[int(sentence_id)],
                    "hop": hop,
                    "similarity": similarity,
                    "path_score": path_score,
                    "source_entity_ids": sorted(source_ids),
                    "activated_entity_ids": sorted(target_ids),
                })
                seen_sentences.add(sentence_id)
                for target_id in target_ids:
                    if target_id not in seen_entities:
                        next_entities[target_id] = max(next_entities.get(target_id, 0.0), path_score)
            if not next_entities:
                break
            seen_entities.update(next_entities)
            active_entities = next_entities

        selected.sort(key=lambda row: (-row["path_score"], int(row["hop"]), row["sentence_id"]))
        output_rows: list[dict[str, Any]] = []
        used_chunks: set[str] = set()
        for row in selected:
            if row["chunk_id"] in used_chunks:
                continue
            used_chunks.add(row["chunk_id"])
            output_rows.append(row)
            if len(output_rows) >= result_limit:
                break
        if not output_rows:
            result = f"No graph sentences found for entity: {entity}"
            metadata.update({"status": "no_entity_match", "chunks_found": 0})
            context.add_retrieval_log(self.name, 0, metadata)
            return result, {"retrieved_tokens": 0, **metadata}

        parts = []
        snippets = []
        for row in output_rows:
            activated = ", ".join(row["activated_entity_ids"]) or "none"
            sources = ", ".join(row["source_entity_ids"]) or "none"
            snippet = self._snippet(row["sentence"])
            snippets.append(snippet)
            parts.append(
                f"Chunk ID: {row['chunk_id']} (LinearHop score: {row['path_score']:.3f}, hop: {row['hop']})\n"
                f"Bridge: {sources} -> sentence {row['sentence_id']} -> entities {activated}\n"
                f"Matched: {snippet}"
            )
        metadata.update({
            "status": "ok",
            "chunks_found": len(output_rows),
            "chunk_ids": [row["chunk_id"] for row in output_rows],
            "sentence_ids": [row["sentence_id"] for row in output_rows],
            "activated_entities": sorted({entity_id for row in output_rows for entity_id in row["activated_entity_ids"]}),
            "hop_trace": output_rows,
        })
        retrieved_tokens = len(self.tokenizer.encode("\n".join(snippets)))
        context.add_retrieval_log(self.name, retrieved_tokens, metadata)
        return "\n\n".join(parts), {"retrieved_tokens": retrieved_tokens, **metadata}
