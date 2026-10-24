"""Evidence-conditioned entity hopping for the A-RAG-derived variant."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool
from arag.tools.bm25_index import BM25Index, tokenize

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
except ImportError:  # pragma: no cover - the regular runtime requires it
    tiktoken = None


def load_chunks(chunks_file: str | Path) -> list[dict[str, str]]:
    rows = json.loads(Path(chunks_file).read_text(encoding="utf-8"))
    if rows and isinstance(rows[0], dict):
        return [{"id": str(row["id"]), "text": str(row.get("text", ""))} for row in rows]
    output = []
    for item in rows:
        parts = str(item).split(":", 1)
        if len(parts) == 2:
            output.append({"id": parts[0], "text": parts[1]})
    return output


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?\n]+", text) if part.strip()]


class EntityHopTool(BaseTool):
    """Search chunks through either direct BM25 or a hidden entity substrate."""

    def __init__(
        self,
        chunks_file: str,
        *,
        backend: str,
        index_file: str | None = None,
        entity_index_file: str | None = None,
        top_entities: int = 3,
        default_top_k: int = 5,
        max_top_k: int = 20,
        entity_source: str = "visible_evidence",
    ) -> None:
        if backend not in {"direct_bm25", "entity_linked"}:
            raise ValueError(f"Unsupported EntityHop backend: {backend}")
        self.backend = backend
        if entity_source not in {"visible_evidence", "question_or_visible_evidence"}:
            raise ValueError(f"Unsupported EntityHop entity source: {entity_source}")
        self.entity_source = entity_source
        self.chunks = load_chunks(chunks_file)
        self.chunks_by_id = {row["id"]: row["text"] for row in self.chunks}
        self.chunk_index = BM25Index(self.chunks)
        if index_file:
            self._assert_index_matches(index_file, self.chunk_index)
        self.top_entities = max(1, int(top_entities))
        self.default_top_k = max(1, int(default_top_k))
        self.max_top_k = max(1, int(max_top_k))
        self.entity_index: BM25Index | None = None
        self.entity_chunks: dict[str, list[str]] = {}
        self.entity_aliases: dict[str, list[str]] = {}
        if backend == "entity_linked":
            if not entity_index_file:
                raise ValueError("entity_linked EntityHop requires entity_index_file")
            self._load_entity_index(entity_index_file)
        if tiktoken is None:
            raise ImportError("tiktoken required for EntityHop")
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    @staticmethod
    def _assert_index_matches(path: str, expected: BM25Index) -> None:
        index = BM25Index.load(path)
        if index.doc_ids != expected.doc_ids or index.texts != expected.texts:
            raise ValueError("EntityHop BM25 index does not match chunks_file")

    def _load_entity_index(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != "entityhop-entity-link-v1":
            raise ValueError(f"Unsupported entity index version: {data.get('version')!r}")
        records = data.get("entities", [])
        docs = {}
        for record in records:
            entity_id = str(record["entity_id"])
            aliases = [str(alias) for alias in record.get("aliases", [])]
            chunk_ids = [str(cid) for cid in record.get("chunk_ids", []) if str(cid) in self.chunks_by_id]
            if not chunk_ids:
                continue
            docs[entity_id] = " ".join(aliases or [str(record.get("canonical", ""))])
            self.entity_chunks[entity_id] = sorted(set(chunk_ids))
            self.entity_aliases[entity_id] = aliases
        self.entity_index = BM25Index(docs)

    @property
    def name(self) -> str:
        return "entity_hop"

    def get_schema(self) -> Dict[str, Any]:
        # This ablation changes only the public policy instruction. Like the
        # original tool, it does not add a new grounding validator or filter.
        question_allowed = self.entity_source == "question_or_visible_evidence"
        description = (
            "Find passages about a concrete entity from the question or visible evidence. "
            "This can be the first retrieval or a follow-up on a newly visible entity; no prior search or read is required. "
            if question_allowed else
            "Find passages about a concrete entity from visible evidence. "
            "Use this only after a search or read has exposed the entity. "
        )
        method = (
            "It matches the entity's words against passage text using BM25, which is useful for finding passages that directly mention the entity. "
            if self.backend == "direct_bm25" else
            "It matches the entity name or aliases to the corpus entity index, then searches the passages linked to those matches. "
        )
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    description + method +
                    "The result contains abbreviated snippets and Chunk IDs; always use read_chunk on useful results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": (
                                "A concrete person, organization, place, work, or other entity from the question or visible evidence."
                                if question_allowed else
                                "A concrete person, organization, place, work, or other entity from visible evidence."
                            ),
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

    def _snippet(self, chunk_id: str, entity: str) -> str:
        text = self.chunks_by_id[chunk_id]
        query_terms = set(tokenize(entity))
        sentences = split_sentences(text)
        matched = [sentence for sentence in sentences if query_terms.intersection(tokenize(sentence))]
        selected = matched[:5] or sentences[:1]
        return "... " + " ... ".join(selected) + " ..." if selected else "(empty chunk)"

    def _render(self, ranked: list[tuple[str, float]], entity: str) -> tuple[str, int]:
        parts = []
        snippets = []
        for chunk_id, score in ranked:
            snippet = self._snippet(chunk_id, entity)
            parts.append(f"Chunk ID: {chunk_id}, Matched entity in chunk: {snippet}")
            snippets.append(snippet)
        text = "\n\n".join(parts)
        return text, len(self.tokenizer.encode("\n".join(snippets))) if snippets else 0

    def execute(self, context: 'AgentContext', entity: str, top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        entity = str(entity).strip()
        top_k = min(max(1, int(top_k or self.default_top_k)), self.max_top_k)
        metadata: dict[str, Any] = {
            "entity": entity,
            "hop_backend": self.backend,
            "top_entities": self.top_entities if self.backend == "entity_linked" else 0,
        }
        allowed_ids = None
        linked_query = entity
        if self.backend == "entity_linked":
            assert self.entity_index is not None
            candidates = self.entity_index.search(entity, self.top_entities)
            metadata["entity_candidates_internal"] = len(candidates)
            metadata["entity_candidate_scores"] = [score for _, score in candidates]
            candidate_terms: list[str] = []
            for candidate_id, _ in candidates:
                allowed_ids = (set(allowed_ids or ()) | set(self.entity_chunks.get(candidate_id, [])))
                candidate_terms.extend(self.entity_aliases.get(candidate_id, []))
            # An alias is often not literally present in the chunk text (for
            # example, the substrate maps ``NYC`` to chunks that say
            # ``New York City``).  The linked backend must still return the
            # corpus evidence attached to the matched entity.  Expand the
            # reranking query with the candidate's corpus-only canonical and
            # alias surfaces; this does not expose the surfaces to the agent.
            if candidate_terms:
                linked_query = " ".join([entity, *candidate_terms])
            metadata["candidate_chunk_count"] = len(allowed_ids or ())
            if not allowed_ids:
                result = f"No entity-linked chunks found for entity: {entity}"
                metadata["status"] = "no_entity_match"
                context.add_retrieval_log(self.name, 0, metadata)
                return result, {"retrieved_tokens": 0, "chunks_found": 0, **metadata}
        ranked = self.chunk_index.search(linked_query, top_k, allowed_ids=allowed_ids)
        if self.backend == "entity_linked" and allowed_ids and not ranked:
            # Keep deterministic linked-corpus behavior for a candidate whose
            # mention is absent from the chunk text.  This is not a direct
            # BM25 fallback: the set is still restricted to linked postings.
            ranked = [(chunk_id, 0.0) for chunk_id in sorted(allowed_ids)[:top_k]]
        if not ranked:
            result = f"No results found for entity: {entity}"
            metadata["status"] = "no_entity_match"
            context.add_retrieval_log(self.name, 0, metadata)
            return result, {"retrieved_tokens": 0, "chunks_found": 0, **metadata}
        result, retrieved_tokens = self._render(ranked, entity)
        chunk_ids = [chunk_id for chunk_id, _ in ranked]
        metadata.update({"status": "ok", "chunks_found": len(chunk_ids), "chunk_ids": chunk_ids})
        context.add_retrieval_log(self.name, retrieved_tokens, metadata)
        return result, {"retrieved_tokens": retrieved_tokens, **metadata}
