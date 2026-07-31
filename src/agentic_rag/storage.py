"""Parquet persistence and read-only substrate access."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq

from agentic_rag.errors import NodeNotFoundError, ScopeNotFoundError, SubstrateNotFoundError
from agentic_rag.models import (
    BuildManifest,
    Chunk,
    ChunkRead,
    Document,
    DocumentScope,
    Entity,
    EntityAlias,
    GoldSupport,
    Mention,
    Sentence,
    SentencePreview,
    SentenceResult,
)

T = TypeVar("T")

SCHEMAS: dict[str, pa.Schema] = {
    "documents": pa.schema(
        [pa.field("doc_id", pa.string(), False), pa.field("title", pa.string(), True)]
    ),
    "chunks": pa.schema(
        [
            pa.field("chunk_id", pa.string(), False),
            pa.field("doc_id", pa.string(), False),
            pa.field("chunk_pos", pa.int64(), False),
            pa.field("text", pa.string(), False),
        ]
    ),
    "sentences": pa.schema(
        [
            pa.field("sentence_id", pa.string(), False),
            pa.field("chunk_id", pa.string(), False),
            pa.field("sentence_pos", pa.int64(), False),
            pa.field("text", pa.string(), False),
        ]
    ),
    "entities": pa.schema(
        [
            pa.field("entity_id", pa.string(), False),
            pa.field("canonical_name", pa.string(), False),
            pa.field("normalized_name", pa.string(), False),
            pa.field("entity_type", pa.string(), True),
        ]
    ),
    "mentions": pa.schema(
        [
            pa.field("mention_id", pa.string(), False),
            pa.field("entity_id", pa.string(), False),
            pa.field("sentence_id", pa.string(), False),
            pa.field("surface_form", pa.string(), False),
            pa.field("mention_start", pa.int64(), False),
            pa.field("mention_end", pa.int64(), False),
        ]
    ),
    "entity_aliases": pa.schema(
        [
            pa.field("alias_id", pa.string(), False),
            pa.field("entity_id", pa.string(), False),
            pa.field("alias", pa.string(), False),
            pa.field("normalized_alias", pa.string(), False),
            pa.field("alias_type", pa.string(), False),
            pa.field("source_sentence_id", pa.string(), True),
        ]
    ),
    "document_scopes": pa.schema(
        [
            pa.field("scope_id", pa.string(), False),
            pa.field("doc_id", pa.string(), False),
        ]
    ),
    "gold_support": pa.schema(
        [
            pa.field("scope_id", pa.string(), False),
            pa.field("doc_id", pa.string(), False),
            pa.field("title", pa.string(), False),
            pa.field("source_sentence_pos", pa.int64(), False),
            pa.field("source_sentence_text", pa.string(), False),
            pa.field("sentence_id", pa.string(), True),
        ]
    ),
    "benchmark_questions": pa.schema(
        [
            pa.field("question_id", pa.string(), False),
            pa.field("scope_id", pa.string(), False),
            pa.field("source", pa.string(), False),
            pa.field("question", pa.string(), False),
            pa.field("answer", pa.string(), False),
            pa.field("question_type", pa.string(), True),
        ]
    ),
}


def write_records(path: Path, records: Iterable[Any], schema_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    schema = SCHEMAS[schema_name]
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SubstrateNotFoundError(f"Required substrate table is missing: {path}")
    return pq.read_table(path).to_pylist()


def _construct(cls: type[T], rows: Iterable[dict[str, Any]]) -> list[T]:
    return [cls(**row) for row in rows]


class Substrate:
    """Read-only typed view over persisted substrate records.

    Evaluation sidecars are intentionally not loaded or exposed here.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise SubstrateNotFoundError(f"manifest.json is missing under {root}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = BuildManifest.model_validate(json.load(handle))

        self.documents = _construct(
            Document, read_rows(root / "nodes" / "documents.parquet")
        )
        self.chunks = _construct(Chunk, read_rows(root / "nodes" / "chunks.parquet"))
        self.sentences = _construct(
            Sentence, read_rows(root / "nodes" / "sentences.parquet")
        )
        self.entities = _construct(
            Entity, read_rows(root / "nodes" / "entities.parquet")
        )
        self.mentions = _construct(
            Mention, read_rows(root / "relations" / "mentions.parquet")
        )
        self.entity_aliases = _construct(
            EntityAlias, read_rows(root / "relations" / "entity_aliases.parquet")
        )
        self.document_scopes = _construct(
            DocumentScope,
            read_rows(root / "relations" / "document_scopes.parquet"),
        )

        self.document_by_id = {record.doc_id: record for record in self.documents}
        self.chunk_by_id = {record.chunk_id: record for record in self.chunks}
        self.sentence_by_id = {
            record.sentence_id: record for record in self.sentences
        }
        self.entity_by_id = {record.entity_id: record for record in self.entities}

        self.sentences_by_chunk: dict[str, list[Sentence]] = {}
        for sentence in self.sentences:
            self.sentences_by_chunk.setdefault(sentence.chunk_id, []).append(sentence)
        for values in self.sentences_by_chunk.values():
            values.sort(key=lambda item: (item.sentence_pos, item.sentence_id))

        self.doc_ids_by_scope: dict[str, set[str]] = {}
        self.scope_ids_by_doc: dict[str, set[str]] = {}
        for relation in self.document_scopes:
            self.doc_ids_by_scope.setdefault(relation.scope_id, set()).add(
                relation.doc_id
            )
            self.scope_ids_by_doc.setdefault(relation.doc_id, set()).add(
                relation.scope_id
            )

        chunk_ids_by_doc: dict[str, set[str]] = {}
        for chunk in self.chunks:
            chunk_ids_by_doc.setdefault(chunk.doc_id, set()).add(chunk.chunk_id)
        sentence_ids_by_chunk: dict[str, set[str]] = {}
        for sentence in self.sentences:
            sentence_ids_by_chunk.setdefault(sentence.chunk_id, set()).add(
                sentence.sentence_id
            )
        self.sentence_ids_by_scope: dict[str, set[str]] = {}
        self.chunk_ids_by_scope: dict[str, set[str]] = {}
        for scope_id, doc_ids in self.doc_ids_by_scope.items():
            chunks: set[str] = set()
            for doc_id in doc_ids:
                chunks.update(chunk_ids_by_doc.get(doc_id, set()))
            sentences: set[str] = set()
            for chunk_id in chunks:
                sentences.update(sentence_ids_by_chunk.get(chunk_id, set()))
            self.chunk_ids_by_scope[scope_id] = chunks
            self.sentence_ids_by_scope[scope_id] = sentences

        self.scope_ids_by_sentence: dict[str, set[str]] = {}
        for scope_id, sentence_ids in self.sentence_ids_by_scope.items():
            for sentence_id in sentence_ids:
                self.scope_ids_by_sentence.setdefault(sentence_id, set()).add(scope_id)
        self.entity_ids_by_scope: dict[str, set[str]] = {
            scope_id: set() for scope_id in self.doc_ids_by_scope
        }
        for mention in self.mentions:
            for scope_id in self.scope_ids_by_sentence.get(
                mention.sentence_id, set()
            ):
                self.entity_ids_by_scope[scope_id].add(mention.entity_id)

    @classmethod
    def open(cls, path: str | Path) -> "Substrate":
        return cls(Path(path))

    def require_scope(self, scope_id: str) -> None:
        if scope_id not in self.doc_ids_by_scope:
            raise ScopeNotFoundError(f"Unknown retrieval scope: {scope_id}")

    def document_for_chunk(self, chunk_id: str) -> Document:
        chunk = self.chunk_by_id.get(chunk_id)
        if chunk is None:
            raise NodeNotFoundError(f"Unknown Chunk ID: {chunk_id}")
        return self.document_by_id[chunk.doc_id]

    def chunk_for_sentence(self, sentence_id: str) -> Chunk:
        sentence = self.sentence_by_id.get(sentence_id)
        if sentence is None:
            raise NodeNotFoundError(f"Unknown Sentence ID: {sentence_id}")
        return self.chunk_by_id[sentence.chunk_id]

    def read_chunk(self, chunk_id: str) -> ChunkRead:
        chunk = self.chunk_by_id.get(chunk_id)
        if chunk is None:
            raise NodeNotFoundError(f"Unknown Chunk ID: {chunk_id}")
        document = self.document_by_id[chunk.doc_id]
        sentences = self.sentences_by_chunk.get(chunk_id, [])
        return ChunkRead(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            title=document.title,
            chunk_pos=chunk.chunk_pos,
            text=chunk.text,
            sentences=[
                SentenceResult(
                    sentence_id=item.sentence_id,
                    text=item.text,
                    parent_chunk_id=chunk.chunk_id,
                    document_id=document.doc_id,
                    title=document.title,
                )
                for item in sentences
            ],
        )
