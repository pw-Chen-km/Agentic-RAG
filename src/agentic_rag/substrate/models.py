"""Public immutable records used by the substrate and retrieval APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class RawDocument:
    doc_id: str
    title: str | None
    text: str


@dataclass(frozen=True, slots=True)
class RawChunk:
    """A source-provided Chunk whose boundary must be preserved."""

    chunk_id: str
    doc_id: str
    chunk_pos: int
    text: str


@dataclass(frozen=True, slots=True)
class BenchmarkQuestion:
    """Evaluation-only question metadata; never loaded by ``Substrate``."""

    question_id: str
    scope_id: str
    source: str
    question: str
    answer: str
    question_type: str | None
    source_question_id: str | None = None
    source_row_index: int | None = None


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    """Pinned source-file identity recorded in the build manifest."""

    role: str
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SourceSentenceProvenance:
    """Evaluation-only mapping from a raw source sentence to the substrate.

    ``original_*`` fields are preserved exactly as supplied by HotpotQA.  A
    blank source sentence has no runtime node and therefore uses a null
    ``sentence_id``.
    """

    scope_id: str
    doc_id: str
    sentence_id: str | None
    original_title: str
    original_sentence_id: int
    original_sentence_text: str


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str | None


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_pos: int
    text: str


@dataclass(frozen=True, slots=True)
class Sentence:
    sentence_id: str
    chunk_id: str
    sentence_pos: int
    text: str


@dataclass(frozen=True, slots=True)
class Entity:
    entity_id: str
    canonical_name: str
    normalized_name: str
    entity_type: str | None


@dataclass(frozen=True, slots=True)
class Mention:
    mention_id: str
    entity_id: str
    sentence_id: str
    surface_form: str
    mention_start: int
    mention_end: int


@dataclass(frozen=True, slots=True)
class EntityAlias:
    alias_id: str
    entity_id: str
    alias: str
    normalized_alias: str
    alias_type: Literal["SURFACE", "ABBREVIATION", "KB"]
    source_sentence_id: str | None


@dataclass(frozen=True, slots=True)
class DocumentScope:
    scope_id: str
    doc_id: str


@dataclass(frozen=True, slots=True)
class GoldSupport:
    scope_id: str
    doc_id: str
    title: str
    source_sentence_pos: int
    source_sentence_text: str
    sentence_id: str | None = None
    question_id: str | None = None
    fact_id: str | None = None


@dataclass(frozen=True, slots=True)
class EntityMentionDraft:
    sentence_index: int
    surface_form: str
    start: int
    end: int
    entity_type: str | None


@dataclass(frozen=True, slots=True)
class ProcessedSentence:
    text: str
    token_count: int
    mentions: tuple[EntityMentionDraft, ...]


@dataclass(frozen=True, slots=True)
class AbbreviationLink:
    short_form: str
    long_form: str
    source_sentence_index: int


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    sentences: tuple[ProcessedSentence, ...]
    abbreviations: tuple[AbbreviationLink, ...] = ()


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BuildWarning(ManifestModel):
    code: str
    message: str
    doc_id: str | None = None


class ModelVersion(ManifestModel):
    name: str
    version: str | None = None


class MatrixShape(ManifestModel):
    rows: int
    columns: int
    nonzero: int


class BuildManifest(ManifestModel):
    schema_version: str = "2.0"
    constructor_version: str
    corpus_id: str
    dataset: str
    split: str
    source_path: str
    source_format: Literal[
        "hotpotqa_scoped",
        "hotpotqa_benchmark_exact",
        "hotpotqa_global_provenance",
        "benchmark_exact",
        "arag_benchmark_exact",
    ] = "hotpotqa_scoped"
    source_artifacts: list[SourceArtifact] = Field(default_factory=list)
    scope_mode: Literal["question", "global"] = "question"
    preserved_source_chunks: bool = False
    created_at_utc: str
    sentence_segmenter: ModelVersion
    ner_model: ModelVersion
    abbreviation_detector: ModelVersion | None
    embedding_model: ModelVersion
    embedding_dimension: int
    # Newer prebuilt substrates may record a remote embedding service.  Keep
    # these optional so older locally-built manifests remain compatible.
    embedding_backend: str | None = None
    embedding_host: str | None = None
    bm25_backend: ModelVersion
    bm25_tokenizer: str
    chunk_tokenizer: str
    max_chunk_tokens: int
    overlapping_chunks: bool = False
    record_counts: dict[str, int]
    matrix_shapes: dict[str, MatrixShape]
    index_versions: dict[str, str]
    warnings: list[BuildWarning] = Field(default_factory=list)


class SentencePreview(ManifestModel):
    sentence_id: str
    text: str


class SentenceResult(ManifestModel):
    """A complete sentence with enough provenance to navigate to its context."""

    sentence_id: str
    text: str
    parent_chunk_id: str
    document_id: str
    title: str | None


class SentenceHit(SentenceResult):
    target: Literal["SENTENCE"] = "SENTENCE"
    score: float


class ChunkHit(ManifestModel):
    target: Literal["CHUNK"] = "CHUNK"
    chunk_id: str
    score: float
    doc_id: str
    title: str | None
    previews: list[SentencePreview]


class EntityHit(ManifestModel):
    target: Literal["ENTITY"] = "ENTITY"
    entity_id: str
    score: float
    canonical_name: str
    entity_type: str | None
    mention_count: int


SearchHit = SentenceHit | ChunkHit | EntityHit


class ChunkRead(ManifestModel):
    chunk_id: str
    doc_id: str
    title: str | None
    chunk_pos: int
    text: str
    sentences: list[SentenceResult]


class BridgeHit(ManifestModel):
    source_entity_id: str
    bridge_sentence_id: str
    target_entity_id: str
    bridge_text: str
    target_canonical_name: str
    chunk_id: str
    doc_id: str
    title: str | None
