"""Deterministic Phase 1–2 substrate construction."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse

from agentic_rag import __version__
from agentic_rag.substrate.adapters import (
    AdapterOutput,
    BenchmarkExactAdapter,
    HotpotQAAdapter,
    HotpotQABenchmarkExactAdapter,
    SourceAdapter,
)
from agentic_rag.config import BuildConfig
from agentic_rag.substrate.embedding import (
    EmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
    save_dense_index,
)
from agentic_rag.errors import BuildError
from agentic_rag.substrate.models import (
    BuildManifest,
    BuildWarning,
    Chunk,
    Document,
    Entity,
    EntityAlias,
    GoldSupport,
    MatrixShape,
    Mention,
    ModelVersion,
    Sentence,
)
from agentic_rag.substrate.storage import write_records
from agentic_rag.substrate.text import (
    DocumentProcessor,
    SpacyDocumentProcessor,
    normalize_document_text,
    normalize_entity_name,
)


@dataclass(frozen=True, slots=True)
class _RawMention:
    doc_id: str
    sentence_id: str
    surface_form: str
    start: int
    end: int
    entity_type: str | None
    key: tuple[str, str | None]


@dataclass(frozen=True, slots=True)
class _AbbreviationDraft:
    doc_id: str
    short_form: str
    long_form: str
    source_sentence_id: str


class _UnionFind:
    def __init__(self, values: Iterable[tuple[str, str | None]]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: tuple[str, str | None]) -> tuple[str, str | None]:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(
        self, left: tuple[str, str | None], right: tuple[str, str | None]
    ) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if _entity_key_sort(left_root) <= _entity_key_sort(right_root):
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _entity_key_sort(key: tuple[str, str | None]) -> tuple[str, str]:
    return (key[0], key[1] or "")


def _compatible_types(
    left: tuple[str, str | None], right: tuple[str, str | None]
) -> bool:
    return left[1] is None or right[1] is None or left[1] == right[1]


class SubstrateBuilder:
    def __init__(
        self,
        config: BuildConfig,
        *,
        adapter: SourceAdapter | None = None,
        processor: DocumentProcessor | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self.config = config
        if adapter is not None:
            self.adapter = adapter
        elif config.source_format == "hotpotqa_benchmark_exact":
            self.adapter = HotpotQABenchmarkExactAdapter(
                scope_id=config.benchmark_scope_id,
                validate_reference_counts=config.validate_benchmark_profile,
            )
        elif config.source_format == "benchmark_exact":
            self.adapter = BenchmarkExactAdapter(
                config.dataset,
                scope_id=config.benchmark_scope_id,
                validate_reference_counts=config.validate_benchmark_profile,
            )
        else:
            self.adapter = HotpotQAAdapter()
        self.processor = processor
        self.embedding_backend = embedding_backend

    def build(self, source: str | Path, output: str | Path) -> BuildManifest:
        source_path = Path(source).resolve()
        output_path = Path(output).resolve()
        if output_path.exists():
            raise BuildError(
                f"Output already exists: {output_path}. Choose a new directory."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        staging = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
        staging.mkdir(parents=True)
        try:
            manifest = self._build_into(source_path, staging)
            staging.rename(output_path)
            return manifest
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _build_into(self, source_path: Path, root: Path) -> BuildManifest:
        adapter_output = self.adapter.load(source_path, self.config.split)
        processor = self.processor or SpacyDocumentProcessor(
            self.config.spacy_model,
            enable_abbreviations=self.config.enable_abbreviations,
        )
        backend = self.embedding_backend or SentenceTransformerEmbeddingBackend(
            self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
            device=self.config.embedding_device,
        )
        warnings: list[BuildWarning] = []

        if adapter_output.source_chunks:
            (
                documents,
                chunks,
                sentences,
                raw_mentions,
                abbreviations,
            ) = self._construct_preserved_chunk_records(
                adapter_output, processor, warnings
            )
        else:
            (
                documents,
                chunks,
                sentences,
                raw_mentions,
                abbreviations,
            ) = self._construct_text_records(adapter_output, processor, warnings)
        entities, mentions, aliases = self._construct_entities(
            raw_mentions, abbreviations, warnings
        )
        gold_support = self._resolve_gold_support(
            adapter_output.gold_support, sentences, chunks, warnings
        )

        self._write_records(
            root,
            documents,
            chunks,
            sentences,
            entities,
            mentions,
            aliases,
            adapter_output,
            gold_support,
        )
        matrix_shapes = self._write_sparse_mappings(
            root, entities, sentences, mentions
        )
        self._write_alias_index(root, aliases)

        from agentic_rag.substrate.bm25_index import build_bm25_index

        document_by_id = {item.doc_id: item for item in documents}
        chunk_by_id = {item.chunk_id: item for item in chunks}
        aliases_by_entity: dict[str, list[str]] = defaultdict(list)
        for alias in aliases:
            aliases_by_entity[alias.entity_id].append(alias.alias)
        sentence_texts = [
            self._retrieval_text_for_sentence(
                item, chunk_by_id, document_by_id
            )
            for item in sentences
        ]
        chunk_texts = [
            self._retrieval_text_for_chunk(item, document_by_id) for item in chunks
        ]
        build_bm25_index(
            root / "indexes" / "bm25_sentence",
            [item.sentence_id for item in sentences],
            sentence_texts,
            stopwords=self.config.bm25_stopwords,
        )
        build_bm25_index(
            root / "indexes" / "bm25_chunk",
            [item.chunk_id for item in chunks],
            chunk_texts,
            stopwords=self.config.bm25_stopwords,
        )

        dense_dimension = save_dense_index(
            root / "indexes" / "dense_sentence",
            [item.sentence_id for item in sentences],
            sentence_texts,
            backend,
        )
        chunk_dimension = save_dense_index(
            root / "indexes" / "dense_chunk",
            [item.chunk_id for item in chunks],
            chunk_texts,
            backend,
        )
        entity_texts = [
            self._retrieval_text_for_entity(
                item, aliases_by_entity.get(item.entity_id, [])
            )
            for item in entities
        ]
        entity_dimension = save_dense_index(
            root / "indexes" / "dense_entity",
            [item.entity_id for item in entities],
            entity_texts,
            backend,
        )
        dimensions = {dense_dimension, chunk_dimension, entity_dimension}
        if len(dimensions) != 1:
            raise BuildError(f"Dense index dimensions disagree: {sorted(dimensions)}")

        manifest = self._make_manifest(
            source_path=source_path,
            adapter_output=adapter_output,
            processor=processor,
            backend=backend,
            dense_dimension=dense_dimension,
            records={
                "documents": len(documents),
                "chunks": len(chunks),
                "sentences": len(sentences),
                "entities": len(entities),
                "mentions": len(mentions),
                "entity_aliases": len(aliases),
                "document_scopes": len(adapter_output.scopes),
                "gold_support": len(gold_support),
                "benchmark_questions": len(adapter_output.benchmark_questions),
            },
            matrix_shapes=matrix_shapes,
            warnings=warnings,
        )
        with (root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(
                manifest.model_dump(mode="json"),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        return manifest

    def _construct_preserved_chunk_records(
        self,
        adapter_output: AdapterOutput,
        processor: DocumentProcessor,
        warnings: list[BuildWarning],
    ) -> tuple[
        list[Document],
        list[Chunk],
        list[Sentence],
        list[_RawMention],
        list[_AbbreviationDraft],
    ]:
        """Process source Chunks without changing their text or boundaries."""

        documents = sorted(
            [Document(doc_id=item.doc_id, title=item.title) for item in adapter_output.documents],
            key=lambda item: item.doc_id,
        )
        document_ids = {item.doc_id for item in documents}
        chunks: list[Chunk] = []
        sentences: list[Sentence] = []
        raw_mentions: list[_RawMention] = []
        abbreviations: list[_AbbreviationDraft] = []

        for raw_chunk in sorted(
            adapter_output.source_chunks,
            key=lambda item: (item.doc_id, item.chunk_pos, item.chunk_id),
        ):
            if raw_chunk.doc_id not in document_ids:
                raise BuildError(
                    f"Preserved Chunk {raw_chunk.chunk_id} references unknown "
                    f"Document {raw_chunk.doc_id}"
                )
            chunks.append(
                Chunk(
                    chunk_id=raw_chunk.chunk_id,
                    doc_id=raw_chunk.doc_id,
                    chunk_pos=raw_chunk.chunk_pos,
                    text=raw_chunk.text,
                )
            )
            processor_text = normalize_document_text(raw_chunk.text)
            if not processor_text:
                warnings.append(
                    BuildWarning(
                        code="empty_source_chunk",
                        message="Preserved source Chunk is empty after normalization",
                        doc_id=raw_chunk.doc_id,
                    )
                )
                continue
            processed = processor.process(processor_text)
            if not processed.sentences:
                warnings.append(
                    BuildWarning(
                        code="source_chunk_without_sentences",
                        message=(
                            f"Preserved source Chunk {raw_chunk.chunk_id} produced "
                            "no Sentences"
                        ),
                        doc_id=raw_chunk.doc_id,
                    )
                )
                continue

            sentence_id_by_index: dict[int, str] = {}
            for sentence_index, processed_sentence in enumerate(processed.sentences):
                sentence_id = f"{raw_chunk.chunk_id}:s:{sentence_index:04d}"
                sentence_id_by_index[sentence_index] = sentence_id
                sentences.append(
                    Sentence(
                        sentence_id=sentence_id,
                        chunk_id=raw_chunk.chunk_id,
                        sentence_pos=sentence_index,
                        text=processed_sentence.text,
                    )
                )
                for mention in processed_sentence.mentions:
                    normalized_name = normalize_entity_name(mention.surface_form)
                    if not normalized_name:
                        continue
                    if (
                        mention.start < 0
                        or mention.end > len(processed_sentence.text)
                        or processed_sentence.text[mention.start : mention.end]
                        != mention.surface_form
                    ):
                        raise BuildError(
                            f"Invalid mention span in {sentence_id}: "
                            f"{mention.start}:{mention.end} {mention.surface_form!r}"
                        )
                    raw_mentions.append(
                        _RawMention(
                            doc_id=raw_chunk.doc_id,
                            sentence_id=sentence_id,
                            surface_form=mention.surface_form,
                            start=mention.start,
                            end=mention.end,
                            entity_type=mention.entity_type,
                            key=(normalized_name, mention.entity_type),
                        )
                    )
            for abbreviation in processed.abbreviations:
                source_sentence_id = sentence_id_by_index.get(
                    abbreviation.source_sentence_index
                )
                if source_sentence_id is None:
                    continue
                abbreviations.append(
                    _AbbreviationDraft(
                        doc_id=raw_chunk.doc_id,
                        short_form=abbreviation.short_form,
                        long_form=abbreviation.long_form,
                        source_sentence_id=source_sentence_id,
                    )
                )

        return (
            documents,
            sorted(chunks, key=lambda item: item.chunk_id),
            sorted(sentences, key=lambda item: item.sentence_id),
            sorted(
                raw_mentions,
                key=lambda item: (
                    item.sentence_id,
                    item.start,
                    item.end,
                    item.surface_form,
                ),
            ),
            sorted(
                abbreviations,
                key=lambda item: (
                    item.doc_id,
                    item.source_sentence_id,
                    item.short_form,
                    item.long_form,
                ),
            ),
        )

    def _construct_text_records(
        self,
        adapter_output: AdapterOutput,
        processor: DocumentProcessor,
        warnings: list[BuildWarning],
    ) -> tuple[
        list[Document],
        list[Chunk],
        list[Sentence],
        list[_RawMention],
        list[_AbbreviationDraft],
    ]:
        documents: list[Document] = []
        chunks: list[Chunk] = []
        sentences: list[Sentence] = []
        raw_mentions: list[_RawMention] = []
        abbreviations: list[_AbbreviationDraft] = []

        for raw_document in sorted(adapter_output.documents, key=lambda item: item.doc_id):
            normalized_text = normalize_document_text(raw_document.text)
            if not normalized_text:
                warnings.append(
                    BuildWarning(
                        code="empty_document",
                        message="Document is empty after normalization",
                        doc_id=raw_document.doc_id,
                    )
                )
                documents.append(
                    Document(doc_id=raw_document.doc_id, title=raw_document.title)
                )
                continue
            processed = processor.process(normalized_text)
            documents.append(
                Document(doc_id=raw_document.doc_id, title=raw_document.title)
            )

            groups: list[list[int]] = []
            current: list[int] = []
            current_tokens = 0
            for sentence_index, processed_sentence in enumerate(processed.sentences):
                token_count = max(1, processed_sentence.token_count)
                if token_count > self.config.max_chunk_tokens:
                    if current:
                        groups.append(current)
                        current = []
                        current_tokens = 0
                    groups.append([sentence_index])
                    warnings.append(
                        BuildWarning(
                            code="long_sentence_chunk",
                            message=(
                                f"Sentence {sentence_index} has {token_count} tokens "
                                f"and was placed in its own chunk"
                            ),
                            doc_id=raw_document.doc_id,
                        )
                    )
                    continue
                if current and current_tokens + token_count > self.config.max_chunk_tokens:
                    groups.append(current)
                    current = []
                    current_tokens = 0
                current.append(sentence_index)
                current_tokens += token_count
            if current:
                groups.append(current)

            sentence_id_by_index: dict[int, str] = {}
            for chunk_pos, sentence_indices in enumerate(groups):
                chunk_id = f"{raw_document.doc_id}:c:{chunk_pos}"
                chunk_text = " ".join(
                    processed.sentences[index].text for index in sentence_indices
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=raw_document.doc_id,
                        chunk_pos=chunk_pos,
                        text=chunk_text,
                    )
                )
                for sentence_index in sentence_indices:
                    processed_sentence = processed.sentences[sentence_index]
                    sentence_id = f"{raw_document.doc_id}:s:{sentence_index}"
                    sentence_id_by_index[sentence_index] = sentence_id
                    sentences.append(
                        Sentence(
                            sentence_id=sentence_id,
                            chunk_id=chunk_id,
                            sentence_pos=sentence_index,
                            text=processed_sentence.text,
                        )
                    )
                    for mention in processed_sentence.mentions:
                        normalized_name = normalize_entity_name(mention.surface_form)
                        if not normalized_name:
                            continue
                        if (
                            mention.start < 0
                            or mention.end > len(processed_sentence.text)
                            or processed_sentence.text[mention.start : mention.end]
                            != mention.surface_form
                        ):
                            raise BuildError(
                                f"Invalid mention span in {sentence_id}: "
                                f"{mention.start}:{mention.end} {mention.surface_form!r}"
                            )
                        key = (normalized_name, mention.entity_type)
                        raw_mentions.append(
                            _RawMention(
                                doc_id=raw_document.doc_id,
                                sentence_id=sentence_id,
                                surface_form=mention.surface_form,
                                start=mention.start,
                                end=mention.end,
                                entity_type=mention.entity_type,
                                key=key,
                            )
                        )
            for abbreviation in processed.abbreviations:
                sentence_id = sentence_id_by_index.get(
                    abbreviation.source_sentence_index
                )
                if sentence_id is None:
                    continue
                abbreviations.append(
                    _AbbreviationDraft(
                        doc_id=raw_document.doc_id,
                        short_form=abbreviation.short_form,
                        long_form=abbreviation.long_form,
                        source_sentence_id=sentence_id,
                    )
                )

        return (
            sorted(documents, key=lambda item: item.doc_id),
            sorted(chunks, key=lambda item: item.chunk_id),
            sorted(sentences, key=lambda item: item.sentence_id),
            sorted(
                raw_mentions,
                key=lambda item: (
                    item.sentence_id,
                    item.start,
                    item.end,
                    item.surface_form,
                ),
            ),
            sorted(
                abbreviations,
                key=lambda item: (
                    item.doc_id,
                    item.source_sentence_id,
                    item.short_form,
                    item.long_form,
                ),
            ),
        )

    def _construct_entities(
        self,
        raw_mentions: list[_RawMention],
        abbreviations: list[_AbbreviationDraft],
        warnings: list[BuildWarning],
    ) -> tuple[list[Entity], list[Mention], list[EntityAlias]]:
        keys = {item.key for item in raw_mentions}
        union_find = _UnionFind(keys)
        keys_by_doc_name: dict[str, dict[str, set[tuple[str, str | None]]]] = {}
        for mention in raw_mentions:
            keys_by_doc_name.setdefault(mention.doc_id, {}).setdefault(
                mention.key[0], set()
            ).add(mention.key)

        for abbreviation in abbreviations:
            doc_names = keys_by_doc_name.get(abbreviation.doc_id, {})
            short_keys = sorted(
                doc_names.get(normalize_entity_name(abbreviation.short_form), set()),
                key=_entity_key_sort,
            )
            long_keys = sorted(
                doc_names.get(normalize_entity_name(abbreviation.long_form), set()),
                key=_entity_key_sort,
            )
            compatible_pairs = [
                (short_key, long_key)
                for short_key in short_keys
                for long_key in long_keys
                if _compatible_types(short_key, long_key)
            ]
            if len(compatible_pairs) == 1:
                union_find.union(*compatible_pairs[0])
            elif len(compatible_pairs) > 1:
                warnings.append(
                    BuildWarning(
                        code="ambiguous_abbreviation",
                        message=(
                            f"Did not merge ambiguous abbreviation "
                            f"{abbreviation.short_form!r} → {abbreviation.long_form!r}"
                        ),
                        doc_id=abbreviation.doc_id,
                    )
                )

        mentions_by_root: dict[tuple[str, str | None], list[_RawMention]] = defaultdict(
            list
        )
        for mention in raw_mentions:
            mentions_by_root[union_find.find(mention.key)].append(mention)

        abbreviation_roots: dict[
            tuple[str, str | None], list[_AbbreviationDraft]
        ] = defaultdict(list)
        for abbreviation in abbreviations:
            doc_names = keys_by_doc_name.get(abbreviation.doc_id, {})
            candidates = (
                doc_names.get(normalize_entity_name(abbreviation.long_form), set())
                | doc_names.get(normalize_entity_name(abbreviation.short_form), set())
            )
            roots = {union_find.find(key) for key in candidates}
            if len(roots) == 1:
                abbreviation_roots[next(iter(roots))].append(abbreviation)

        roots = sorted(mentions_by_root, key=_entity_key_sort)
        entity_id_by_root: dict[tuple[str, str | None], str] = {}
        entities: list[Entity] = []
        for ordinal, root in enumerate(roots):
            entity_id = f"{self.config.corpus_id}:e:{ordinal:08d}"
            entity_id_by_root[root] = entity_id
            cluster_mentions = mentions_by_root[root]
            surface_counts = Counter(item.surface_form for item in cluster_mentions)
            long_forms = {
                abbreviation.long_form
                for abbreviation in abbreviation_roots.get(root, [])
            }
            candidates = set(surface_counts) | long_forms

            def canonical_sort(surface: str) -> tuple[int, int, int, int, str, str]:
                return (
                    -int(surface in long_forms),
                    -surface_counts[surface],
                    -len(surface.split()),
                    -len(surface),
                    surface.casefold(),
                    surface,
                )

            canonical_name = sorted(candidates, key=canonical_sort)[0]
            types = {item.entity_type for item in cluster_mentions if item.entity_type}
            entity_type = next(iter(types)) if len(types) == 1 else None
            entities.append(
                Entity(
                    entity_id=entity_id,
                    canonical_name=canonical_name,
                    normalized_name=normalize_entity_name(canonical_name),
                    entity_type=entity_type,
                )
            )

        mentions: list[Mention] = []
        seen_mention_ids: set[str] = set()
        for draft in raw_mentions:
            root = union_find.find(draft.key)
            mention_id = (
                f"{draft.sentence_id}:m:{draft.start}-{draft.end}"
            )
            if mention_id in seen_mention_ids:
                raise BuildError(f"Duplicate Mention ID: {mention_id}")
            seen_mention_ids.add(mention_id)
            mentions.append(
                Mention(
                    mention_id=mention_id,
                    entity_id=entity_id_by_root[root],
                    sentence_id=draft.sentence_id,
                    surface_form=draft.surface_form,
                    mention_start=draft.start,
                    mention_end=draft.end,
                )
            )

        aliases: list[EntityAlias] = []
        for root in roots:
            entity_id = entity_id_by_root[root]
            alias_drafts: set[tuple[str, str, str | None]] = {
                (mention.surface_form, "SURFACE", None)
                for mention in mentions_by_root[root]
            }
            for abbreviation in abbreviation_roots.get(root, []):
                alias_drafts.add(
                    (
                        abbreviation.short_form,
                        "ABBREVIATION",
                        abbreviation.source_sentence_id,
                    )
                )
                alias_drafts.add((abbreviation.long_form, "SURFACE", None))
            ordered = sorted(
                alias_drafts,
                key=lambda item: (
                    normalize_entity_name(item[0]),
                    item[1],
                    item[2] or "",
                    item[0],
                ),
            )
            for ordinal, (alias, alias_type, source_sentence_id) in enumerate(ordered):
                aliases.append(
                    EntityAlias(
                        alias_id=f"{entity_id}:a:{ordinal}",
                        entity_id=entity_id,
                        alias=alias,
                        normalized_alias=normalize_entity_name(alias),
                        alias_type=alias_type,  # type: ignore[arg-type]
                        source_sentence_id=source_sentence_id,
                    )
                )
        return (
            sorted(entities, key=lambda item: item.entity_id),
            sorted(mentions, key=lambda item: item.mention_id),
            sorted(aliases, key=lambda item: item.alias_id),
        )

    def _resolve_gold_support(
        self,
        support: tuple[GoldSupport, ...],
        sentences: list[Sentence],
        chunks: list[Chunk],
        warnings: list[BuildWarning],
    ) -> list[GoldSupport]:
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        by_doc: dict[str, list[Sentence]] = defaultdict(list)
        for sentence in sentences:
            by_doc[chunk_by_id[sentence.chunk_id].doc_id].append(sentence)
        for values in by_doc.values():
            values.sort(key=lambda item: item.sentence_pos)

        resolved: list[GoldSupport] = []
        for item in support:
            candidates = by_doc.get(item.doc_id, [])
            expected = normalize_document_text(item.source_sentence_text)
            sentence_id: str | None = None
            if item.source_sentence_pos < len(candidates):
                positional = candidates[item.source_sentence_pos]
                if normalize_document_text(positional.text) == expected:
                    sentence_id = positional.sentence_id
            if sentence_id is None:
                exact = [
                    sentence.sentence_id
                    for sentence in candidates
                    if normalize_document_text(sentence.text) == expected
                ]
                if len(exact) == 1:
                    sentence_id = exact[0]
            if sentence_id is None:
                warnings.append(
                    BuildWarning(
                        code="unresolved_gold_support",
                        message=(
                            f"Could not map source sentence {item.source_sentence_pos} "
                            f"to a substrate Sentence ID"
                        ),
                        doc_id=item.doc_id,
                    )
                )
            resolved.append(replace(item, sentence_id=sentence_id))
        return sorted(
            resolved,
            key=lambda item: (
                item.scope_id,
                item.doc_id,
                item.source_sentence_pos,
            ),
        )

    def _write_records(
        self,
        root: Path,
        documents: list[Document],
        chunks: list[Chunk],
        sentences: list[Sentence],
        entities: list[Entity],
        mentions: list[Mention],
        aliases: list[EntityAlias],
        adapter_output: AdapterOutput,
        gold_support: list[GoldSupport],
    ) -> None:
        write_records(root / "nodes" / "documents.parquet", documents, "documents")
        write_records(root / "nodes" / "chunks.parquet", chunks, "chunks")
        write_records(root / "nodes" / "sentences.parquet", sentences, "sentences")
        write_records(root / "nodes" / "entities.parquet", entities, "entities")
        write_records(
            root / "relations" / "mentions.parquet", mentions, "mentions"
        )
        write_records(
            root / "relations" / "entity_aliases.parquet",
            aliases,
            "entity_aliases",
        )
        write_records(
            root / "relations" / "document_scopes.parquet",
            sorted(
                adapter_output.scopes, key=lambda item: (item.scope_id, item.doc_id)
            ),
            "document_scopes",
        )
        write_records(
            root / "evaluation" / "gold_support.parquet",
            gold_support,
            "gold_support",
        )
        write_records(
            root / "evaluation" / "benchmark_questions.parquet",
            adapter_output.benchmark_questions,
            "benchmark_questions",
        )

    def _write_sparse_mappings(
        self,
        root: Path,
        entities: list[Entity],
        sentences: list[Sentence],
        mentions: list[Mention],
    ) -> dict[str, MatrixShape]:
        entity_row = {item.entity_id: index for index, item in enumerate(entities)}
        sentence_column = {
            item.sentence_id: index for index, item in enumerate(sentences)
        }
        pairs = sorted(
            {
                (entity_row[mention.entity_id], sentence_column[mention.sentence_id])
                for mention in mentions
            }
        )
        if pairs:
            rows, columns = zip(*pairs, strict=True)
            values = np.ones(len(pairs), dtype=np.uint8)
            entity_to_sentence = sparse.csr_matrix(
                (values, (rows, columns)),
                shape=(len(entities), len(sentences)),
                dtype=np.uint8,
            )
        else:
            entity_to_sentence = sparse.csr_matrix(
                (len(entities), len(sentences)), dtype=np.uint8
            )
        sentence_to_entity = entity_to_sentence.transpose().tocsr()
        relations = root / "relations"
        relations.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(
            relations / "entity_to_sentence.npz", entity_to_sentence, compressed=True
        )
        sparse.save_npz(
            relations / "sentence_to_entity.npz", sentence_to_entity, compressed=True
        )
        return {
            "entity_to_sentence": MatrixShape(
                rows=entity_to_sentence.shape[0],
                columns=entity_to_sentence.shape[1],
                nonzero=int(entity_to_sentence.nnz),
            ),
            "sentence_to_entity": MatrixShape(
                rows=sentence_to_entity.shape[0],
                columns=sentence_to_entity.shape[1],
                nonzero=int(sentence_to_entity.nnz),
            ),
        }

    def _write_alias_index(self, root: Path, aliases: list[EntityAlias]) -> None:
        lookup: dict[str, set[str]] = defaultdict(set)
        for alias in aliases:
            lookup[alias.normalized_alias].add(alias.entity_id)
        path = root / "indexes" / "entity_alias" / "lookup.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                {key: sorted(values) for key, values in sorted(lookup.items())},
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

    @staticmethod
    def _retrieval_text_for_sentence(
        sentence: Sentence,
        chunks: dict[str, Chunk],
        documents: dict[str, Document],
    ) -> str:
        chunk = chunks[sentence.chunk_id]
        title = documents[chunk.doc_id].title
        return f"{title} {sentence.text}".strip() if title else sentence.text

    @staticmethod
    def _retrieval_text_for_chunk(
        chunk: Chunk, documents: dict[str, Document]
    ) -> str:
        document = documents[chunk.doc_id]
        return f"{document.title} {chunk.text}".strip() if document.title else chunk.text

    @staticmethod
    def _retrieval_text_for_entity(
        entity: Entity, aliases: list[str]
    ) -> str:
        parts = [entity.canonical_name, *aliases]
        if entity.entity_type:
            parts.append(entity.entity_type)
        return " ".join(dict.fromkeys(parts))

    def _make_manifest(
        self,
        *,
        source_path: Path,
        adapter_output: AdapterOutput,
        processor: DocumentProcessor,
        backend: EmbeddingBackend,
        dense_dimension: int,
        records: dict[str, int],
        matrix_shapes: dict[str, MatrixShape],
        warnings: list[BuildWarning],
    ) -> BuildManifest:
        try:
            bm25_version = importlib.metadata.version("bm25s")
        except importlib.metadata.PackageNotFoundError:
            bm25_version = None
        return BuildManifest(
            constructor_version=__version__,
            corpus_id=self.config.corpus_id,
            dataset=self.adapter.dataset_name,
            split=self.config.split,
            source_path=source_path.as_posix(),
            source_format=self.config.source_format,
            source_artifacts=list(adapter_output.source_artifacts),
            scope_mode=adapter_output.scope_mode,
            preserved_source_chunks=bool(adapter_output.source_chunks),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            sentence_segmenter=ModelVersion(
                name=processor.segmenter_name, version=processor.segmenter_version
            ),
            ner_model=ModelVersion(
                name=processor.ner_name, version=processor.ner_version
            ),
            abbreviation_detector=(
                ModelVersion(
                    name=processor.abbreviation_name,
                    version=processor.abbreviation_version,
                )
                if processor.abbreviation_name
                else None
            ),
            embedding_model=ModelVersion(name=backend.name, version=backend.version),
            embedding_dimension=dense_dimension,
            bm25_backend=ModelVersion(name="bm25s", version=bm25_version),
            bm25_tokenizer="unicode-nfkc-casefold-regex-v1",
            chunk_tokenizer=processor.chunk_tokenizer_name,
            max_chunk_tokens=self.config.max_chunk_tokens,
            overlapping_chunks=adapter_output.overlapping_chunks,
            record_counts=records,
            matrix_shapes=matrix_shapes,
            index_versions={
                "entity_alias": "1",
                "bm25_sentence": "1",
                "bm25_chunk": "1",
                "dense_entity": "1",
                "dense_sentence": "1",
                "dense_chunk": "1",
                "entity_sentence_sparse": "1",
            },
            warnings=warnings,
        )
