"""Structural and retrieval-substrate validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse

from agentic_rag.storage import Substrate


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: dict[str, int | bool] = Field(default_factory=dict)


def validate_substrate(path: str | Path) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    substrate = Substrate.open(path)

    def check_unique(values: list[str], label: str) -> None:
        if len(values) != len(set(values)):
            errors.append(f"{label} IDs are not unique")

    check_unique([item.doc_id for item in substrate.documents], "Document")
    check_unique([item.chunk_id for item in substrate.chunks], "Chunk")
    check_unique([item.sentence_id for item in substrate.sentences], "Sentence")
    check_unique([item.entity_id for item in substrate.entities], "Entity")
    check_unique([item.mention_id for item in substrate.mentions], "Mention")
    check_unique([item.alias_id for item in substrate.entity_aliases], "EntityAlias")

    chunk_ids = set(substrate.chunk_by_id)
    doc_ids = set(substrate.document_by_id)
    sentence_ids = set(substrate.sentence_by_id)
    entity_ids = set(substrate.entity_by_id)
    for chunk in substrate.chunks:
        if chunk.doc_id not in doc_ids:
            errors.append(f"Chunk {chunk.chunk_id} has missing Document {chunk.doc_id}")
    for sentence in substrate.sentences:
        if sentence.chunk_id not in chunk_ids:
            errors.append(
                f"Sentence {sentence.sentence_id} has missing Chunk {sentence.chunk_id}"
            )
    for mention in substrate.mentions:
        sentence = substrate.sentence_by_id.get(mention.sentence_id)
        if sentence is None:
            errors.append(
                f"Mention {mention.mention_id} has missing Sentence {mention.sentence_id}"
            )
            continue
        if mention.entity_id not in entity_ids:
            errors.append(
                f"Mention {mention.mention_id} has missing Entity {mention.entity_id}"
            )
        if (
            mention.mention_start < 0
            or mention.mention_end > len(sentence.text)
            or sentence.text[mention.mention_start : mention.mention_end]
            != mention.surface_form
        ):
            errors.append(f"Mention {mention.mention_id} has an invalid text span")
    for alias in substrate.entity_aliases:
        if alias.entity_id not in entity_ids:
            errors.append(
                f"Alias {alias.alias_id} has missing Entity {alias.entity_id}"
            )
        if (
            alias.source_sentence_id is not None
            and alias.source_sentence_id not in sentence_ids
        ):
            errors.append(
                f"Alias {alias.alias_id} has missing source Sentence "
                f"{alias.source_sentence_id}"
            )

    chunk_positions: dict[str, list[int]] = {}
    for chunk in substrate.chunks:
        chunk_positions.setdefault(chunk.doc_id, []).append(chunk.chunk_pos)
    for doc_id, positions in chunk_positions.items():
        if sorted(positions) != list(range(len(positions))):
            errors.append(f"Document {doc_id} has non-contiguous Chunk positions")

    for chunk_id, contained in substrate.sentences_by_chunk.items():
        positions = [item.sentence_pos for item in contained]
        if positions != sorted(positions):
            errors.append(f"Chunk {chunk_id} has out-of-order Sentences")

    entity_to_sentence = sparse.load_npz(
        substrate.root / "relations" / "entity_to_sentence.npz"
    ).tocsr()
    sentence_to_entity = sparse.load_npz(
        substrate.root / "relations" / "sentence_to_entity.npz"
    ).tocsr()
    expected_shape = (len(substrate.entities), len(substrate.sentences))
    if entity_to_sentence.shape != expected_shape:
        errors.append(
            f"Entity→Sentence shape {entity_to_sentence.shape} != {expected_shape}"
        )
    if sentence_to_entity.shape != expected_shape[::-1]:
        errors.append(
            f"Sentence→Entity shape {sentence_to_entity.shape} != "
            f"{expected_shape[::-1]}"
        )
    transpose_difference = entity_to_sentence.transpose().tocsr() - sentence_to_entity
    if transpose_difference.nnz != 0:
        errors.append("Entity→Sentence and Sentence→Entity matrices are not transposes")

    entity_row = {
        item.entity_id: index for index, item in enumerate(substrate.entities)
    }
    sentence_column = {
        item.sentence_id: index for index, item in enumerate(substrate.sentences)
    }
    expected_pairs = {
        (entity_row[item.entity_id], sentence_column[item.sentence_id])
        for item in substrate.mentions
    }
    actual_rows, actual_columns = entity_to_sentence.nonzero()
    actual_pairs = set(zip(actual_rows.tolist(), actual_columns.tolist(), strict=True))
    if expected_pairs != actual_pairs:
        errors.append("Sparse incidence entries do not match Mention records")

    manifest_counts = substrate.manifest.record_counts
    actual_counts = {
        "documents": len(substrate.documents),
        "chunks": len(substrate.chunks),
        "sentences": len(substrate.sentences),
        "entities": len(substrate.entities),
        "mentions": len(substrate.mentions),
        "entity_aliases": len(substrate.entity_aliases),
        "document_scopes": len(substrate.document_scopes),
    }
    for label, count in actual_counts.items():
        if manifest_counts.get(label) != count:
            errors.append(
                f"Manifest count for {label} is {manifest_counts.get(label)}, "
                f"actual is {count}"
            )

    for scope_id, doc_scope_ids in substrate.doc_ids_by_scope.items():
        unknown_docs = doc_scope_ids - doc_ids
        if unknown_docs:
            errors.append(
                f"Scope {scope_id} references unknown Documents: {sorted(unknown_docs)}"
            )
    if not substrate.doc_ids_by_scope:
        warnings.append("Substrate contains no retrieval scopes")

    return ValidationReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        checks={
            "documents": len(substrate.documents),
            "chunks": len(substrate.chunks),
            "sentences": len(substrate.sentences),
            "entities": len(substrate.entities),
            "mentions": len(substrate.mentions),
            "incidence_nonzero": int(entity_to_sentence.nnz),
            "matrix_transpose": transpose_difference.nnz == 0,
        },
    )
