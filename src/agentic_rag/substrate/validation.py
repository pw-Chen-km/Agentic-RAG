"""Structural and retrieval-substrate validation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse

from agentic_rag.substrate.storage import EvaluationSidecars, Substrate
from agentic_rag.substrate.text import normalize_document_text


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

    provenance_count = 0
    gold_support_count = 0
    benchmark_question_count = 0
    if substrate.manifest.source_format == "hotpotqa_global_provenance":
        if substrate.manifest.schema_version != "2.1":
            errors.append(
                "hotpotqa_global_provenance requires manifest schema version 2.1"
            )
        if substrate.manifest.index_versions.get("source_sentence_provenance") != "1":
            errors.append(
                "Source-sentence provenance version is missing from the manifest"
            )

        evaluation = EvaluationSidecars.open(substrate.root)
        provenance = evaluation.source_sentence_provenance
        gold_support = evaluation.gold_support
        questions = evaluation.benchmark_questions
        provenance_count = len(provenance)
        gold_support_count = len(gold_support)
        benchmark_question_count = len(questions)

        evaluation_counts = {
            "source_sentence_provenance": provenance_count,
            "gold_support": gold_support_count,
            "benchmark_questions": benchmark_question_count,
        }
        for label, count in evaluation_counts.items():
            if manifest_counts.get(label) != count:
                errors.append(
                    f"Manifest count for {label} is {manifest_counts.get(label)}, "
                    f"actual is {count}"
                )

        question_ids = [item.question_id for item in questions]
        check_unique(question_ids, "BenchmarkQuestion")
        question_by_id = {item.question_id: item for item in questions}
        scope_ids = set(substrate.doc_ids_by_scope)
        if len(scope_ids) != 1:
            errors.append(
                "HotpotQA global provenance must contain exactly one retrieval scope"
            )
        for question in questions:
            if question.scope_id not in scope_ids:
                errors.append(
                    f"Question {question.question_id} references unknown scope "
                    f"{question.scope_id}"
                )

        provenance_sentence_ids = [
            item.sentence_id for item in provenance if item.sentence_id is not None
        ]
        check_unique(provenance_sentence_ids, "SourceSentenceProvenance Sentence")
        provenance_keys = [
            (item.original_title, item.original_sentence_id) for item in provenance
        ]
        if len(provenance_keys) != len(set(provenance_keys)):
            errors.append(
                "SourceSentenceProvenance original title/sentence IDs are not unique"
            )
        provenance_by_sentence = {
            item.sentence_id: item
            for item in provenance
            if item.sentence_id is not None
        }
        for item in provenance:
            document = substrate.document_by_id.get(item.doc_id)
            if document is None:
                errors.append(
                    f"Source provenance references missing Document {item.doc_id}"
                )
                continue
            if document.title != item.original_title:
                errors.append(
                    f"Source provenance title for {item.doc_id} does not match "
                    "the runtime Document title"
                )
            if item.scope_id not in substrate.scope_ids_by_doc.get(item.doc_id, set()):
                errors.append(
                    f"Source provenance for {item.doc_id} references unknown scope "
                    f"{item.scope_id}"
                )
            if item.sentence_id is None:
                if normalize_document_text(item.original_sentence_text):
                    errors.append(
                        f"Non-blank source sentence {item.original_title!r}"
                        f"[{item.original_sentence_id}] has no runtime Sentence ID"
                    )
                continue
            sentence = substrate.sentence_by_id.get(item.sentence_id)
            if sentence is None:
                errors.append(
                    f"Source provenance references missing Sentence {item.sentence_id}"
                )
                continue
            chunk = substrate.chunk_by_id[sentence.chunk_id]
            if chunk.doc_id != item.doc_id:
                errors.append(
                    f"Source Sentence {item.sentence_id} belongs to Document "
                    f"{chunk.doc_id}, not {item.doc_id}"
                )
            if sentence.sentence_pos != item.original_sentence_id:
                errors.append(
                    f"Source Sentence {item.sentence_id} has position "
                    f"{sentence.sentence_pos}, expected {item.original_sentence_id}"
                )
            if normalize_document_text(item.original_sentence_text) != sentence.text:
                errors.append(
                    f"Source Sentence {item.sentence_id} text does not match exact "
                    "source provenance after normalization"
                )

        if set(provenance_sentence_ids) != sentence_ids:
            missing = sorted(sentence_ids - set(provenance_sentence_ids))
            extra = sorted(set(provenance_sentence_ids) - sentence_ids)
            errors.append(
                "Runtime/provenance Sentence IDs disagree: "
                f"runtime-only={missing[:5]}, provenance-only={extra[:5]}"
            )

        fact_keys = [(item.question_id, item.fact_id) for item in gold_support]
        if len(fact_keys) != len(set(fact_keys)):
            errors.append("Gold support question/fact keys are not unique")
        facts_by_question: dict[str, list[str]] = defaultdict(list)
        for item in gold_support:
            if item.question_id is None or item.fact_id is None:
                errors.append(
                    "HotpotQA global-provenance gold support is missing "
                    "question_id or fact_id"
                )
                continue
            question = question_by_id.get(item.question_id)
            if question is None:
                errors.append(
                    f"Gold support references unknown question {item.question_id}"
                )
            elif question.scope_id != item.scope_id:
                errors.append(
                    f"Gold support {item.question_id}/{item.fact_id} scope does "
                    "not match its question"
                )
            facts_by_question[item.question_id].append(item.fact_id)
            if item.sentence_id is None:
                errors.append(
                    f"Gold support {item.question_id}/{item.fact_id} has no "
                    "runtime Sentence ID"
                )
                continue
            source = provenance_by_sentence.get(item.sentence_id)
            if source is None:
                errors.append(
                    f"Gold support {item.question_id}/{item.fact_id} references "
                    f"Sentence {item.sentence_id} without source provenance"
                )
                continue
            if (
                source.doc_id != item.doc_id
                or source.original_title != item.title
                or source.original_sentence_id != item.source_sentence_pos
                or source.original_sentence_text != item.source_sentence_text
            ):
                errors.append(
                    f"Gold support {item.question_id}/{item.fact_id} does not "
                    "exactly match its source provenance"
                )

        for question_id in question_ids:
            fact_ids = sorted(facts_by_question.get(question_id, []))
            expected_fact_ids = [f"sf:{index:04d}" for index in range(len(fact_ids))]
            if fact_ids != expected_fact_ids:
                errors.append(
                    f"Question {question_id} has non-contiguous gold fact IDs: "
                    f"{fact_ids}"
                )

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
            "source_sentence_provenance": provenance_count,
            "gold_support": gold_support_count,
            "benchmark_questions": benchmark_question_count,
        },
    )
