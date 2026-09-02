"""Source adapters for the five benchmark datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Protocol

from agentic_rag.evaluation.profiles import (
    DatasetProfile,
    DuplicateQuestionIdPolicy,
    get_dataset_profile,
    normalize_task_type,
)
from agentic_rag.errors import InputFormatError
from agentic_rag.substrate.models import (
    BenchmarkQuestion,
    DocumentScope,
    GoldSupport,
    RawChunk,
    RawDocument,
    SourceArtifact,
    SourceSentenceProvenance,
)
from agentic_rag.substrate.text import normalize_document_text


@dataclass(frozen=True, slots=True)
class AdapterOutput:
    documents: tuple[RawDocument, ...]
    scopes: tuple[DocumentScope, ...]
    gold_support: tuple[GoldSupport, ...]
    source_sentence_provenance: tuple[SourceSentenceProvenance, ...] = ()
    source_chunks: tuple[RawChunk, ...] = ()
    benchmark_questions: tuple[BenchmarkQuestion, ...] = ()
    source_artifacts: tuple[SourceArtifact, ...] = ()
    scope_mode: Literal["question", "global"] = "question"
    overlapping_chunks: bool = False


class SourceAdapter(Protocol):
    dataset_name: str

    def load(self, source_path: Path, split: str) -> AdapterOutput: ...

    def iter_documents(
        self, source_path: Path, split: str
    ) -> Iterator[RawDocument]: ...


def _load_json_or_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise InputFormatError(f"Source file does not exist: {path}")
    try:
        if path.suffix.lower() == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise InputFormatError(
                            f"JSONL line {line_number} is not an object"
                        )
                    rows.append(value)
            return rows
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise InputFormatError(f"Invalid JSON in {path}: {exc}") from exc
    if isinstance(value, dict):
        for key in ("data", "examples", "records"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                value = candidate
                break
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise InputFormatError(
            "Source input must be a JSON array, JSONL objects, or an object "
            "containing a data/examples/records array"
        )
    return value


def _normalize_context(raw_context: object) -> list[tuple[str, list[str]]]:
    if isinstance(raw_context, dict):
        titles = raw_context.get("title") or raw_context.get("titles")
        sentences = raw_context.get("sentences") or raw_context.get("text")
        if isinstance(titles, list) and isinstance(sentences, list):
            return [
                (str(title), [str(sentence) for sentence in sentence_list])
                for title, sentence_list in zip(titles, sentences, strict=False)
            ]
    if isinstance(raw_context, list):
        contexts: list[tuple[str, list[str]]] = []
        for index, entry in enumerate(raw_context):
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or not isinstance(entry[1], list)
            ):
                raise InputFormatError(
                    f"Context entry {index} must be [title, [sentences...]]"
                )
            contexts.append((str(entry[0]), [str(sentence) for sentence in entry[1]]))
        return contexts
    raise InputFormatError("HotpotQA record is missing a supported context field")


def _normalize_supporting_facts(raw: object) -> list[tuple[str, int]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        titles = raw.get("title") or raw.get("titles")
        positions = raw.get("sent_id") or raw.get("sentence_id")
        if isinstance(titles, list) and isinstance(positions, list):
            return [
                (str(title), int(position))
                for title, position in zip(titles, positions, strict=False)
            ]
    if isinstance(raw, list):
        supports = []
        for entry in raw:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                supports.append((str(entry[0]), int(entry[1])))
        return supports
    raise InputFormatError("supporting_facts must be a list or columnar object")


class HotpotQAAdapter:
    """Parse HotpotQA-style query-scoped contexts."""

    dataset_name = "hotpotqa"

    def load(self, source_path: Path, split: str) -> AdapterOutput:
        rows = _load_json_or_jsonl(source_path)
        documents: list[RawDocument] = []
        scopes: list[DocumentScope] = []
        gold_support: list[GoldSupport] = []
        seen_scopes: set[str] = set()
        seen_docs: set[str] = set()

        for row_index, row in enumerate(rows):
            scope_id = str(row.get("_id") or row.get("id") or row.get("qid") or "")
            if not scope_id:
                raise InputFormatError(f"Record {row_index} has no _id/id/qid")
            if scope_id in seen_scopes:
                raise InputFormatError(f"Duplicate question scope ID: {scope_id}")
            seen_scopes.add(scope_id)
            contexts = _normalize_context(row.get("context"))
            title_to_context: dict[str, tuple[str, list[str]]] = {}

            for context_index, (title, source_sentences) in enumerate(contexts):
                doc_id = f"hotpotqa:{split}:{scope_id}:ctx:{context_index}"
                if doc_id in seen_docs:
                    raise InputFormatError(f"Duplicate document ID: {doc_id}")
                seen_docs.add(doc_id)
                text = " ".join(sentence.strip() for sentence in source_sentences)
                documents.append(RawDocument(doc_id=doc_id, title=title, text=text))
                scopes.append(DocumentScope(scope_id=scope_id, doc_id=doc_id))
                title_to_context.setdefault(title, (doc_id, source_sentences))

            supports = _normalize_supporting_facts(row.get("supporting_facts"))
            for title, source_pos in supports:
                context = title_to_context.get(title)
                if context is None:
                    raise InputFormatError(
                        f"Supporting-fact title {title!r} is absent from scope {scope_id}"
                    )
                doc_id, source_sentences = context
                if source_pos < 0 or source_pos >= len(source_sentences):
                    raise InputFormatError(
                        f"Supporting-fact sentence index {source_pos} is invalid "
                        f"for {title!r} in scope {scope_id}"
                    )
                gold_support.append(
                    GoldSupport(
                        scope_id=scope_id,
                        doc_id=doc_id,
                        title=title,
                        source_sentence_pos=source_pos,
                        source_sentence_text=source_sentences[source_pos],
                    )
                )

        return AdapterOutput(
            documents=tuple(documents),
            scopes=tuple(scopes),
            gold_support=tuple(gold_support),
        )

    def iter_documents(self, source_path: Path, split: str) -> Iterator[RawDocument]:
        yield from self.load(source_path, split).documents


class HotpotQAGlobalProvenanceAdapter:
    """Build a global, title-deduplicated HotpotQA corpus with exact lineage.

    Unlike :class:`HotpotQAAdapter`, this adapter does not duplicate the same
    Wikipedia document once per question.  Every unique title becomes one
    document in a single benchmark scope, while raw title and sentence-index
    provenance is emitted only through evaluation sidecars.
    """

    dataset_name = "hotpotqa"

    def __init__(
        self,
        scope_id: str | None = None,
        *,
        validate_reference_counts: bool = False,
    ) -> None:
        self.profile = get_dataset_profile("hotpotqa")
        self.scope_id = scope_id
        self.validate_reference_counts = validate_reference_counts

    def load(self, source_path: Path, split: str) -> AdapterOutput:
        rows = _load_json_or_jsonl(source_path)
        scope_id = self.scope_id or self.profile.scope_id(split)
        title_sentences: dict[str, tuple[str, ...]] = {}
        parsed_rows: list[
            tuple[
                int,
                str,
                dict[str, tuple[str, ...]],
                list[tuple[str, int]],
                dict,
            ]
        ] = []
        seen_question_ids: set[str] = set()

        for row_index, row in enumerate(rows):
            question_id = str(row.get("_id") or row.get("id") or row.get("qid") or "")
            if not question_id:
                raise InputFormatError(f"Record {row_index} has no _id/id/qid")
            if question_id in seen_question_ids:
                raise InputFormatError(f"Duplicate HotpotQA question ID: {question_id}")
            seen_question_ids.add(question_id)

            contexts = _normalize_context(row.get("context"))
            question_context: dict[str, tuple[str, ...]] = {}
            for title, source_sentences in contexts:
                if not title:
                    raise InputFormatError(
                        f"HotpotQA question {question_id} contains a blank title"
                    )
                sentence_tuple = tuple(source_sentences)
                local_existing = question_context.get(title)
                if local_existing is not None and local_existing != sentence_tuple:
                    raise InputFormatError(
                        f"HotpotQA question {question_id} repeats title {title!r} "
                        "with inconsistent text"
                    )
                question_context[title] = sentence_tuple
                global_existing = title_sentences.get(title)
                if global_existing is not None and global_existing != sentence_tuple:
                    raise InputFormatError(
                        f"HotpotQA title {title!r} has inconsistent sentence text "
                        "across questions"
                    )
                title_sentences[title] = sentence_tuple

            supports = _normalize_supporting_facts(row.get("supporting_facts"))
            for title, source_pos in supports:
                source_sentences = question_context.get(title)
                if source_sentences is None:
                    raise InputFormatError(
                        f"Supporting-fact title {title!r} is absent from question "
                        f"{question_id}"
                    )
                if source_pos < 0 or source_pos >= len(source_sentences):
                    raise InputFormatError(
                        f"Supporting-fact sentence index {source_pos} is invalid "
                        f"for {title!r} in question {question_id}"
                    )
                if not normalize_document_text(source_sentences[source_pos]):
                    raise InputFormatError(
                        f"Supporting fact {title!r}[{source_pos}] in question "
                        f"{question_id} is blank"
                    )
            parsed_rows.append(
                (row_index, question_id, question_context, supports, row)
            )

        ordered_titles = sorted(
            title_sentences, key=lambda value: (value.casefold(), value)
        )
        doc_id_by_title = {
            title: f"hotpotqa:{split}:global_provenance:d:{ordinal:06d}"
            for ordinal, title in enumerate(ordered_titles)
        }
        documents = tuple(
            RawDocument(
                doc_id=doc_id_by_title[title],
                title=title,
                text=" ".join(title_sentences[title]),
            )
            for title in ordered_titles
        )
        scopes = tuple(
            DocumentScope(scope_id=scope_id, doc_id=item.doc_id) for item in documents
        )

        provenance: list[SourceSentenceProvenance] = []
        sentence_id_by_source: dict[tuple[str, int], str | None] = {}
        for title in ordered_titles:
            doc_id = doc_id_by_title[title]
            for source_pos, source_text in enumerate(title_sentences[title]):
                sentence_id = (
                    f"{doc_id}:s:{source_pos:04d}"
                    if normalize_document_text(source_text)
                    else None
                )
                sentence_id_by_source[(title, source_pos)] = sentence_id
                provenance.append(
                    SourceSentenceProvenance(
                        scope_id=scope_id,
                        doc_id=doc_id,
                        sentence_id=sentence_id,
                        original_title=title,
                        original_sentence_id=source_pos,
                        original_sentence_text=source_text,
                    )
                )

        questions: list[BenchmarkQuestion] = []
        gold_support: list[GoldSupport] = []
        task_type_counts: Counter[str] = Counter()
        for row_index, question_id, question_context, supports, row in parsed_rows:
            question = row.get("question")
            answer = row.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise InputFormatError(
                    f"HotpotQA question {question_id} has no question text"
                )
            if not isinstance(answer, str) or not answer.strip():
                raise InputFormatError(
                    f"HotpotQA question {question_id} has no non-empty answer"
                )
            question_type = normalize_task_type(
                self.profile,
                row.get("type") or row.get("question_type"),
                question_id=question_id,
            )
            task_type_counts[question_type] += 1
            questions.append(
                BenchmarkQuestion(
                    question_id=question_id,
                    scope_id=scope_id,
                    source=str(row.get("source") or "hotpotqa"),
                    question=question,
                    answer=answer,
                    question_type=question_type,
                    source_question_id=question_id,
                    source_row_index=row_index,
                )
            )
            for fact_index, (title, source_pos) in enumerate(supports):
                source_text = question_context[title][source_pos]
                sentence_id = sentence_id_by_source[(title, source_pos)]
                if sentence_id is None:
                    raise InputFormatError(
                        f"Supporting fact {title!r}[{source_pos}] in question "
                        f"{question_id} has no substrate sentence"
                    )
                gold_support.append(
                    GoldSupport(
                        scope_id=scope_id,
                        doc_id=doc_id_by_title[title],
                        title=title,
                        source_sentence_pos=source_pos,
                        source_sentence_text=source_text,
                        sentence_id=sentence_id,
                        question_id=question_id,
                        fact_id=f"sf:{fact_index:04d}",
                    )
                )

        if self.validate_reference_counts:
            expected_task_types = Counter(dict(self.profile.reference_task_type_counts))
            errors: list[str] = []
            if len(questions) != self.profile.reference_question_count:
                errors.append(
                    f"questions expected {self.profile.reference_question_count}, "
                    f"found {len(questions)}"
                )
            if task_type_counts != expected_task_types:
                errors.append(
                    f"task type counts expected {dict(expected_task_types)}, found "
                    f"{dict(task_type_counts)}"
                )
            if errors:
                raise InputFormatError(
                    "HotpotQA global-provenance reference profile mismatch: "
                    + "; ".join(errors)
                )

        return AdapterOutput(
            documents=documents,
            scopes=scopes,
            gold_support=tuple(gold_support),
            source_sentence_provenance=tuple(provenance),
            benchmark_questions=tuple(
                sorted(questions, key=lambda item: item.question_id)
            ),
            source_artifacts=(_source_artifact(source_path, "hotpotqa_source"),),
            scope_mode="global",
        )

    def iter_documents(self, source_path: Path, split: str) -> Iterator[RawDocument]:
        yield from self.load(source_path, split).documents


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_artifact(path: Path, role: str) -> SourceArtifact:
    return SourceArtifact(
        role=role,
        # Manifest paths use forward slashes on every operating system so
        # provenance snapshots and tests are portable across rebuild hosts.
        path=path.resolve().as_posix(),
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _benchmark_files(
    source_path: Path, *, subset: str | None = None
) -> tuple[Path, Path]:
    if source_path.is_dir():
        direct_chunks = source_path / "chunks.json"
        direct_questions = source_path / "questions.json"
        nested = source_path / subset if subset else None
        if direct_chunks.is_file() and direct_questions.is_file():
            chunks_path = direct_chunks
            questions_path = direct_questions
        elif nested is not None:
            chunks_path = nested / "chunks.json"
            questions_path = nested / "questions.json"
        else:
            chunks_path = direct_chunks
            questions_path = direct_questions
    elif source_path.name == "chunks.json":
        chunks_path = source_path
        questions_path = source_path.with_name("questions.json")
    elif source_path.name == "questions.json":
        questions_path = source_path
        chunks_path = source_path.with_name("chunks.json")
    else:
        raise InputFormatError(
            "benchmark_exact source must be a directory containing chunks.json "
            "and questions.json, or either one of those files"
        )
    missing = [
        str(path) for path in (chunks_path, questions_path) if not path.is_file()
    ]
    if missing:
        raise InputFormatError(
            f"benchmark_exact source is missing required files: {missing}"
        )
    return chunks_path, questions_path


class BenchmarkExactAdapter:
    """Load one benchmark dataset as a global retrieval corpus.

    The collection uses a common ``chunks.json`` / ``questions.json``
    shape. Source Chunk boundaries and numeric source order are preserved. Gold
    questions and answers are emitted only through the evaluation sidecar.
    """

    def __init__(
        self,
        dataset: str | DatasetProfile,
        scope_id: str | None = None,
        *,
        validate_reference_counts: bool = False,
        _legacy_question_ids: bool = False,
        _document_title: str | None = None,
        _validate_task_types: bool = True,
        _reject_blank_answers: bool = True,
    ) -> None:
        self.profile = (
            dataset
            if isinstance(dataset, DatasetProfile)
            else get_dataset_profile(dataset)
        )
        self.dataset_name = self.profile.key
        self.scope_id = scope_id
        self.validate_reference_counts = validate_reference_counts
        self._legacy_question_ids = _legacy_question_ids
        self._document_title = _document_title
        self._validate_task_types = _validate_task_types
        self._reject_blank_answers = _reject_blank_answers

    def load(self, source_path: Path, split: str) -> AdapterOutput:
        chunks_path, questions_path = _benchmark_files(
            source_path, subset=self.profile.key
        )
        scope_id = self.scope_id or self.profile.scope_id(split)
        document_id = self.profile.document_id(split)

        try:
            with chunks_path.open("r", encoding="utf-8") as handle:
                raw_chunks = json.load(handle)
        except json.JSONDecodeError as exc:
            raise InputFormatError(f"Invalid JSON in {chunks_path}: {exc}") from exc
        if not isinstance(raw_chunks, list) or not all(
            isinstance(item, str) for item in raw_chunks
        ):
            raise InputFormatError(
                "benchmark_exact chunks.json must be a JSON list of 'id:text' strings"
            )

        parsed_chunks: list[tuple[int, str]] = []
        seen_source_ids: set[int] = set()
        for index, item in enumerate(raw_chunks):
            source_id, separator, text = item.partition(":")
            if not separator or not source_id.isdigit():
                raise InputFormatError(
                    f"benchmark_exact chunk {index} must use a numeric 'id:text' prefix"
                )
            numeric_id = int(source_id)
            if numeric_id in seen_source_ids:
                raise InputFormatError(
                    f"Duplicate benchmark_exact source Chunk ID: {source_id}"
                )
            if not text.strip():
                raise InputFormatError(
                    f"benchmark_exact source Chunk {source_id} has empty text"
                )
            seen_source_ids.add(numeric_id)
            parsed_chunks.append((numeric_id, text))

        source_positions = [item[0] for item in parsed_chunks]
        if source_positions != list(range(len(parsed_chunks))):
            raise InputFormatError(
                "benchmark_exact source Chunk IDs must be ordered and contiguous from 0"
            )
        chunks = tuple(
            RawChunk(
                chunk_id=self.profile.chunk_id(split, source_id),
                doc_id=document_id,
                chunk_pos=source_id,
                text=text,
            )
            for source_id, text in parsed_chunks
        )

        rows = _load_json_or_jsonl(questions_path)
        source_question_ids: list[str] = []
        for row_index, row in enumerate(rows):
            source_question_id = str(row.get("id") or row.get("_id") or "")
            if not source_question_id.strip():
                raise InputFormatError(
                    f"benchmark_exact question record {row_index} has no id"
                )
            source_question_ids.append(source_question_id)

        source_id_counts = Counter(source_question_ids)
        duplicate_source_ids = sorted(
            item for item, count in source_id_counts.items() if count > 1
        )
        if (
            duplicate_source_ids
            and self.profile.duplicate_question_id_policy
            is DuplicateQuestionIdPolicy.REJECT
        ):
            raise InputFormatError(
                f"Duplicate Benchmark {self.profile.key} question IDs: "
                f"{duplicate_source_ids}"
            )

        questions: list[BenchmarkQuestion] = []
        for row_index, row in enumerate(rows):
            source_question_id = source_question_ids[row_index]
            question_id = (
                source_question_id
                if self._legacy_question_ids
                else (f"{self.profile.key}:benchmark_exact:q:{row_index:06d}")
            )
            question = row.get("question")
            answer = row.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise InputFormatError(
                    "benchmark_exact question "
                    f"{source_question_id} has no question text"
                )
            if not isinstance(answer, str) or (
                self._reject_blank_answers and not answer.strip()
            ):
                raise InputFormatError(
                    "benchmark_exact question "
                    f"{source_question_id} has no non-empty string answer"
                )
            question_type = (
                normalize_task_type(
                    self.profile,
                    row.get("question_type"),
                    question_id=source_question_id,
                )
                if self._validate_task_types
                else (
                    str(row["question_type"])
                    if row.get("question_type") is not None
                    else None
                )
            )
            questions.append(
                BenchmarkQuestion(
                    question_id=question_id,
                    scope_id=scope_id,
                    source=(
                        str(row.get("source") or self.profile.key)
                        if self._legacy_question_ids
                        else self.profile.key
                    ),
                    question=question,
                    answer=answer,
                    question_type=question_type,
                    source_question_id=(
                        None if self._legacy_question_ids else source_question_id
                    ),
                    source_row_index=(None if self._legacy_question_ids else row_index),
                )
            )

        if self.validate_reference_counts:
            self._validate_reference_shape(
                chunk_count=len(chunks),
                questions=questions,
                unique_source_question_ids=len(source_id_counts),
            )

        return AdapterOutput(
            documents=(
                RawDocument(
                    doc_id=document_id,
                    title=self._document_title,
                    text="",
                ),
            ),
            scopes=(DocumentScope(scope_id=scope_id, doc_id=document_id),),
            gold_support=(),
            source_chunks=chunks,
            benchmark_questions=tuple(
                sorted(questions, key=lambda item: item.question_id)
            ),
            source_artifacts=(
                _source_artifact(chunks_path, "chunks"),
                _source_artifact(questions_path, "questions"),
            ),
            scope_mode="global",
            overlapping_chunks=True,
        )

    def _validate_reference_shape(
        self,
        *,
        chunk_count: int,
        questions: list[BenchmarkQuestion],
        unique_source_question_ids: int,
    ) -> None:
        errors: list[str] = []
        if chunk_count != self.profile.reference_chunk_count:
            errors.append(
                f"Chunks expected {self.profile.reference_chunk_count}, "
                f"found {chunk_count}"
            )
        if len(questions) != self.profile.reference_question_count:
            errors.append(
                f"questions expected {self.profile.reference_question_count}, "
                f"found {len(questions)}"
            )
        if unique_source_question_ids != self.profile.reference_unique_question_ids:
            errors.append(
                "unique question IDs expected "
                f"{self.profile.reference_unique_question_ids}, found "
                f"{unique_source_question_ids}"
            )
        actual_task_types = Counter(
            item.question_type for item in questions if item.question_type is not None
        )
        expected_task_types = dict(self.profile.reference_task_type_counts)
        if actual_task_types != expected_task_types:
            errors.append(
                f"task type counts expected {expected_task_types}, found "
                f"{dict(sorted(actual_task_types.items()))}"
            )
        if errors:
            raise InputFormatError(
                f"Benchmark {self.profile.key} reference profile mismatch: "
                + "; ".join(errors)
            )

    def iter_documents(self, source_path: Path, split: str) -> Iterator[RawDocument]:
        yield from self.load(source_path, split).documents


class HotpotQABenchmarkExactAdapter(BenchmarkExactAdapter):
    """Preserve the established HotpotQA sidecar ID contract."""

    dataset_name = "hotpotqa"

    def __init__(
        self,
        scope_id: str | None = None,
        *,
        validate_reference_counts: bool = False,
    ) -> None:
        super().__init__(
            "hotpotqa",
            scope_id,
            validate_reference_counts=validate_reference_counts,
            _legacy_question_ids=True,
            _document_title="HotpotQA reduced benchmark corpus",
            _validate_task_types=False,
            _reject_blank_answers=False,
        )


def document_scope_map(scopes: Iterable[DocumentScope]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for relation in scopes:
        result.setdefault(relation.scope_id, set()).add(relation.doc_id)
    return result
