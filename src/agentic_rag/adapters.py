"""Source adapters and HotpotQA parsing."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from agentic_rag.errors import InputFormatError
from agentic_rag.models import (
    BenchmarkQuestion,
    DocumentScope,
    GoldSupport,
    RawChunk,
    RawDocument,
    SourceArtifact,
)


@dataclass(frozen=True, slots=True)
class AdapterOutput:
    documents: tuple[RawDocument, ...]
    scopes: tuple[DocumentScope, ...]
    gold_support: tuple[GoldSupport, ...]
    source_chunks: tuple[RawChunk, ...] = ()
    benchmark_questions: tuple[BenchmarkQuestion, ...] = ()
    source_artifacts: tuple[SourceArtifact, ...] = ()


class SourceAdapter(Protocol):
    dataset_name: str

    def load(self, source_path: Path, split: str) -> AdapterOutput:
        ...

    def iter_documents(
        self, source_path: Path, split: str
    ) -> Iterator[RawDocument]:
        ...


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
            "HotpotQA input must be a JSON array, JSONL objects, or an object "
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
            contexts.append(
                (str(entry[0]), [str(sentence) for sentence in entry[1]])
            )
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

    def iter_documents(
        self, source_path: Path, split: str
    ) -> Iterator[RawDocument]:
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
        path=str(path.resolve()),
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _benchmark_files(source_path: Path) -> tuple[Path, Path]:
    if source_path.is_dir():
        chunks_path = source_path / "chunks.json"
        questions_path = source_path / "questions.json"
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
    missing = [str(path) for path in (chunks_path, questions_path) if not path.is_file()]
    if missing:
        raise InputFormatError(
            f"benchmark_exact source is missing required files: {missing}"
        )
    return chunks_path, questions_path


class HotpotQABenchmarkExactAdapter:
    """Load the reduced A-RAG/LinearRAG benchmark as one global corpus.

    The supplied 1,000-token Chunk boundaries and numeric adjacency are kept
    intact. Question-to-context mappings are deliberately not represented in
    retrieval scopes; all questions search the same corpus.
    """

    dataset_name = "hotpotqa"

    def __init__(self, scope_id: str | None = None) -> None:
        self.scope_id = scope_id

    def load(self, source_path: Path, split: str) -> AdapterOutput:
        chunks_path, questions_path = _benchmark_files(source_path)
        scope_id = self.scope_id or f"hotpotqa:benchmark_exact:{split}"
        document_id = f"hotpotqa:{split}:benchmark_exact:corpus"

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
            if not text:
                raise InputFormatError(
                    f"benchmark_exact source Chunk {source_id} has empty text"
                )
            seen_source_ids.add(numeric_id)
            parsed_chunks.append((numeric_id, text))

        source_positions = [item[0] for item in parsed_chunks]
        if source_positions != list(range(len(parsed_chunks))):
            raise InputFormatError(
                "benchmark_exact source Chunk IDs must be ordered and contiguous "
                "from 0"
            )
        chunks = tuple(
            RawChunk(
                chunk_id=f"hotpotqa:{split}:benchmark_exact:c:{source_id:06d}",
                doc_id=document_id,
                chunk_pos=source_id,
                text=text,
            )
            for source_id, text in parsed_chunks
        )

        rows = _load_json_or_jsonl(questions_path)
        questions: list[BenchmarkQuestion] = []
        seen_question_ids: set[str] = set()
        for row_index, row in enumerate(rows):
            question_id = str(row.get("id") or row.get("_id") or "")
            if not question_id:
                raise InputFormatError(
                    f"benchmark_exact question record {row_index} has no id"
                )
            if question_id in seen_question_ids:
                raise InputFormatError(
                    f"Duplicate benchmark_exact question ID: {question_id}"
                )
            question = row.get("question")
            answer = row.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise InputFormatError(
                    f"benchmark_exact question {question_id} has no question text"
                )
            if not isinstance(answer, str):
                raise InputFormatError(
                    f"benchmark_exact question {question_id} has no string answer"
                )
            seen_question_ids.add(question_id)
            questions.append(
                BenchmarkQuestion(
                    question_id=question_id,
                    scope_id=scope_id,
                    source=str(row.get("source") or "hotpotqa"),
                    question=question,
                    answer=answer,
                    question_type=(
                        str(row["question_type"])
                        if row.get("question_type") is not None
                        else None
                    ),
                )
            )

        return AdapterOutput(
            documents=(
                RawDocument(
                    doc_id=document_id,
                    title="HotpotQA reduced benchmark corpus",
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
        )

    def iter_documents(
        self, source_path: Path, split: str
    ) -> Iterator[RawDocument]:
        yield from self.load(source_path, split).documents


def document_scope_map(scopes: Iterable[DocumentScope]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for relation in scopes:
        result.setdefault(relation.scope_id, set()).add(relation.doc_id)
    return result
