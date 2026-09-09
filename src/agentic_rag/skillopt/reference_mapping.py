"""Read-only, evaluation-only mapping to original annotated evidence.

This module never rebuilds a corpus, loads an embedding model, or turns a text
similarity into source identity.  A mapping is accepted only from an existing
sentence-provenance sidecar, or from a titled document whose complete literal
source content can be verified.  Global untitled benchmark corpora therefore
remain explicitly unavailable unless an authoritative sidecar is supplied.

All offsets are zero-based, half-open offsets in the *original* Python strings.
Whitespace may be normalized to locate a range, but every emitted range is
checked for literal equality. HotpotQA's existing authoritative sidecar may
additionally verify explicitly labelled NFKC-equivalent whole sentences; their
original offsets are retained, including one-to-many compatibility characters.
No such normalization is applied to untitled or inferred source matching.
Facts, including unmapped facts, are never removed from a question.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REFERENCE_MAPPING_VERSION = "source-ranges-v2"
_DATASET_ALIASES = {"2wikimultihopqa": "2wikimultihop", "2wiki": "2wikimultihop"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class _Normalized:
    text: str
    offsets: tuple[tuple[int, int], ...]


def _normalize(text: str) -> _Normalized:
    """Collapse whitespace only; retain a checked offset table, not casefolding."""
    chars: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\s+|\S", text):
        value = match.group()
        if value.isspace():
            if chars:
                chars.append(" ")
                offsets.append((match.start(), match.end()))
        else:
            chars.append(value)
            offsets.append((match.start(), match.end()))
    if chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return _Normalized("".join(chars), tuple(offsets))


def _occurrences(haystack: str, needle: str) -> list[int]:
    if not needle:
        return []
    found: list[int] = []
    cursor = 0
    while (position := haystack.find(needle, cursor)) >= 0:
        found.append(position)
        cursor = position + 1
    return found


@dataclass(frozen=True)
class _Source:
    key: str
    title: str
    text: str
    sentences: tuple[str, ...] | None


@dataclass(frozen=True)
class _Unit:
    unit_id: str
    text: str
    doc_id: str
    chunk_id: str
    position: int
    kind: str


@dataclass(frozen=True)
class _Binding:
    unit: _Unit
    unit_norm_start: int
    source_norm_start: int
    length: int
    verification: str


@dataclass(frozen=True)
class _NFKCBinding:
    """An entire original sentence verified by the saved Hotpot sidecar."""
    unit: _Unit
    unit_start: int
    unit_end: int
    source_norm_start: int
    original_sentence: str


def _nfkc_text(text: str) -> str:
    return _normalize(unicodedata.normalize("NFKC", text)).text


def _source(title: str, text: str, sentences: Sequence[str] | None = None) -> _Source:
    values = tuple(sentences) if sentences is not None else None
    identity = [title, [_normalize(x).text for x in values] if values is not None else _normalize(text).text]
    return _Source(_fingerprint(identity), title, text, values)


def _context(raw: Mapping[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    context = raw.get("context")
    if isinstance(context, Mapping):
        titles = context.get("title", context.get("titles", []))
        sentences = context.get("sentences", [])
        if not isinstance(titles, list) or not isinstance(sentences, list) or len(titles) != len(sentences):
            return []
        context = list(zip(titles, sentences))
    result = []
    if isinstance(context, list):
        for item in context:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            title, texts = item
            if isinstance(title, str) and isinstance(texts, list) and all(isinstance(t, str) for t in texts):
                result.append((title, tuple(texts)))
    return result


def _supports(raw: Mapping[str, Any]) -> list[Any] | None:
    value = raw.get("supporting_facts")
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        titles = value.get("title", value.get("titles", []))
        positions = value.get("sent_id", value.get("sentence_id", []))
        if isinstance(titles, list) and isinstance(positions, list) and len(titles) == len(positions):
            return list(zip(titles, positions))
    return None


def _sentence_start(texts: Sequence[str], index: int) -> int:
    # The corpus adapter joins source sentences with whitespace. Blank entries
    # keep their original sentence number, but contribute no normalized chars.
    return sum(len(_normalize(x).text) + 1 for x in texts[:index] if _normalize(x).text)


def _covers_nonwhitespace(text: str, ranges: Sequence[tuple[int, int]]) -> bool:
    covered: set[int] = set()
    for start, end in ranges:
        covered.update(range(start, end))
    required = {i for i, char in enumerate(text) if not char.isspace()}
    return bool(required) and required <= covered


def _literal_ranges(binding: _Binding, fact_text: str, fact_norm_start: int) -> list[dict[str, Any]]:
    """Intersect a source binding with a fact and emit only proven equal chars."""
    fact_norm = _normalize(fact_text)
    left = max(binding.source_norm_start, fact_norm_start)
    right = min(binding.source_norm_start + binding.length, fact_norm_start + len(fact_norm.text))
    if left >= right:
        return []
    unit_norm = _normalize(binding.unit.text)
    pairs: list[tuple[int, int, int, int]] = []
    for source_pos in range(left, right):
        unit_pos = binding.unit_norm_start + source_pos - binding.source_norm_start
        fact_pos = source_pos - fact_norm_start
        if unit_norm.text[unit_pos] != fact_norm.text[fact_pos]:
            raise ValueError("verified source binding no longer matches fact")
        unit_start, unit_end = unit_norm.offsets[unit_pos]
        fact_start, fact_end = fact_norm.offsets[fact_pos]
        if binding.unit.text[unit_start:unit_end] != fact_text[fact_start:fact_end]:
            # Only whitespace is allowed to differ. It is not needed for the
            # coverage denominator, and cannot be advertised as a literal span.
            if unit_norm.text[unit_pos] != " ":
                raise ValueError("non-whitespace source character changed")
            continue
        if pairs and pairs[-1][1] == unit_start and pairs[-1][3] == fact_start:
            prior = pairs[-1]
            pairs[-1] = (prior[0], unit_end, prior[2], fact_end)
        else:
            pairs.append((unit_start, unit_end, fact_start, fact_end))
    return [
        {"unit_id": binding.unit.unit_id, "unit_type": binding.unit.kind,
         "unit_text": binding.unit.text, "unit_start": a, "unit_end": b,
         "fact_start": c, "fact_end": d, "verification": binding.verification}
        for a, b, c, d in pairs
        if any(not char.isspace() for char in fact_text[c:d])
    ]


class ReferenceMapper:
    """Immutable-input mapper; ``for_question`` performs no filesystem writes.

    ``source_row`` is the row number in the original annotation JSON, not the
    potentially different row number in reformatted benchmark questions.
    ``canonical_question`` must contain question/answer and should contain
    source_question_id. ID, question and answer mismatches are fatal join errors.
    """

    def __init__(self, dataset: str, substrate_path: Path, raw_source_path: Path) -> None:
        self.dataset = _DATASET_ALIASES.get(dataset.lower(), dataset.lower())
        if self.dataset not in {"hotpotqa", "2wikimultihop", "musique"}:
            raise ValueError(f"Unsupported exact-source dataset: {dataset}")
        self.substrate_path = Path(substrate_path)
        self.raw_source_path = Path(raw_source_path)
        self.rows = json.loads(self.raw_source_path.read_text(encoding="utf-8"))
        if not isinstance(self.rows, list) or not all(isinstance(row, dict) for row in self.rows):
            raise ValueError("original reference source must be a JSON list of objects")
        self.documents = self._read("nodes/documents.parquet")
        chunks = self._read("nodes/chunks.parquet")
        sentences = self._read("nodes/sentences.parquet")
        self.chunks = {
            str(row["chunk_id"]): _Unit(str(row["chunk_id"]), str(row["text"]), str(row["doc_id"]), str(row["chunk_id"]), int(row["chunk_pos"]), "CHUNK")
            for row in chunks
        }
        self.sentences = {}
        for row in sentences:
            chunk_id = str(row["chunk_id"])
            if chunk_id not in self.chunks:
                raise ValueError(f"sentence references missing chunk: {chunk_id}")
            self.sentences[str(row["sentence_id"])] = _Unit(str(row["sentence_id"]), str(row["text"]), self.chunks[chunk_id].doc_id, chunk_id, int(row["sentence_pos"]), "SENTENCE")
        self.sources: dict[str, _Source] = {}
        for row in self.rows:
            if self.dataset == "musique":
                for paragraph in row.get("paragraphs", []):
                    if isinstance(paragraph, Mapping) and isinstance(paragraph.get("title"), str) and isinstance(paragraph.get("paragraph_text"), str):
                        source = _source(paragraph["title"], paragraph["paragraph_text"])
                        self.sources[source.key] = source
            else:
                for title, texts in _context(row):
                    source = _source(title, " ".join(texts), texts)
                    self.sources[source.key] = source
        self.bindings: dict[str, list[_Binding]] = defaultdict(list)
        self.nfkc_bindings: dict[str, list[_NFKCBinding]] = defaultdict(list)
        self.mapping_issues: list[dict[str, Any]] = []
        self._bind_provenance()
        self._bind_complete_titled_documents()
        self.metadata = {
            "version": REFERENCE_MAPPING_VERSION, "dataset": self.dataset,
            "source_path": str(self.raw_source_path.resolve()), "source_sha256": _sha256(self.raw_source_path),
            "normalization": "whitespace_literal_or_explicit_nfkc_for_saved_hotpot_provenance_only",
            "index_modified": False, "embedding_used": False,
            "source_count": len(self.sources), "verified_source_count": len({
                key for key in self.sources if self.bindings.get(key) or self.nfkc_bindings.get(key)
            }),
            "nfkc_authoritative_binding_count": sum(len(value) for value in self.nfkc_bindings.values()),
            "mapping_issues": self.mapping_issues,
        }

    def _read(self, relative: str, *, optional: bool = False) -> list[dict[str, Any]]:
        path = self.substrate_path / relative
        if optional and not path.exists():
            return []
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()

    def _bind_provenance(self) -> None:
        rows = self._read("evaluation/source_sentence_provenance.parquet", optional=True)
        by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_doc[str(row["doc_id"])].append(row)
        doc_titles = {str(row["doc_id"]): row.get("title") for row in self.documents}
        for doc_id, records in by_doc.items():
            records.sort(key=lambda r: int(r["original_sentence_id"]))
            positions = [int(r["original_sentence_id"]) for r in records]
            titles = {r["original_title"] for r in records}
            if positions != list(range(len(records))) or titles != {doc_titles.get(doc_id)}:
                self.mapping_issues.append({"code": "invalid_provenance_document", "doc_id": doc_id})
                continue
            title = str(records[0]["original_title"])
            texts = tuple(str(r["original_sentence_text"]) for r in records)
            source = _source(title, " ".join(texts), texts)
            if source.key not in self.sources:
                self.mapping_issues.append({"code": "source_version_mismatch", "doc_id": doc_id})
                continue
            by_chunk: dict[str, list[tuple[_Unit, int]]] = defaultdict(list)
            nfkc_units: dict[str, str] = {}
            for record in records:
                source_pos = int(record["original_sentence_id"])
                unit = self.sentences.get(str(record.get("sentence_id")))
                if unit is None:
                    continue
                norm = _normalize(texts[source_pos]).text
                if unit.doc_id != doc_id:
                    self.mapping_issues.append({"code": "source_sentence_text_mismatch", "unit_id": unit.unit_id})
                    continue
                offset = _sentence_start(texts, source_pos)
                if _normalize(unit.text).text == norm:
                    self.bindings[source.key].append(_Binding(unit, 0, offset, len(norm), "saved_source_sentence_provenance"))
                elif self.dataset == "hotpotqa" and _nfkc_text(unit.text) == _nfkc_text(texts[source_pos]):
                    nfkc_units[unit.unit_id] = texts[source_pos]
                    self.nfkc_bindings[source.key].append(_NFKCBinding(unit, 0, len(unit.text), offset, texts[source_pos]))
                else:
                    self.mapping_issues.append({"code": "source_sentence_text_mismatch", "unit_id": unit.unit_id})
                    continue
                by_chunk[unit.chunk_id].append((unit, offset))
            for chunk_id, values in by_chunk.items():
                chunk = self.chunks[chunk_id]
                chunk_norm = _normalize(chunk.text)
                norm = chunk_norm.text
                ordered = sorted(values, key=lambda x: (x[1], x[0].unit_id))
                if norm != " ".join(_normalize(unit.text).text for unit, _ in ordered):
                    self.mapping_issues.append({"code": "chunk_sentence_order_mismatch", "unit_id": chunk_id})
                    continue
                cursor = 0
                for unit, offset in ordered:
                    sentence = _normalize(unit.text).text
                    location = norm.find(sentence, cursor)
                    if location < 0:
                        self.mapping_issues.append({"code": "chunk_sentence_order_mismatch", "unit_id": chunk_id})
                        continue
                    if unit.unit_id in nfkc_units:
                        self.nfkc_bindings[source.key].append(_NFKCBinding(
                            chunk, chunk_norm.offsets[location][0], chunk_norm.offsets[location + len(sentence) - 1][1],
                            offset, nfkc_units[unit.unit_id],
                        ))
                    else:
                        self.bindings[source.key].append(_Binding(chunk, location, offset, len(sentence), "saved_source_sentence_provenance"))
                    cursor = location + len(sentence)

    def _bind_complete_titled_documents(self) -> None:
        """No corpus-wide text lookup: verify title AND full source version."""
        by_title: dict[str, list[_Source]] = defaultdict(list)
        by_doc: dict[str, list[_Unit]] = defaultdict(list)
        for source in self.sources.values():
            by_title[source.title].append(source)
        for unit in self.chunks.values():
            by_doc[unit.doc_id].append(unit)
        for document in self.documents:
            title = document.get("title")
            if not isinstance(title, str) or not title:
                continue
            doc_id = str(document["doc_id"])
            chunks = sorted(by_doc.get(doc_id, []), key=lambda u: (u.position, u.unit_id))
            candidates: list[tuple[_Source, list[_Binding]]] = []
            for source in by_title.get(title, []):
                if (any(b.unit.doc_id == doc_id for b in self.bindings.get(source.key, []))
                    or any(b.unit.doc_id == doc_id for b in self.nfkc_bindings.get(source.key, []))):
                    continue
                body = _normalize(source.text).text
                bindings = []
                ranges = []
                prior_start = -1
                for chunk in chunks:
                    text = _normalize(chunk.text).text
                    matches = _occurrences(body, text)
                    if len(matches) != 1:
                        break
                    start = matches[0]
                    if start < prior_start:
                        break
                    prior_start = start
                    bindings.append(_Binding(chunk, 0, start, len(text), "verified_complete_titled_source"))
                    ranges.append((start, start + len(text)))
                else:
                    if _covers_nonwhitespace(body, ranges):
                        candidates.append((source, bindings))
            if len(candidates) != 1:
                if len(candidates) > 1:
                    self.mapping_issues.append({"code": "ambiguous_source_version", "doc_id": doc_id})
                continue
            source, bindings = candidates[0]
            self.bindings[source.key].extend(bindings)
            for binding in bindings:
                chunk = binding.unit
                chunk_norm = _normalize(chunk.text).text
                cursor = 0
                children = sorted((s for s in self.sentences.values() if s.chunk_id == chunk.unit_id), key=lambda s: (s.position, s.unit_id))
                for sentence in children:
                    text = _normalize(sentence.text).text
                    position = chunk_norm.find(text, cursor)
                    if position < 0:
                        self.mapping_issues.append({"code": "chunk_sentence_order_mismatch", "unit_id": sentence.unit_id})
                        continue
                    self.bindings[source.key].append(_Binding(sentence, 0, binding.source_norm_start + position, len(text), "verified_complete_titled_source"))
                    cursor = position + len(text)

    def for_question(self, canonical_question: Mapping[str, Any], source_row: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if isinstance(source_row, bool) or not isinstance(source_row, int) or not 0 <= source_row < len(self.rows):
            raise ValueError("source_row must identify an existing original source row")
        raw = self.rows[source_row]
        raw_id = str(raw.get("_id", raw.get("id", "")))
        for field in ("question", "answer"):
            if canonical_question.get(field) != raw.get(field):
                raise ValueError(f"original source {field} mismatch at row {source_row}")
        claimed_id = canonical_question.get("source_question_id")
        if claimed_id is not None:
            claimed_id = str(claimed_id)
            if self.dataset == "musique" and claimed_id.startswith("musique_"):
                claimed_id = claimed_id.removeprefix("musique_")
            if claimed_id != raw_id:
                raise ValueError(f"original source ID mismatch at row {source_row}")
        question_id = str(canonical_question.get("id", canonical_question.get("question_id", raw_id)))
        issues: list[dict[str, Any]] = []
        facts = self._paragraph_facts(raw, source_row, issues) if self.dataset == "musique" else self._sentence_facts(raw, source_row, issues)
        if not facts:
            issues.append({"code": "missing_gold_reference", "message": "No non-empty original annotated evidence; this is not zero coverage."})
        for fact in facts:
            mappings = fact["mappings"]
            complete = _covers_nonwhitespace(fact["text"], [(m["fact_start"], m["fact_end"]) for m in mappings])
            fact["mapping_status"] = "ready" if complete else "unavailable"
            if not complete and fact.get("annotation_status") != "invalid":
                issues.append({"code": "unmapped_gold_reference", "fact_id": fact["fact_id"], "source": fact["source"], "mapped_nonwhitespace_characters": len({i for m in mappings for i in range(m["fact_start"], m["fact_end"]) if not fact["text"][i].isspace()}), "total_nonwhitespace_characters": sum(not c.isspace() for c in fact["text"])})
        for issue in issues:
            issue.update({"question_id": question_id, "raw_question_id": raw_id, "source_row_index": source_row})
        ready = bool(facts) and not issues
        reference = {
            "version": REFERENCE_MAPPING_VERSION,
            "kind": "paragraph" if self.dataset == "musique" else "sentence",
            "method": "paragraph_span" if self.dataset == "musique" else "exact_source",
            "status": "ready" if ready else "unavailable",
            "facts": facts, "source_sha256": self.metadata["source_sha256"],
            "source_row_index": source_row, "raw_question_id": raw_id,
        }
        if issues:
            reference["reason"] = "; ".join(sorted({str(issue["code"]) for issue in issues}))
        return reference, issues

    def _fact(self, ordinal: int, text: str, source: _Source | None, source_start: int, metadata: dict[str, Any]) -> dict[str, Any]:
        mappings = []
        if source is not None:
            metadata["source_content_sha256"] = source.key
            for binding in self.bindings.get(source.key, []):
                mappings.extend(_literal_ranges(binding, text, source_start))
            for binding in self.nfkc_bindings.get(source.key, []):
                if binding.source_norm_start != source_start or _normalize(binding.original_sentence).text != _normalize(text).text:
                    continue
                segment = binding.unit.text[binding.unit_start:binding.unit_end]
                if _nfkc_text(segment) != _nfkc_text(text):
                    raise ValueError("authoritative NFKC source binding no longer matches original sentence")
                mappings.append({
                    "unit_id": binding.unit.unit_id, "unit_type": binding.unit.kind,
                    "unit_text": binding.unit.text, "unit_start": binding.unit_start, "unit_end": binding.unit_end,
                    "fact_start": 0, "fact_end": len(text), "normalization": "NFKC",
                    "verification": "saved_source_sentence_provenance",
                })
        unique = {(m["unit_id"], m["unit_start"], m["unit_end"], m["fact_start"], m["fact_end"]): m for m in mappings}
        return {"fact_id": f"gold:{ordinal:04d}", "text": text, "source": metadata, "mappings": [unique[key] for key in sorted(unique)]}

    def _sentence_facts(self, raw: Mapping[str, Any], row: int, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        contexts: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for title, texts in _context(raw):
            if texts not in contexts[title]:
                contexts[title].append(texts)
        supports = _supports(raw)
        if supports is None:
            issues.append({"code": "invalid_gold_annotation", "message": "supporting_facts is missing or malformed"})
            return []
        facts = []
        for ordinal, pair in enumerate(supports):
            metadata = {"source_row_index": row, "original_title": None, "original_sentence_id": None}
            valid = isinstance(pair, (tuple, list)) and len(pair) == 2
            if valid:
                title, index = pair
                metadata.update(original_title=title, original_sentence_id=index)
                if isinstance(title, str) and len(contexts.get(title, [])) > 1:
                    # A valid sentence number cannot disambiguate two different
                    # originals bearing the same title. Do not misclassify this
                    # as one of the coordinate-invalid questions to exclude.
                    fact = self._fact(ordinal, "", None, 0, metadata)
                    fact["annotation_status"] = "unavailable"
                    fact["source"]["candidate_source_count"] = len(contexts[title])
                    facts.append(fact)
                    issues.append({"code": "ambiguous_gold_source", "fact_id": fact["fact_id"], "source": metadata})
                    continue
                valid = isinstance(title, str) and isinstance(index, int) and not isinstance(index, bool) and len(contexts.get(title, [])) == 1 and 0 <= index < len(contexts[title][0])
            if not valid:
                fact = self._fact(ordinal, "", None, 0, metadata)
                fact["annotation_status"] = "invalid"
                facts.append(fact)
                issues.append({"code": "invalid_gold_coordinate", "fact_id": fact["fact_id"], "source": metadata, "annotation": pair})
                continue
            texts = contexts[title][0]
            text = texts[index]
            source = _source(title, " ".join(texts), texts)
            fact = self._fact(ordinal, text, source, _sentence_start(texts, index), metadata)
            if not text.strip():
                fact["annotation_status"] = "invalid"
                issues.append({"code": "empty_gold_reference", "fact_id": fact["fact_id"]})
            facts.append(fact)
        return facts

    def _paragraph_facts(self, raw: Mapping[str, Any], row: int, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        paragraphs = raw.get("paragraphs")
        if not isinstance(paragraphs, list):
            issues.append({"code": "invalid_gold_annotation", "message": "paragraphs is missing"})
            return []
        seen = set()
        supporting = []
        for paragraph in paragraphs:
            if not isinstance(paragraph, Mapping) or not isinstance(paragraph.get("idx"), int) or isinstance(paragraph.get("idx"), bool) or not isinstance(paragraph.get("is_supporting"), bool):
                issues.append({"code": "invalid_gold_annotation", "message": "paragraph index or support flag is invalid"})
                continue
            index = paragraph["idx"]
            if index in seen:
                issues.append({"code": "invalid_gold_annotation", "message": "duplicate original paragraph index", "paragraph_idx": index})
            seen.add(index)
            if paragraph["is_supporting"]:
                supporting.append(paragraph)
        decomposition = raw.get("question_decomposition")
        if not isinstance(decomposition, list) or not all(isinstance(x, Mapping) and isinstance(x.get("paragraph_support_idx"), int) and not isinstance(x.get("paragraph_support_idx"), bool) for x in decomposition):
            issues.append({"code": "invalid_gold_annotation", "message": "question_decomposition support indices are missing or invalid"})
        elif {x["paragraph_support_idx"] for x in decomposition} != {p["idx"] for p in supporting}:
            issues.append({"code": "support_annotation_mismatch", "message": "is_supporting and decomposition disagree"})
        facts = []
        for ordinal, paragraph in enumerate(supporting):
            metadata = {"source_row_index": row, "original_title": paragraph.get("title"), "original_paragraph_id": paragraph["idx"]}
            text, title = paragraph.get("paragraph_text"), paragraph.get("title")
            if not isinstance(text, str) or not text.strip() or not isinstance(title, str) or not title:
                fact = self._fact(ordinal, text if isinstance(text, str) else "", None, 0, metadata)
                fact["annotation_status"] = "invalid"
                issues.append({"code": "empty_gold_reference", "fact_id": fact["fact_id"]})
            else:
                fact = self._fact(ordinal, text, _source(title, text), 0, metadata)
            facts.append(fact)
        return facts


def build_reference_mapper(dataset: str, substrate_path: str | Path, raw_source_path: str | Path) -> ReferenceMapper:
    """Build an evaluation-only mapper without touching an index or embeddings."""
    return ReferenceMapper(dataset, Path(substrate_path), Path(raw_source_path))
