from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agentic_rag.skillopt.reference_mapping import build_reference_mapper


def _table(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows) if rows else pa.table({column: pa.array([], type=pa.string()) for column in columns})
    pq.write_table(table, path)


def _fixture(tmp_path: Path, raw: list[dict], *, title: str | None, chunks: list[str], provenance: list[dict] | None = None, sentence_texts: list[list[str]] | None = None) -> tuple[Path, Path]:
    root = tmp_path / "index"
    _table(root / "nodes/documents.parquet", [{"doc_id": "doc:1", "title": title}], ["doc_id", "title"])
    _table(root / "nodes/chunks.parquet", [{"doc_id": "doc:1", "chunk_id": f"c:{i}", "chunk_pos": i, "text": text} for i, text in enumerate(chunks)], ["doc_id", "chunk_id", "chunk_pos", "text"])
    if sentence_texts is None:
        sentence_texts = [[chunk] for chunk in chunks]
    _table(root / "nodes/sentences.parquet", [{"sentence_id": f"s:{ci}:{si}", "chunk_id": f"c:{ci}", "sentence_pos": si, "text": text} for ci, texts in enumerate(sentence_texts) for si, text in enumerate(texts)], ["sentence_id", "chunk_id", "sentence_pos", "text"])
    if provenance is not None:
        _table(root / "evaluation/source_sentence_provenance.parquet", provenance, ["doc_id", "sentence_id", "original_title", "original_sentence_id", "original_sentence_text"])
    (root / "unchanged_embedding.bin").write_bytes(b"existing frozen vector bytes")
    source = tmp_path / "raw.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    return root, source


def _wiki(context: list, supports: list, *, identifier="raw:1") -> dict:
    return {"_id": identifier, "question": "Who?", "answer": "Alice", "context": context, "supporting_facts": supports}


def _mu(paragraphs: list[dict], *, indices: list[int] | None = None) -> dict:
    indices = [p["idx"] for p in paragraphs if p["is_supporting"]] if indices is None else indices
    return {"id": "raw:1", "question": "Who?", "answer": "Alice", "paragraphs": paragraphs, "question_decomposition": [{"paragraph_support_idx": index} for index in indices]}


def _paragraph(index: int, text: str, *, title="Book", support=True) -> dict:
    return {"idx": index, "title": title, "paragraph_text": text, "is_supporting": support}


def _question(*, raw_id="raw:1") -> dict:
    return {"id": "canonical:000000", "source_question_id": raw_id, "question": "Who?", "answer": "Alice"}


def _provenance(texts: list[str], *, title="Book") -> list[dict]:
    result = []
    for i, text in enumerate(texts):
        result.append({"doc_id": "doc:1", "sentence_id": f"s:0:{i}" if text.strip() else None, "original_title": title, "original_sentence_id": i, "original_sentence_text": text})
    return result


def _all_ranges_literal(reference: dict) -> None:
    for fact in reference["facts"]:
        for mapping in fact["mappings"]:
            assert mapping["unit_text"][mapping["unit_start"]:mapping["unit_end"]] == fact["text"][mapping["fact_start"]:mapping["fact_end"]]


def _snapshot(root: Path) -> dict:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}


def test_hotpot_saved_sentence_and_chunk_mapping_are_read_only(tmp_path):
    texts = ["Alice founded Acme.", "Acme is in Paris."]
    raw = _wiki([["Book", texts]], [["Book", 1]])
    root, source = _fixture(tmp_path, [raw], title="Book", chunks=[" ".join(texts)], sentence_texts=[texts], provenance=_provenance(texts))
    before = _snapshot(root)
    mapper = build_reference_mapper("hotpotqa", root, source)
    reference, issues = mapper.for_question(_question(), 0)
    assert reference["status"] == "ready"
    assert issues == []
    assert {m["unit_id"] for m in reference["facts"][0]["mappings"]} == {"s:0:1", "c:0"}
    chunk = next(m for m in reference["facts"][0]["mappings"] if m["unit_id"] == "c:0")
    assert chunk["unit_start"] == len(texts[0]) + 1
    _all_ranges_literal(reference)
    assert before == _snapshot(root)
    assert mapper.metadata["embedding_used"] is False


def test_whitespace_only_alignment_emits_verified_original_offsets(tmp_path):
    text = "  Alice\t founded   Acme.  "
    raw = _wiki([["Book", [text]]], [["Book", 0]])
    root, source = _fixture(tmp_path, [raw], title="Book", chunks=["Alice founded Acme."], provenance=_provenance([text]))
    reference, issues = build_reference_mapper("hotpotqa", root, source).for_question(_question(), 0)
    assert reference["status"] == "ready"
    assert not issues
    _all_ranges_literal(reference)
    assert any(m["fact_start"] == 2 for m in reference["facts"][0]["mappings"])


def test_nonwhitespace_normalization_does_not_forge_exact_mapping(tmp_path):
    text = "Alice founded Acme."
    root, source = _fixture(tmp_path, [_wiki([["Book", [text]]], [["Book", 0]])], title="Book", chunks=[text.lower()], provenance=_provenance([text]))
    reference, issues = build_reference_mapper("hotpotqa", root, source).for_question(_question(), 0)
    assert reference["status"] == "unavailable"
    assert reference["facts"][0]["mappings"] == []
    assert "unmapped_gold_reference" in {i["code"] for i in issues}


def test_untitled_corpus_exact_common_text_is_not_source_identity(tmp_path):
    text = "Alice founded Acme."
    root, source = _fixture(tmp_path, [_wiki([["Book", [text]]], [["Book", 0]])], title=None, chunks=[text])
    reference, _ = build_reference_mapper("2wiki", root, source).for_question(_question(), 0)
    assert reference["status"] == "unavailable"
    assert reference["facts"][0]["text"] == text
    assert reference["facts"][0]["mappings"] == []


def test_same_title_alone_does_not_match_wrong_source_version(tmp_path):
    text = "Alice founded Acme."
    raw = _wiki([["Book", [text, "This version is old."]]], [["Book", 0]])
    root, source = _fixture(tmp_path, [raw], title="Book", chunks=[text + " This version is new."])
    reference, _ = build_reference_mapper("2wikimultihop", root, source).for_question(_question(), 0)
    assert reference["status"] == "unavailable"
    assert not reference["facts"][0]["mappings"]


def test_ambiguous_raw_source_is_not_coordinate_invalid_exclusion(tmp_path):
    raw = _wiki([["Book", ["Alice founded Acme."]], ["Book", ["Bob founded Acme."]]], [["Book", 0]])
    root, source = _fixture(tmp_path, [raw], title="Book", chunks=["Alice founded Acme."])
    reference, issues = build_reference_mapper("2wiki", root, source).for_question(_question(), 0)
    assert reference["status"] == "unavailable"
    assert len(reference["facts"]) == 1
    codes = {i["code"] for i in issues}
    assert "ambiguous_gold_source" in codes
    assert "invalid_gold_coordinate" not in codes


def test_2wiki_invalid_coordinates_retain_full_fact_denominator(tmp_path):
    raw = _wiki([["Book", ["Alice founded Acme."]]], [["Book", 0], ["Book", 1]])
    root, source = _fixture(tmp_path, [raw], title="Book", chunks=["Alice founded Acme."])
    reference, issues = build_reference_mapper("2wikimultihop", root, source).for_question(_question(), 0)
    assert reference["status"] == "unavailable"
    assert len(reference["facts"]) == 2
    assert reference["facts"][0]["mapping_status"] == "ready"
    assert reference["facts"][1]["annotation_status"] == "invalid"
    assert "invalid_gold_coordinate" in {i["code"] for i in issues}


def test_blank_original_sentence_preserves_following_sentence_number(tmp_path):
    texts = ["", "Alice founded Acme."]
    raw = _wiki([["Book", texts]], [["Book", 1]])
    root, source = _fixture(tmp_path, [raw], title="Book", chunks=[texts[1]])
    reference, issues = build_reference_mapper("2wiki", root, source).for_question(_question(), 0)
    assert reference["status"] == "ready"
    assert reference["facts"][0]["source"]["original_sentence_id"] == 1
    assert not issues
    _all_ranges_literal(reference)


@pytest.mark.parametrize("supports", [[], [["Book", 0]]])
def test_empty_gold_is_unavailable_not_perfect_or_zero_coverage(tmp_path, supports):
    root, source = _fixture(tmp_path, [_wiki([["Book", [""]]], supports)], title="Book", chunks=["Unrelated text."])
    reference, issues = build_reference_mapper("2wiki", root, source).for_question(_question(), 0)
    assert reference["status"] == "unavailable"
    assert {i["code"] for i in issues} & {"missing_gold_reference", "empty_gold_reference"}
    assert "coverage" not in reference


def test_missing_second_source_never_shrinks_denominator(tmp_path):
    raw = _wiki([["Book", ["Alice founded Acme."]], ["Other", ["Acme is in Paris."]]], [["Book", 0], ["Other", 0]])
    root, source = _fixture(tmp_path, [raw], title="Book", chunks=["Alice founded Acme."])
    reference, issues = build_reference_mapper("2wiki", root, source).for_question(_question(), 0)
    assert reference["status"] == "unavailable"
    assert len(reference["facts"]) == 2
    assert [f["mapping_status"] for f in reference["facts"]] == ["ready", "unavailable"]
    assert len([i for i in issues if i["code"] == "unmapped_gold_reference"]) == 1


def test_musique_paragraph_span_crosses_multiple_frozen_chunks(tmp_path):
    first, second = "Alice founded Acme.", "Acme is in Paris."
    text = first + " " + second
    raw = _mu([_paragraph(7, text)])
    root, source = _fixture(tmp_path, [raw], title="Book", chunks=[first, second])
    reference, issues = build_reference_mapper("musique", root, source).for_question(_question(raw_id="musique_raw:1"), 0)
    assert reference["status"] == "ready"
    assert reference["kind"] == "paragraph"
    assert reference["method"] == "paragraph_span"
    assert not issues
    mappings = reference["facts"][0]["mappings"]
    assert {m["unit_id"] for m in mappings} == {"c:0", "c:1", "s:0:0", "s:1:0"}
    assert next(m for m in mappings if m["unit_id"] == "c:1")["fact_start"] == len(first) + 1
    _all_ranges_literal(reference)


def test_musique_same_title_different_paragraphs_do_not_merge(tmp_path):
    p1, p2 = _paragraph(1, "Alice founded Acme."), _paragraph(2, "Acme is in Paris.")
    root, source = _fixture(tmp_path, [_mu([p1, p2])], title="Book", chunks=[p1["paragraph_text"]])
    reference, _ = build_reference_mapper("musique", root, source).for_question(_question(), 0)
    assert reference["status"] == "unavailable"
    facts = reference["facts"]
    assert len(facts) == 2
    assert facts[0]["mappings"] and not facts[1]["mappings"]
    assert facts[0]["source"]["source_content_sha256"] != facts[1]["source"]["source_content_sha256"]


def test_duplicate_literal_paragraphs_keep_all_original_coordinates(tmp_path):
    text = "Alice founded Acme."
    raw = _mu([_paragraph(3, text), _paragraph(9, text)])
    root, source = _fixture(tmp_path, [raw], title="Book", chunks=[text])
    reference, issues = build_reference_mapper("musique", root, source).for_question(_question(), 0)
    assert reference["status"] == "ready"
    assert len(reference["facts"]) == 2
    assert {f["source"]["original_paragraph_id"] for f in reference["facts"]} == {3, 9}
    assert not issues


def test_musique_decomposition_disagreement_is_reported(tmp_path):
    text = "Alice founded Acme."
    raw = _mu([_paragraph(1, text)], indices=[2])
    root, source = _fixture(tmp_path, [raw], title="Book", chunks=[text])
    reference, issues = build_reference_mapper("musique", root, source).for_question(_question(), 0)
    assert reference["status"] == "unavailable"
    assert len(reference["facts"]) == 1
    assert "support_annotation_mismatch" in {i["code"] for i in issues}


def test_reversed_chunk_order_cannot_prove_complete_source(tmp_path):
    first, second = "Alice founded Acme.", "Acme is in Paris."
    root, source = _fixture(tmp_path, [_mu([_paragraph(1, first + " " + second)])], title="Book", chunks=[second, first])
    reference, _ = build_reference_mapper("musique", root, source).for_question(_question(), 0)
    assert reference["status"] == "unavailable"
    assert not reference["facts"][0]["mappings"]


def test_ambiguous_repeated_substring_cannot_be_used_as_partial_source(tmp_path):
    text = "Alice founded Acme. Alice founded Acme."
    root, source = _fixture(tmp_path, [_mu([_paragraph(1, text)])], title="Book", chunks=["Alice founded Acme.", "Alice founded Acme."])
    reference, _ = build_reference_mapper("musique", root, source).for_question(_question(), 0)
    assert reference["status"] == "unavailable"
    assert not reference["facts"][0]["mappings"]


def test_saved_provenance_source_version_must_match_raw(tmp_path):
    texts = ["Alice founded Acme.", "This version is new."]
    raw = _wiki([["Book", [texts[0], "This version is old."]]], [["Book", 0]])
    root, source = _fixture(tmp_path, [raw], title="Book", chunks=[" ".join(texts)], sentence_texts=[texts], provenance=_provenance(texts))
    mapper = build_reference_mapper("hotpotqa", root, source)
    reference, _ = mapper.for_question(_question(), 0)
    assert reference["status"] == "unavailable"
    assert any(issue["code"] == "source_version_mismatch" for issue in mapper.metadata["mapping_issues"])


@pytest.mark.parametrize("changed", [{"question": "Different?"}, {"answer": "Bob"}, {"source_question_id": "wrong"}])
def test_join_mismatch_is_fatal_not_silent_remapping(tmp_path, changed):
    text = "Alice founded Acme."
    root, source = _fixture(tmp_path, [_wiki([["Book", [text]]], [["Book", 0]])], title="Book", chunks=[text])
    mapper = build_reference_mapper("2wiki", root, source)
    with pytest.raises(ValueError, match="mismatch"):
        mapper.for_question({**_question(), **changed}, 0)


def test_reference_mapping_is_deterministic(tmp_path):
    text = "Alice founded Acme."
    root, source = _fixture(tmp_path, [_wiki([["Book", [text]]], [["Book", 0]])], title="Book", chunks=[text])
    mapper = build_reference_mapper("2wiki", root, source)
    first = mapper.for_question(_question(), 0)
    second = build_reference_mapper("2wiki", root, source).for_question(_question(), 0)
    assert first == second


@pytest.mark.parametrize("original,runtime", [
    ("It covers 5 km².", "It covers 5 km2."),
    ("Wait… now.", "Wait... now."),
])
def test_hotpot_authoritative_sidecar_preserves_explicit_nfkc_ranges_without_changing_index(tmp_path, original, runtime):
    raw = _wiki([["Book", [original]]], [["Book", 0]])
    root, source = _fixture(tmp_path, [raw], title="Book", chunks=[runtime], provenance=_provenance([original]))
    before = _snapshot(root)
    mapper = build_reference_mapper("hotpotqa", root, source)
    reference, issues = mapper.for_question(_question(), 0)
    assert reference["status"] == "ready"
    assert issues == []
    assert reference["facts"][0]["text"] == original
    mappings = reference["facts"][0]["mappings"]
    assert {mapped["unit_id"] for mapped in mappings} == {"s:0:0", "c:0"}
    for mapped in mappings:
        assert mapped["normalization"] == "NFKC"
        assert mapped["verification"] == "saved_source_sentence_provenance"
        shown = mapped["unit_text"][mapped["unit_start"]:mapped["unit_end"]]
        fact = original[mapped["fact_start"]:mapped["fact_end"]]
        assert unicodedata.normalize("NFKC", shown) == unicodedata.normalize("NFKC", fact)
    assert mapper.metadata["nfkc_authoritative_binding_count"] == 2
    assert _snapshot(root) == before


def test_nfkc_expansion_does_not_shift_subsequent_original_sentence_offsets(tmp_path):
    originals = ["Wait… now.", "It covers 5 km².", "Alice founded Acme."]
    runtime = [unicodedata.normalize("NFKC", text) for text in originals]
    raw = _wiki([["Book", originals]], [["Book", index] for index in range(3)])
    root, source = _fixture(tmp_path, [raw], title="Book", chunks=[" ".join(runtime)],
                            sentence_texts=[runtime], provenance=_provenance(originals))
    reference, issues = build_reference_mapper("hotpotqa", root, source).for_question(_question(), 0)
    assert reference["status"] == "ready"
    assert not issues
    last = reference["facts"][2]
    chunk = next(mapped for mapped in last["mappings"] if mapped["unit_id"] == "c:0")
    assert chunk["unit_start"] == len(runtime[0]) + len(runtime[1]) + 2
    assert chunk["unit_text"][chunk["unit_start"]:chunk["unit_end"]] == originals[2]
    assert "normalization" not in chunk


@pytest.mark.parametrize("dataset,with_sidecar", [("hotpotqa", False), ("2wiki", False), ("2wiki", True)])
def test_nfkc_is_not_a_general_source_matching_fallback(tmp_path, dataset, with_sidecar):
    original, runtime = "It covers 5 km².", "It covers 5 km2."
    root, source = _fixture(tmp_path, [_wiki([["Book", [original]]], [["Book", 0]])],
                            title="Book", chunks=[runtime],
                            provenance=_provenance([original]) if with_sidecar else None)
    reference, _ = build_reference_mapper(dataset, root, source).for_question(_question(), 0)
    assert reference["status"] == "unavailable"
    assert reference["facts"][0]["mappings"] == []
