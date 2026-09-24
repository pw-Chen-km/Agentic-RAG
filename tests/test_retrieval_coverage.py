import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from agentic_rag.evaluation.retrieval_coverage import CoverageAuditor, summarize
from agentic_rag.substrate.storage import Substrate


def test_person_query_and_visible_entity_coverage_are_offline(built_substrate):
    substrate = Substrate.open(built_substrate)
    auditor = CoverageAuditor(substrate)
    scope_id = next(scope for scope, docs in substrate.doc_ids_by_scope.items()
                    if any(substrate.document_by_id[doc].title == "Marie Curie" for doc in docs))
    entity = next(item for item in substrate.entities if item.canonical_name == "Marie Curie")
    mention = next(item for item in substrate.mentions if item.entity_id == entity.entity_id)
    sentence = substrate.sentence_by_id[mention.sentence_id]
    first = {"context_reference_map": {"typed_refs": {"E1": {
        "node_type": "ENTITY", "stable_id": entity.entity_id}}},
        "visible_source_spans": [],
        "resolved_decision": {"action": {"type": "SEARCH", "query": "Marie Curie birthplace"}},
        "observation": {"status": "ok", "results": [{"title": "Marie Curie", "text": sentence.text}]}}
    second = {"context_reference_map": {"typed_refs": {}},
        "visible_source_spans": [{"sentence_id": sentence.sentence_id, "visible": True,
            "complete": True, "start": 0, "end": len(sentence.text)}],
        "decision": {"action": {"type": "EXPAND", "source_ref": "E1", "query": None}},
        "resolved_decision": {"action": {"type": "EXPAND",
            "kind": "ENTITY_MENTIONED_IN_SENTENCE", "source_id": entity.entity_id, "query": None}},
        "observation": {"status": "ok", "results": [{"sentence_id": sentence.sentence_id}]}}
    episode = {"episode_id": "C3--q1", "scope_id": scope_id, "trajectory": [first, second]}
    result = auditor.audit_episode(episode, "hotpotqa")
    query = result["person_queries"][0]
    assert query["person"] == "Marie Curie"
    assert query["top5_contains_person"] is True
    assert query["matching_document_ids"]
    assert query["text_mention_present"] is True
    assert result["visible_entities"][0]["sentence_candidates"] >= 1
    assert result["navigations"][0]["entity_ref"] == "E1"
    assert result["navigations"][0]["old_source_ratio"] == 1.0
    totals = summarize([result])["hotpotqa/C3"]
    assert totals["top5_contains_person"] == 1
    assert totals["navigation_old_units"] == 1


def test_unidentified_name_is_not_counted_as_retrieval_failure(built_substrate):
    auditor = CoverageAuditor(Substrate.open(built_substrate))
    names, status = auditor.detect_people("Which century was this?")
    assert names == []
    assert status in {"unavailable", "no_person_detected"}


def test_coverage_manifest_rejects_wrong_substrate_hash(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/audit_retrieval_coverage.py"
    spec = importlib.util.spec_from_file_location("audit_retrieval_coverage_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    substrate_manifest = tmp_path / "manifest.json"
    substrate_manifest.write_text('{}', encoding="utf-8")
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(json.dumps({"dataset": "hotpotqa",
        "substrate_manifest_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="substrate manifest mismatch"):
        module.validate_manifest("hotpotqa", run_manifest, substrate_manifest)
    run_manifest.write_text(json.dumps({"dataset": "hotpotqa",
        "substrate_manifest_sha256": hashlib.sha256(substrate_manifest.read_bytes()).hexdigest()}),
        encoding="utf-8")
    module.validate_manifest("hotpotqa", run_manifest, substrate_manifest)
