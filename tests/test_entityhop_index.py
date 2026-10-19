from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from arag.tools.entity_link_index import VERSION, build_entity_index, write_entity_index


class FakeNLP:
    def __init__(self, entities_by_text):
        self.entities_by_text = entities_by_text

    def __call__(self, text):
        entities = [SimpleNamespace(text=surface, label_=label) for surface, label in self.entities_by_text.get(text, [])]
        return SimpleNamespace(ents=entities)


def test_corpus_only_entity_linking_and_explicit_alias(tmp_path: Path):
    rows = [
        {"id": "1", "text": "International Business Machines (IBM) was founded in 1911."},
        {"id": "2", "text": "IBM later opened a research lab."},
    ]
    nlp = FakeNLP(
        {
            rows[0]["text"]: [("International Business Machines", "ORG"), ("IBM", "ORG")],
            rows[1]["text"]: [("IBM", "ORG")],
        }
    )
    source = tmp_path / "chunks.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    result = build_entity_index(rows, nlp=nlp, enable_abbreviations=False, source_path=source)
    assert result["version"] == VERSION
    assert result["source"]["questions_used"] is False
    assert result["source"]["answers_used"] is False
    assert result["source"]["gold_evidence_used"] is False
    assert len(result["entities"]) == 1
    entity = result["entities"][0]
    assert entity["canonical"] == "International Business Machines"
    assert set(entity["aliases"]) == {"International Business Machines", "IBM"}
    assert entity["chunk_ids"] == ["1", "2"]
    assert entity["entity_id"] == "e000001"


def test_regex_does_not_create_entity_without_ner_mention():
    text = "Acronym Only Group (AOG) has no named-entity annotation."
    result = build_entity_index(
        [{"id": "1", "text": text}],
        nlp=FakeNLP({text: []}),
        enable_abbreviations=False,
    )
    assert result["entities"] == []


def test_write_is_json_and_has_stable_version(tmp_path: Path):
    text = "Acme Corporation (AC)"
    result = build_entity_index(
        [{"id": "1", "text": text}],
        nlp=FakeNLP({text: [("Acme Corporation", "ORG")]}),
        enable_abbreviations=False,
    )
    target = tmp_path / "entity_linked.json"
    write_entity_index(result, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["version"] == VERSION
    assert payload["entities"][0]["entity_id"] == "e000001"

