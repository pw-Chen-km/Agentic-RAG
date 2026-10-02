"""Resolve displayed source text before exposing offline entity references."""
from __future__ import annotations

import copy
import time
from collections import defaultdict
from typing import Any

from agentic_rag.substrate.storage import Substrate


class ObservationProjector:
    def __init__(self, substrate: Substrate, *, expose_entities: bool = False) -> None:
        self.substrate = substrate
        self.expose_entities = expose_entities
        self.mentions_by_sentence = defaultdict(list)
        for mention in substrate.mentions:
            self.mentions_by_sentence[mention.sentence_id].append(mention)

    def project(self, results: list[dict[str, Any]], *, action_type: str):
        started = time.perf_counter()
        projected = copy.deepcopy(results)
        spans, mention_audit = [], []
        entities, sentences, chunks, passages, eligible = set(), set(), set(), set(), set()

        def visit(value):
            if isinstance(value, list):
                for child in value:
                    visit(child)
                return
            if not isinstance(value, dict):
                return
            sid = value.get("sentence_id")
            cid = value.get("parent_chunk_id", value.get("chunk_id"))
            text = value.get("text")
            if sid in self.substrate.sentence_by_id and isinstance(text, str):
                sentence = self.substrate.sentence_by_id[sid]
                cid = sentence.chunk_id
                chunks.add(cid)  # provenance only; never grants passage text
                start = sentence.text.find(text) if text else -1
                complete = text == sentence.text
                spans.append({"span_type": "sentence", "sentence_id": sid,
                              "chunk_id": cid, "start": start, "end": start + len(text),
                              "text": text, "visible": start >= 0, "complete": complete})
                if complete:
                    sentences.add(sid)
                    if not value.get("navigation_only", False) and value.get("evidence_eligible", True):
                        eligible.add(sid)
                annotations = []
                for mention in self.mentions_by_sentence[sid]:
                    visible = (start >= 0 and start <= mention.mention_start
                               and mention.mention_end <= start + len(text))
                    mention_audit.append({"span_type": "mention", "sentence_id": sid,
                                          "entity_id": mention.entity_id,
                                          "start": mention.mention_start, "end": mention.mention_end,
                                          "surface_form": mention.surface_form, "visible": visible})
                    # Partial text is not included in v2 memory, so it cannot
                    # introduce a usable entity even if one mention is whole.
                    if visible and complete and self.expose_entities:
                        entities.add(mention.entity_id)
                        annotations.append({"entity_id": mention.entity_id,
                                            "surface_form": mention.surface_form})
                if annotations:
                    value["visible_entity_mentions"] = annotations
            elif cid in self.substrate.chunk_by_id and isinstance(text, str):
                if text == self.substrate.chunk_by_id[cid].text:
                    chunks.add(cid)
                    passages.add(cid)
            for key, child in list(value.items()):
                if key != "visible_entity_mentions":
                    visit(child)

        visit(projected)
        delta = {"visible_entity_ids": sorted(entities), "visible_sentence_ids": sorted(sentences),
                 "visible_chunk_ids": sorted(chunks), "visible_passage_ids": sorted(passages),
                 "eligible_sentence_ids": sorted(eligible), "read_chunk_ids": []}
        return projected, delta, {"projected_source_spans": spans,
                                  "visible_source_spans": spans + mention_audit,
                                  "entity_mention_audit": mention_audit,
                                  "annotation_lookup_ms": (time.perf_counter() - started) * 1000}
