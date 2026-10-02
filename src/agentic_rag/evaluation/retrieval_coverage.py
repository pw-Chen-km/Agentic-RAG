"""Offline, gold-free diagnostics for retrieval and visible entity links."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from agentic_rag.substrate.storage import Substrate


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _contains(text: str, name: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(_normalized(name)) + r"(?!\w)",
                     _normalized(text)) is not None


def _seen_sentences(spans: list[dict[str, Any]], substrate: Substrate) -> set[str]:
    seen = set()
    for span in spans:
        sid = span.get("sentence_id")
        sentence = substrate.sentence_by_id.get(sid)
        if (sentence is not None and span.get("visible") is True
                and span.get("complete") is True and span.get("start") == 0
                and span.get("end", -1) >= len(sentence.text)):
            seen.add(sid)
    return seen


class CoverageAuditor:
    def __init__(self, substrate: Substrate) -> None:
        self.substrate = substrate
        self.entity_sentences: dict[str, set[str]] = defaultdict(set)
        for mention in substrate.mentions:
            self.entity_sentences[mention.entity_id].add(mention.sentence_id)
        self.person_names: dict[str, str] = {}
        for entity in substrate.entities:
            if entity.entity_type in {"PERSON", "PER"}:
                self.person_names[_normalized(entity.canonical_name)] = entity.canonical_name
        for alias in substrate.entity_aliases:
            entity = substrate.entity_by_id.get(alias.entity_id)
            if entity is not None and entity.entity_type in {"PERSON", "PER"}:
                self.person_names.setdefault(_normalized(alias.alias), alias.alias)
        self.person_by_first_word: dict[str, set[str]] = defaultdict(set)
        for name in self.person_names:
            if len(name.split()) >= 2:
                self.person_by_first_word[name.split()[0]].add(name)
        self._corpus_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self.ner_model = substrate.manifest.ner_model.name
        self._nlp = None
        if self.ner_model.startswith("spacy:"):
            try:
                import spacy
                self._nlp = spacy.load(self.ner_model.removeprefix("spacy:"))
            except (ImportError, OSError):
                pass

    def detect_people(self, query: str) -> tuple[list[str], str]:
        names: dict[str, str] = {}
        normalized_query = _normalized(query)
        for word in re.findall(r"(?u)\b\w+\b", normalized_query):
            for normalized_name in self.person_by_first_word.get(word, ()):
                if _contains(query, normalized_name):
                    names[normalized_name] = self.person_names[normalized_name]
        if self._nlp is not None:
            for entity in self._nlp(query).ents:
                if entity.label_ in {"PERSON", "PER"}:
                    names.setdefault(_normalized(entity.text), entity.text)
        status = "identified" if names else "no_person_detected" if self._nlp is not None else "unavailable"
        return sorted(names.values(), key=str.casefold), status

    def corpus_presence(self, scope_id: str, name: str) -> dict[str, Any]:
        key = (scope_id, _normalized(name))
        if key not in self._corpus_cache:
            self.substrate.require_scope(scope_id)
            title_docs = []
            for doc_id in sorted(self.substrate.doc_ids_by_scope[scope_id]):
                title = self.substrate.document_by_id[doc_id].title or ""
                normalized_title = _normalized(title)
                if normalized_title == key[1] or normalized_title.startswith(key[1] + " ("):
                    title_docs.append(doc_id)
            text_present = any(
                _contains(self.substrate.sentence_by_id[sid].text, name)
                for sid in self.substrate.sentence_ids_by_scope[scope_id]
            )
            self._corpus_cache[key] = {"matching_document_ids": title_docs,
                                       "text_mention_present": text_present}
        return self._corpus_cache[key]

    def entity_candidates(self, scope_id: str, entity_id: str, seen: set[str]) -> dict[str, Any]:
        sentence_ids = self.entity_sentences.get(entity_id, set()) & self.substrate.sentence_ids_by_scope[scope_id]
        passage_ids = {self.substrate.sentence_by_id[sid].chunk_id for sid in sentence_ids}
        unseen_passages = {
            cid for cid in passage_ids
            if any(sentence.sentence_id not in seen
                   for sentence in self.substrate.sentences_by_chunk.get(cid, ()))
        }
        return {"sentence_candidates": len(sentence_ids),
                "unseen_sentence_candidates": len(sentence_ids - seen),
                "passage_candidates": len(passage_ids),
                "passages_with_unseen_text": len(unseen_passages)}

    def audit_episode(self, episode: dict[str, Any], dataset: str) -> dict[str, Any]:
        scope_id = episode["scope_id"]
        self.substrate.require_scope(scope_id)
        person_queries, visible_entities, navigations = [], [], []
        first_seen_entities: set[str] = set()
        for turn, step in enumerate(episode.get("trajectory", []), 1):
            seen = _seen_sentences(step.get("visible_source_spans", []), self.substrate)
            references = (step.get("context_reference_map") or {}).get("typed_refs", {})
            for ref, node in sorted(references.items()):
                if node.get("node_type") != "ENTITY" or ref in first_seen_entities:
                    continue
                entity_id = node["stable_id"]
                if entity_id not in self.substrate.entity_by_id:
                    raise ValueError(f"unknown visible entity: {entity_id}")
                first_seen_entities.add(ref)
                visible_entities.append({"turn": turn, "entity_ref": ref, "entity_id": entity_id,
                    "canonical_name": self.substrate.entity_by_id[entity_id].canonical_name,
                    **self.entity_candidates(scope_id, entity_id, seen)})
            raw_action = ((step.get("decision") or {}).get("action") or {})
            action = ((step.get("resolved_decision") or step.get("decision") or {}).get("action") or {})
            observation = step.get("observation") or {}
            action_type = action.get("type")
            if action_type not in {"SEARCH", "EXPAND"}:
                continue
            query = action.get("query")
            results = observation.get("results") or []
            completed = observation.get("status") == "ok"
            if isinstance(query, str):
                people, identification = self.detect_people(query)
                for name in people:
                    presence = self.corpus_presence(scope_id, name)
                    person_queries.append({"turn": turn, "action_type": action_type,
                        "query": query, "person": name, "identification": identification,
                        "status": "completed" if completed else "not_executed",
                        "top5_contains_person": (
                            any(_contains((item.get("title") or "") + " " + (item.get("text") or ""), name)
                                for item in results[:5]) if completed else None),
                        **presence})
                if not people:
                    person_queries.append({"turn": turn, "action_type": action_type,
                        "query": query, "identification": identification,
                        "status": "not_applicable" if identification == "no_person_detected" else "unavailable"})
            if action_type == "EXPAND" and completed:
                landing = "sentence" if action.get("kind") == "ENTITY_MENTIONED_IN_SENTENCE" else "passage"
                ids = {item.get("sentence_id" if landing == "sentence" else "chunk_id")
                       for item in results}
                ids.discard(None)
                if landing == "sentence":
                    old = sum(sid in seen for sid in ids)
                else:
                    old = sum(all(sentence.sentence_id in seen
                                  for sentence in self.substrate.sentences_by_chunk.get(cid, ()))
                              for cid in ids)
                navigations.append({"turn": turn, "entity_ref": raw_action.get("source_ref"),
                    "entity_id": action.get("source_id"), "landing": landing,
                    "returned_units": len(ids), "old_units": old,
                    "old_source_ratio": old / len(ids) if ids else None,
                    "status": "measured" if ids else "zero_returned"})
        return {"dataset": dataset, "condition": episode["episode_id"].split("--", 1)[0],
                "episode_id": episode["episode_id"], "scope_id": scope_id,
                "person_queries": person_queries, "visible_entities": visible_entities,
                "navigations": navigations}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        group = groups[row["dataset"] + "/" + row["condition"]]
        group["episodes"] += 1
        for query in row["person_queries"]:
            if query.get("status") == "completed":
                group["identified_person_queries"] += 1
                group["top5_contains_person"] += bool(query["top5_contains_person"])
            elif query.get("status") == "unavailable":
                group["person_detection_unavailable"] += 1
        for entity in row["visible_entities"]:
            group["visible_entity_refs"] += 1
            group["entity_sentence_candidates"] += entity["sentence_candidates"]
            group["unseen_entity_sentence_candidates"] += entity["unseen_sentence_candidates"]
        for navigation in row["navigations"]:
            group["navigation_actions"] += 1
            group["navigation_returned_units"] += navigation["returned_units"]
            group["navigation_old_units"] += navigation["old_units"]
    return {key: dict(value) for key, value in sorted(groups.items())}
