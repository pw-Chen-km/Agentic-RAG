"""Relation-free Entity–Sentence bridging over persisted sparse incidence."""

from __future__ import annotations

from pathlib import Path

from scipy import sparse

from agentic_rag.errors import NodeNotFoundError
from agentic_rag.substrate.models import BridgeHit
from agentic_rag.substrate.storage import Substrate


class SubstrateBridge:
    def __init__(self, substrate: Substrate | str | Path) -> None:
        self.substrate = (
            substrate if isinstance(substrate, Substrate) else Substrate.open(substrate)
        )
        self.entity_to_sentence = sparse.load_npz(
            self.substrate.root / "relations" / "entity_to_sentence.npz"
        ).tocsr()
        self.sentence_to_entity = sparse.load_npz(
            self.substrate.root / "relations" / "sentence_to_entity.npz"
        ).tocsr()
        self.entity_row = {
            item.entity_id: index
            for index, item in enumerate(self.substrate.entities)
        }
        self.sentence_row = {
            item.sentence_id: index
            for index, item in enumerate(self.substrate.sentences)
        }

    def sentences_for_entity(
        self, entity_id: str, scope_id: str | None = None
    ) -> list[str]:
        row = self.entity_row.get(entity_id)
        if row is None:
            raise NodeNotFoundError(f"Unknown Entity ID: {entity_id}")
        sentence_ids = [
            self.substrate.sentences[index].sentence_id
            for index in self.entity_to_sentence.getrow(row).indices
        ]
        if scope_id is not None:
            self.substrate.require_scope(scope_id)
            allowed = self.substrate.sentence_ids_by_scope[scope_id]
            sentence_ids = [item for item in sentence_ids if item in allowed]
        return sorted(sentence_ids)

    def entities_for_sentence(self, sentence_id: str) -> list[str]:
        row = self.sentence_row.get(sentence_id)
        if row is None:
            raise NodeNotFoundError(f"Unknown Sentence ID: {sentence_id}")
        return sorted(
            self.substrate.entities[index].entity_id
            for index in self.sentence_to_entity.getrow(row).indices
        )

    def entity_sentence_entity(
        self,
        entity_id: str,
        scope_id: str,
        *,
        include_source: bool = False,
        top_k: int | None = None,
    ) -> list[BridgeHit]:
        self.substrate.require_scope(scope_id)
        if entity_id not in self.substrate.entity_ids_by_scope[scope_id]:
            if entity_id not in self.entity_row:
                raise NodeNotFoundError(f"Unknown Entity ID: {entity_id}")
            raise NodeNotFoundError(
                f"Entity {entity_id} is not present in scope {scope_id}"
            )
        hits: list[BridgeHit] = []
        for sentence_id in self.sentences_for_entity(entity_id, scope_id):
            sentence = self.substrate.sentence_by_id[sentence_id]
            chunk = self.substrate.chunk_by_id[sentence.chunk_id]
            document = self.substrate.document_by_id[chunk.doc_id]
            for target_entity_id in self.entities_for_sentence(sentence_id):
                if target_entity_id == entity_id and not include_source:
                    continue
                target = self.substrate.entity_by_id[target_entity_id]
                hits.append(
                    BridgeHit(
                        source_entity_id=entity_id,
                        bridge_sentence_id=sentence_id,
                        target_entity_id=target_entity_id,
                        bridge_text=sentence.text,
                        target_canonical_name=target.canonical_name,
                        chunk_id=chunk.chunk_id,
                        doc_id=document.doc_id,
                        title=document.title,
                    )
                )
        hits.sort(
            key=lambda item: (
                item.bridge_sentence_id,
                item.target_entity_id,
            )
        )
        return hits[:top_k] if top_k is not None else hits
