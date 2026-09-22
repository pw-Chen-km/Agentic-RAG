"""Resolve eligible evidence references to text and provenance."""

from __future__ import annotations

from collections.abc import Iterable

from agentic_rag.agent.models import (
    ChunkRef,
    EpisodeState,
    EvidenceRef,
    ResolvedEvidence,
    SentenceRef,
)
from agentic_rag.errors import EvidenceEligibilityError, NodeNotFoundError
from agentic_rag.substrate.storage import Substrate


class EvidenceResolver:
    def __init__(self, substrate: Substrate) -> None:
        self.substrate = substrate

    def resolve(
        self,
        refs: Iterable[EvidenceRef],
        state: EpisodeState,
        scope_id: str,
    ) -> list[ResolvedEvidence]:
        self.substrate.require_scope(scope_id)
        unique: list[EvidenceRef] = []
        seen: set[tuple[str, str]] = set()
        for ref in refs:
            key = (ref.unit, ref.id)
            if key not in seen:
                seen.add(key)
                unique.append(ref)

        selected_chunk_ids = {
            ref.id for ref in unique if isinstance(ref, ChunkRef)
        }
        resolved: list[ResolvedEvidence] = []
        for ref in unique:
            if isinstance(ref, SentenceRef):
                if ref.id not in self.substrate.sentence_ids_by_scope[scope_id]:
                    raise NodeNotFoundError(
                        f"Sentence {ref.id} is not present in scope {scope_id}"
                    )
                if ref.id not in state.eligible_sentence_ids:
                    raise EvidenceEligibilityError(
                        f"Sentence has not been shown as complete evidence: {ref.id}"
                    )
                sentence = self.substrate.sentence_by_id[ref.id]
                if sentence.chunk_id in selected_chunk_ids:
                    continue
                chunk = self.substrate.chunk_by_id[sentence.chunk_id]
                document = self.substrate.document_by_id[chunk.doc_id]
                resolved.append(
                    ResolvedEvidence(
                        ref=ref,
                        text=sentence.text,
                        document_id=document.doc_id,
                        title=document.title,
                        parent_chunk_id=chunk.chunk_id,
                    )
                )
                continue

            if ref.id not in self.substrate.chunk_ids_by_scope[scope_id]:
                raise NodeNotFoundError(
                    f"Chunk {ref.id} is not present in scope {scope_id}"
                )
            if ref.id not in state.visible_chunk_ids:
                raise EvidenceEligibilityError(
                    f"Chunk must be shown before evidence resolution: {ref.id}"
                )
            chunk = self.substrate.chunk_by_id[ref.id]
            document = self.substrate.document_by_id[chunk.doc_id]
            resolved.append(
                ResolvedEvidence(
                    ref=ref,
                    text=chunk.text,
                    document_id=document.doc_id,
                    title=document.title,
                    parent_chunk_id=chunk.chunk_id,
                    contained_sentence_ids=[
                        item.sentence_id
                        for item in self.substrate.sentences_by_chunk.get(
                            chunk.chunk_id, []
                        )
                    ],
                )
            )
        return resolved
