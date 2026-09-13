"""Frozen interface variants used by the Agentic RAG interface study.

The study changes only what the policy can see and request.  Retrieval and
substrate code remain shared so that a condition cannot accidentally acquire
an extra backend capability through a second configuration file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from agentic_rag.agent.models import ExpansionKind, SearchMethod, SearchTarget


class EntityContinuation(StrEnum):
    NONE = "none"
    CHUNK = "chunk"
    SENTENCE = "sentence"
    ANNOTATION_ONLY = "annotation-only"


@dataclass(frozen=True, slots=True)
class InterfaceContract:
    """One compiled study condition and its complete capability view."""

    name: str
    global_sentence_search: bool
    entity_annotation: bool
    entity_continuation: EntityContinuation
    top_k: int = 5
    read_full_chunk: bool = True

    @classmethod
    def from_name(cls, name: str) -> "InterfaceContract":
        return get_interface_contract(name)

    @classmethod
    def from_config(cls, name: str) -> "InterfaceContract":
        return get_interface_contract(name)

    def __post_init__(self) -> None:
        if self.name not in {"C0", "C1", "C2", "C3", "C4", "A1"}:
            raise ValueError(f"unknown interface condition: {self.name}")
        if self.top_k != 5:
            raise ValueError("the interface study fixes top_k=5")
        if self.entity_continuation is EntityContinuation.CHUNK and not self.entity_annotation:
            raise ValueError("entity continuation requires entity annotations")
        if self.entity_continuation is EntityContinuation.SENTENCE and not self.entity_annotation:
            raise ValueError("entity continuation requires entity annotations")
        if self.name == "A1" and self.entity_continuation is not EntityContinuation.ANNOTATION_ONLY:
            raise ValueError("A1 is annotation-only")

    @property
    def legal_search_pairs(self) -> frozenset[tuple[SearchMethod, SearchTarget]]:
        pairs = {(SearchMethod.DENSE, SearchTarget.CHUNK)}
        if self.global_sentence_search:
            pairs.add((SearchMethod.DENSE, SearchTarget.SENTENCE))
        return frozenset(pairs)

    @property
    def enabled_expansions(self) -> tuple[ExpansionKind, ...]:
        if self.entity_continuation is EntityContinuation.CHUNK:
            return (ExpansionKind.ENTITY_MENTIONED_IN_CHUNK,)
        if self.entity_continuation is EntityContinuation.SENTENCE:
            return (ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE,)
        return ()

    @property
    def capability_view(self) -> dict[str, Any]:
        return {
            "global_sentence_search": self.global_sentence_search,
            "entity_annotation": self.entity_annotation,
            "entity_continuation": self.entity_continuation.value,
            "legal_search_pairs": sorted(
                f"{method.value}->{target.value}"
                for method, target in self.legal_search_pairs
            ),
            "enabled_expansions": [item.value for item in self.enabled_expansions],
            "top_k": self.top_k,
            "read_full_chunk": self.read_full_chunk,
        }

    @property
    def protocol(self) -> str:
        search = [f"{m.value} -> {t.value}" for m, t in sorted(self.legal_search_pairs, key=lambda p: (p[0].value, p[1].value))]
        expand = ", ".join(item.value for item in self.enabled_expansions) or "none"
        annotation = "visible offline entity mentions" if self.entity_annotation else "no entity annotations"
        return (
            "Agentic RAG interface study contract " + self.name + ".\n"
            "Return exactly one structured PolicyDecision with an Assessment and one action.\n"
            f"SEARCH pairs: {', '.join(search)}. top_k is fixed at 5.\n"
            f"Entity annotations: {annotation}. EXPAND kinds: {expand}.\n"
            "READ returns a complete chunk and may be used only once per chunk.\n"
            "FINISH evidence must copy visible S# references or a complete READ C# reference.\n"
            "Never invent references, entity IDs, hidden evidence, or unsupported actions."
        )

    def compile(self) -> dict[str, Any]:
        """Return the single serializable contract source used by all layers."""
        payload = {
            "name": self.name,
            "capabilities": self.capability_view,
            "protocol": self.protocol,
            "validator_allowlist": self.capability_view["legal_search_pairs"],
            "artifact_contract": "agentic-rag-interface-study-v1",
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return {**payload, "digest": hashlib.sha256(encoded).hexdigest()}

    def allows_search(self, method: SearchMethod | str, target: SearchTarget | str) -> bool:
        method = SearchMethod(getattr(method, "value", method))
        target = SearchTarget(getattr(target, "value", target))
        return (method, target) in self.legal_search_pairs

    def allows_expansion(self, kind: ExpansionKind | str) -> bool:
        return ExpansionKind(getattr(kind, "value", kind)) in self.enabled_expansions


INTERFACE_CONTRACTS: dict[str, InterfaceContract] = {
    "C0": InterfaceContract("C0", False, False, EntityContinuation.NONE),
    "C1": InterfaceContract("C1", True, False, EntityContinuation.NONE),
    "C2": InterfaceContract("C2", False, True, EntityContinuation.CHUNK),
    "C3": InterfaceContract("C3", False, True, EntityContinuation.SENTENCE),
    "C4": InterfaceContract("C4", True, True, EntityContinuation.SENTENCE),
    "A1": InterfaceContract("A1", True, True, EntityContinuation.ANNOTATION_ONLY),
}


def get_interface_contract(name: str) -> InterfaceContract:
    try:
        return INTERFACE_CONTRACTS[name.upper()]
    except KeyError as exc:
        raise ValueError(f"unknown interface condition: {name}") from exc


def compile_interface_contract(name: str) -> dict[str, Any]:
    return get_interface_contract(name).compile()
