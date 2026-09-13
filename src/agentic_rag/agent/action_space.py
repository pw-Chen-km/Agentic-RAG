"""Derive one turn's structural action affordances from frozen state."""

from __future__ import annotations

from collections.abc import Sequence

from agentic_rag.agent.models import (
    ActionSpaceMode,
    AvailableActionSpace,
    ContextReferenceMap,
    EpisodeState,
    ExpandActionOption,
    ExpansionDirection,
    ExpansionKind,
    SearchActionOption,
    SearchMethod,
    SearchTarget,
)
from agentic_rag.agent.interface import InterfaceContract
from agentic_rag.agent.references import expected_expansion_source


_SEARCH_OPTION_ORDER = (
    (SearchMethod.LEXICAL, SearchTarget.ENTITY),
    (SearchMethod.BM25, SearchTarget.SENTENCE),
    (SearchMethod.BM25, SearchTarget.CHUNK),
    (SearchMethod.DENSE, SearchTarget.ENTITY),
    (SearchMethod.DENSE, SearchTarget.SENTENCE),
    (SearchMethod.DENSE, SearchTarget.CHUNK),
)


class AvailableActionSpaceBuilder:
    """Compute legal structural variants without choosing a strategy."""

    def __init__(
        self,
        enabled_expansions: Sequence[ExpansionKind],
        interface_contract: InterfaceContract | None = None,
    ) -> None:
        self.interface_contract = interface_contract
        self.enabled_expansions = tuple(
            interface_contract.enabled_expansions
            if interface_contract is not None
            else enabled_expansions
        )

    def build(
        self,
        state: EpisodeState,
        references: ContextReferenceMap,
        *,
        mode: ActionSpaceMode = ActionSpaceMode.NORMAL,
    ) -> AvailableActionSpace:
        evidence_refs = _sorted_refs(
            ref
            for ref, item in references.typed_refs.items()
            if item.can_use_as_evidence
        )
        if mode is ActionSpaceMode.BUDGET_FINALIZE:
            return AvailableActionSpace(
                mode=mode,
                finish_evidence_refs=tuple(evidence_refs),
            )

        retrieval_open = (
            state.remaining_step_budget > 0
            and state.remaining_policy_attempt_budget > 0
            and state.remaining_retrieved_token_budget > 0
        )
        if not retrieval_open:
            return AvailableActionSpace(
                mode=mode,
                finish_evidence_refs=tuple(evidence_refs),
            )

        pairs = (
            self.interface_contract.legal_search_pairs
            if self.interface_contract is not None
            else frozenset(_SEARCH_OPTION_ORDER)
        )
        search_options = tuple(
            SearchActionOption(method=method, target=target)
            for method, target in _SEARCH_OPTION_ORDER
            if (method, target) in pairs
        )
        expand_options: list[ExpandActionOption] = []
        for kind in self.enabled_expansions:
            expected_type = expected_expansion_source(kind)
            source_refs = _sorted_refs(
                ref
                for ref, item in references.typed_refs.items()
                if item.node_type == expected_type
            )
            if not source_refs:
                continue
            directions = (
                tuple(ExpansionDirection)
                if kind is ExpansionKind.CHUNK_ADJACENT_CHUNK
                else ()
            )
            expand_options.append(
                ExpandActionOption(
                    kind=kind,
                    source_refs=tuple(source_refs),
                    directions=directions,
                )
            )

        read_refs = _sorted_refs(
            ref for ref, item in references.typed_refs.items() if item.can_read
        )
        return AvailableActionSpace(
            mode=mode,
            search_options=search_options,
            expand_options=tuple(expand_options),
            read_refs=tuple(read_refs),
            finish_evidence_refs=tuple(evidence_refs),
        )


def _sorted_refs(values) -> list[str]:
    return sorted(
        values,
        key=lambda ref: ({"E": 0, "S": 1, "C": 2}[ref[0]], int(ref[1:])),
    )
