"""Static Policy interface and optional reference-aware action affordances."""

from __future__ import annotations

from collections.abc import Sequence

from agentic_rag.agent.models import AvailableActionSpace, ExpansionKind
from agentic_rag.agent.interface import InterfaceContract


ACTION_PROTOCOL = """\
You answer the question using information made available by the current
retrieval interface. At each step, choose one action that is listed as
available, or finish. The interface does not prescribe an action order and no
retrieval action is preferred by default.

Action interface: the current turn's native tool definitions are authoritative.
They state the arguments accepted by each available action. A search returns the complete
unit described by that tool; an entity action starts from an entity reference
shown with its name; finish returns an answer and any references that support
it. Use only references shown in the current observation. Never invent an
entity, source text, reference, or unavailable action. You may finish with no
citation when no eligible evidence is visible.

Reference examples: source_ref: "E2", chunk_ref: "C4", and
evidence_refs: ["S2", "C1"]. Copy the exact references shown in the current
observation.
Never output E#, S#, C#, or another placeholder instead of a reference.
"""


def render_action_protocol(
    enabled_expansions: Sequence[ExpansionKind] | InterfaceContract,
) -> str:
    """Render the stable protocol plus this run's enabled expansion enums."""

    if isinstance(enabled_expansions, InterfaceContract):
        return enabled_expansions.protocol

    enabled = [item.value for item in enabled_expansions]
    expansion_lines = (
        "\n".join(f"- {item}" for item in enabled)
        if enabled
        else "- No EXPAND kinds are enabled."
    )
    return f"{ACTION_PROTOCOL}\nEnabled EXPAND kinds for this run:\n{expansion_lines}"


def render_available_action_options(
    action_space: AvailableActionSpace,
) -> str:
    """List currently reference-valid action templates without choosing one."""

    sections = [
        "Currently available action options (structurally/reference-valid; "
        "query choices may still duplicate history):",
    ]
    if action_space.search_options:
        sections.extend(
            [
                "",
                "SEARCH:",
                *(
                    f"- {item.method.value} -> {item.target.value}"
                    for item in action_space.search_options
                ),
            ]
        )

    expansion_lines: list[str] = []
    for option in action_space.expand_options:
        source_pool = ", ".join(option.source_refs)
        if option.directions:
            suffix = "direction in [" + ", ".join(
                item.value for item in option.directions
            ) + "]"
        else:
            suffix = "direction=null"
        expansion_lines.append(
            f"- {option.kind.value}: source_ref in [{source_pool}], {suffix}"
        )
    if expansion_lines:
        sections.extend(["", "EXPAND:", *expansion_lines])

    if action_space.read_refs:
        sections.extend(
            ["", "READ:", f"- chunk_ref in [{', '.join(action_space.read_refs)}]"]
        )

    if action_space.finish_available:
        sections.extend(
            [
                "",
                "FINISH:",
                (
                    "- evidence_refs may use any non-empty subset of "
                    f"[{', '.join(action_space.finish_evidence_refs)}]"
                    if action_space.finish_evidence_refs
                    else "- no eligible evidence reference is currently visible"
                ),
            ]
        )
    return "\n".join(sections)
