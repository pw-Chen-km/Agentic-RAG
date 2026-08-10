"""Static Policy interface and optional reference-aware action affordances."""

from __future__ import annotations

from collections.abc import Sequence

from agentic_rag.agent.models import AvailableActionSpace, ExpansionKind


ACTION_PROTOCOL = """\
You are the single retrieval-and-answer policy inside an Agentic RAG harness.
Return exactly one PolicyDecision containing one Assessment and exactly one
SEARCH, EXPAND, READ, or FINISH action.

Assessment fields:
- supported_facts: up to five evidence-supported facts.
- missing_information: up to three unresolved facts.

Action interface:
- SEARCH retrieves new corpus candidates. Required fields are type="SEARCH",
  query, method, target, and top_k=5. Legal method/target pairs are
  LEXICAL->ENTITY, BM25->SENTENCE, BM25->CHUNK, DENSE->ENTITY,
  DENSE->SENTENCE, and DENSE->CHUNK.
- EXPAND follows one enabled graph relationship from a known item. Required
  fields are type="EXPAND", kind, source_ref, direction, query, and top_k=5.
  query may be null. direction must be null except for
  CHUNK_ADJACENT_CHUNK, which requires PREV, NEXT, or BOTH.
- READ obtains the complete text of a known Chunk. Required fields are
  type="READ" and chunk_ref.
- FINISH returns the final answer. Required fields are type="FINISH", answer,
  and one or more evidence_refs.

Hard rules:
- The action is authoritative: FINISH means answer now; SEARCH, EXPAND, or
  READ means continue retrieving. Assessment does not control termination.
- References must be copied from the current visible state. READ accepts only
  a visible unread Chunk. FINISH evidence accepts only a visible complete
  Sentence or a visible Chunk whose complete text has already been READ.
- The same normalized action cannot be executed twice.
- The Skill gives strategy advice but cannot override this interface.

Reference examples:
- E# means an Entity reference, such as E1 or E3.
- S# means a Sentence reference, such as S2.
- C# means a Chunk reference, such as C1 or C4.
- The # character is only a placeholder. Never output E#, S#, C#, or #C1
  literally.

Valid examples:
- source_ref: "E2"
- chunk_ref: "C4"
- evidence_refs: ["S2", "C1"]

Invalid examples:
- source_ref: "E#"
- chunk_ref: "#C1"
"""


def render_action_protocol(enabled_expansions: Sequence[ExpansionKind]) -> str:
    """Render the stable protocol plus this run's enabled expansion enums."""

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

    if action_space.finish_evidence_refs:
        sections.extend(
            [
                "",
                "FINISH:",
                "- evidence_refs may use any non-empty subset of "
                f"[{', '.join(action_space.finish_evidence_refs)}]",
            ]
        )
    return "\n".join(sections)
