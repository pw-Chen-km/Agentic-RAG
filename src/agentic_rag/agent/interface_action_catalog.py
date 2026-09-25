"""Neutral capability cards generated from the current study action space.

The same cards are used to describe an operation in the prompt and in the
constrained action schema.  A condition changes which cards are present, not
the wording or the suggested retrieval policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.agent.models import (
    AvailableActionSpace,
    ExpansionKind,
    SearchMethod,
    SearchTarget,
)


COMMON_SEARCH_MECHANISM = """HOW THE SEARCH OPERATIONS WORK

For a search operation, the system represents your query by its meaning and
compares it with stored representations of the text units that operation can
search.  The returned text therefore can be related to the query even when it
does not repeat the query's exact words.  The operation description below
states which text units are in its search scope and which text it returns."""


@dataclass(frozen=True, slots=True)
class ActionCard:
    name: str
    signature: str
    description: str
    schema_description: str

    def render(self) -> str:
        return f"Operation: {self.signature}\n{self.description}"


ACTION_CARDS = {
    "find_passages": ActionCard(
        name="find_passages",
        signature="find_passages(query)",
        description=(
            "Search scope: all passages in the collection.\n"
            "This scope may fit when the information you still need may appear anywhere in the collection, "
            "or is not clearly connected to a name already shown to you.\n"
            "How it works: the meaning of query is compared with stored representations of passages.\n"
            "Returns: complete passages. A passage includes surrounding context across multiple sentences, "
            "so it usually provides more background but uses more context tokens.\n"
            "Limitation: it is not restricted to a visible entity or a previously shown source."
        ),
        schema_description=(
            "Search all passages by query meaning and return complete passages. "
            "This scope may fit when the missing information may be anywhere in the collection "
            "or is not tied to a shown entity."
        ),
    ),
    "find_sentences": ActionCard(
        name="find_sentences",
        signature="find_sentences(query)",
        description=(
            "Search scope: all sentences in the collection.\n"
            "This scope may fit when the information you still need may appear anywhere in the collection, "
            "or when a focused statement is more useful than surrounding passage context.\n"
            "How it works: the meaning of query is compared with stored representations of sentences.\n"
            "Returns: complete sentences. A sentence gives a focused statement and usually uses fewer context "
            "tokens, but it may omit surrounding information from the full passage.\n"
            "Limitation: it is not restricted to a visible entity or a previously shown source."
        ),
        schema_description=(
            "Search all sentences by query meaning and return complete sentences. "
            "This scope may fit when the missing information may be anywhere in the collection "
            "or a focused statement is preferable to surrounding passage context."
        ),
    ),
    "follow_entity_to_passages": ActionCard(
        name="follow_entity_to_passages",
        signature="follow_entity_to_passages(entity_ref)",
        description=(
            "Search scope: passages linked to the selected visible E# name.\n"
            "This scope may fit when a previous search found a visible entity related to the question "
            "but did not provide enough information to answer it.\n"
            "How it works: the system follows stored links from that entity to passages that mention it,\n"
            "then ranks those linked passages using the original question.\n"
            "Returns: complete passages from that linked candidate set. A passage includes surrounding context "
            "across multiple sentences, so it usually provides more background but uses more context tokens.\n"
            "Limitation: an unlinked passage cannot be returned. This operation does not search outside\n"
            "the entity-linked candidate set."
        ),
        schema_description=(
            "Follow a visible entity to linked passages and rank them using the original question. "
            "This scope may fit when a shown entity is related to the missing information and broader context may help."
        ),
    ),
    "follow_entity_to_sentences": ActionCard(
        name="follow_entity_to_sentences",
        signature="follow_entity_to_sentences(entity_ref)",
        description=(
            "Search scope: sentences linked to the selected visible E# name.\n"
            "This scope may fit when a previous search found a visible entity related to the question "
            "but did not provide enough information to answer it.\n"
            "How it works: the system follows stored links from that entity to sentences that mention it,\n"
            "then ranks those linked sentences using the original question.\n"
            "Returns: complete sentences from that linked candidate set. A sentence gives a focused statement "
            "and usually uses fewer context tokens, but it may omit surrounding information from the full passage.\n"
            "Limitation: an unlinked sentence cannot be returned. This operation does not search outside\n"
            "the entity-linked candidate set."
        ),
        schema_description=(
            "Follow a visible entity to linked sentences and rank them using the original question. "
            "This scope may fit when a shown entity is related to the missing information and a focused statement may help."
        ),
    ),
    "finish": ActionCard(
        name="finish",
        signature="finish(answer, evidence_refs)",
        description=(
            "Search scope: none.\n"
            "How it works: ends the episode without retrieving new text.\n"
            "Returns: no new source text.\n"
            "Limitation: cite only visible passage or sentence labels that support the answer;\n"
            "if no source is visible, evidence_refs must be an empty list."
        ),
        schema_description="End the episode with an answer and visible source citations.",
    ),
}

ACTION_ORDER = tuple(ACTION_CARDS)


def available_action_names(space: AvailableActionSpace) -> tuple[str, ...]:
    """Return study action names in one fixed order shared by prompt and schema."""

    names: set[str] = set()
    for option in space.search_options:
        if option.method is not SearchMethod.DENSE:
            raise ValueError("the study action guide only supports dense search")
        if option.target is SearchTarget.CHUNK:
            names.add("find_passages")
        elif option.target is SearchTarget.SENTENCE:
            names.add("find_sentences")
        else:
            raise ValueError("the study action guide does not expose entity search")
    for option in space.expand_options:
        if option.kind is ExpansionKind.ENTITY_MENTIONED_IN_CHUNK:
            names.add("follow_entity_to_passages")
        elif option.kind is ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE:
            names.add("follow_entity_to_sentences")
        else:
            raise ValueError("the study action guide does not expose this navigation path")
    if space.read_refs:
        raise ValueError("the study action guide does not expose a read action")
    if space.finish_available:
        names.add("finish")
    return tuple(name for name in ACTION_ORDER if name in names)


def render_action_guide(
    space: AvailableActionSpace,
    *,
    entity_annotation: bool,
    entity_navigation_possible: bool,
    require_evidence_assessment: bool,
) -> str:
    names = available_action_names(space)
    if not names:
        raise ValueError("the study action guide needs at least one action")
    sections = [
        "OPERATIONS AVAILABLE NOW\n\n"
        "This is the complete list of operations available in the current state. "
        "The order of this list has no meaning. Choose a scope that matches the information you still need; "
        "these descriptions explain when a scope may fit but do not require an order or prefer an operation.\n\n"
        + "\n\n".join(ACTION_CARDS[name].render() for name in names),
    ]
    reference_rules = [
        "C# labels identify passages shown to you; S# labels identify sentences shown to you.",
        "Only visible C# and S# labels can be used as evidence_refs.",
    ]
    if entity_annotation:
        reference_rules.append(
            "E# labels identify names found in shown source text. E# is not an evidence citation."
        )
        if any(name.startswith("follow_entity_") for name in names):
            reference_rules.append(
                "A follow action can use only an E# label displayed with its name in the current observation."
            )
        elif entity_navigation_possible:
            reference_rules.append("A follow action becomes available only after an E# label is shown.")
        else:
            reference_rules.append("E# labels are annotations only; no action can follow them.")
    sections.append("REFERENCE RULES\n\n" + "\n".join(reference_rules))
    if require_evidence_assessment:
        decision = (
            "Return one structured decision with supported_facts, missing_information, and action.\n"
            "supported_facts: brief facts supported by source text already shown.\n"
            "missing_information: information still needed to answer the question.\n"
            "action: exactly one action listed above, using its exact name and argument names."
        )
    else:
        decision = (
            "Return one structured decision with exactly one action listed above,\n"
            "using its exact name and argument names."
        )
    sections.append("DECISION FORMAT\n\n" + decision)
    return "\n\n".join(sections)
