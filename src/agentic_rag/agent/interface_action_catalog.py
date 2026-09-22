"""Plain-language action guide generated from the current study action space."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.agent.models import (
    AvailableActionSpace,
    ExpansionKind,
    SearchMethod,
    SearchTarget,
)


COMMON_SEARCH_MECHANISM = """HOW SEARCH WORKS

The system turns your query into an embedding, a vector representing its meaning.
It compares that vector with stored vectors for the text units searched by the selected action.
Results need not contain the exact words in your query."""


@dataclass(frozen=True, slots=True)
class ActionCard:
    name: str
    signature: str
    description: str
    schema_description: str

    def render(self) -> str:
        return f"{self.signature}\n\n{self.description}"


ACTION_CARDS = {
    "find_passages": ActionCard(
        name="find_passages",
        signature="find_passages(query)",
        description=(
            "Searches all passages in the collection for text related to your query.\n"
            "Returns complete passages, including the sentences around a possible answer.\n"
            "A passage usually adds more text to your context than a sentence."
        ),
        schema_description="Search all passages by meaning and return complete passages.",
    ),
    "find_sentences": ActionCard(
        name="find_sentences",
        signature="find_sentences(query)",
        description=(
            "Searches all sentences in the collection for text related to your query.\n"
            "Returns individual sentences without their surrounding passage.\n"
            "Each result usually adds less text, but surrounding context may be absent."
        ),
        schema_description="Search all sentences by meaning and return complete sentences.",
    ),
    "follow_entity_to_passages": ActionCard(
        name="follow_entity_to_passages",
        signature="follow_entity_to_passages(entity_ref, query)",
        description=(
            "Starts from a visible E# name and finds passages that mention the same entity.\n"
            "Returns complete passages with their surrounding context.\n"
            "The query orders these linked passages by meaning; it cannot bring in a passage\n"
            "that is not linked to the selected entity. If query is null, the original question\n"
            "is used for ordering."
        ),
        schema_description="Follow a visible entity to passages mentioning it, then rank those passages by query meaning.",
    ),
    "follow_entity_to_sentences": ActionCard(
        name="follow_entity_to_sentences",
        signature="follow_entity_to_sentences(entity_ref, query)",
        description=(
            "Starts from a visible E# name and finds sentences that mention the same entity.\n"
            "Returns individual sentences without their surrounding passage.\n"
            "The query orders these linked sentences by meaning; it cannot bring in a sentence\n"
            "that is not linked to the selected entity. If query is null, the original question\n"
            "is used for ordering."
        ),
        schema_description="Follow a visible entity to sentences mentioning it, then rank those sentences by query meaning.",
    ),
    "finish": ActionCard(
        name="finish",
        signature="finish(answer, evidence_refs)",
        description=(
            "Ends this episode without retrieving new text. Give your answer and cite visible\n"
            "passage or sentence labels that support it. If no source is visible, evidence_refs\n"
            "must be an empty list."
        ),
        schema_description="End the episode with an answer and any visible source citations.",
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
        "AVAILABLE ACTIONS THIS TURN\n\n"
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
