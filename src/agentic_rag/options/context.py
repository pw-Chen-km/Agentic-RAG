"""Progressively disclosed contexts for the Options agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from agentic_rag.agent.context import BuiltPolicyContext, PolicyContextBuilder
from agentic_rag.agent.models import (
    EpisodeState,
    Message,
    PolicyView,
    StepRecord,
)
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.options.catalog import OptionCatalog
from agentic_rag.options.schema import policy_schema, schema_sha256, selector_schema
from agentic_rag.substrate.storage import Substrate


@dataclass(frozen=True, slots=True)
class BuiltOptionContext:
    messages: list[Message]
    base: BuiltPolicyContext
    decision_format: type[BaseModel]
    decision_schema_sha256: str


class OptionContextBuilder:
    """Builds compact chronological input without repeating old evidence text."""

    def __init__(self, substrate: Substrate, base: PolicyContextBuilder, catalog: OptionCatalog) -> None:
        self.substrate = substrate
        self.base = base
        self.catalog = catalog

    def selector(
        self,
        query: str,
        skill: SkillDocument | str,
        state: EpisodeState,
        trajectory: Sequence[StepRecord],
        *,
        scope_id: str,
        last_option_id: str | None,
        last_option_status: str | None = None,
    ) -> tuple[BuiltPolicyContext, list[Message], type[BaseModel], str, tuple[str, ...]]:
        # ``PolicyContextBuilder`` also renders the complete legacy Skill into
        # its messages.  Selector turns intentionally use progressive
        # disclosure: they receive only option summaries, so do not pass the
        # full catalog skill through that builder.  The returned base object is
        # still used for the frozen action space and compact state projection.
        base = self.base.build(
            query,
            "Option selector: choose a sub-goal from the listed summaries.",
            state,
            trajectory,
            scope_id=scope_id,
        )
        ids = self.available_option_ids(state, base, last_option_id=last_option_id)
        fmt = selector_schema(ids)
        context = self._context_payload(
            query, state, trajectory, base,
            last_option_id=last_option_id,
            last_option_status=last_option_status,
        )
        option_summaries = self.catalog.summaries(tuple(item for item in ids if item != "FALLBACK"))
        option_summaries.append({
            "option_id": "FALLBACK",
            "name": "Primitive fallback",
            "goal": "Choose one legal retrieval primitive when no option clearly fits, then return to the selector",
            "initiation": ["No option has a clear initiation match"],
            "primitive_actions": ["SEARCH", "EXPAND", "READ"],
        })
        messages = [
            Message(
                role="system",
                content=(
                    "You are the Option Selector. Choose exactly one option_id; "
                    "do not output a primitive action. An option is a temporary "
                    "sub-goal. After COMPLETE or BLOCKED control returns here. "
                    "Use only the listed option IDs.\n\nAvailable options:\n"
                    + json.dumps(option_summaries, ensure_ascii=False, separators=(",", ":"))
                ),
            ),
            Message(role="system", content="Fixed runtime guidance:\n" + self.catalog.fixed_runtime_guidance),
            Message(role="system", content="Current option catalog hash: " + self.catalog.sha256()),
            Message(role="user", content=f"Question:\n{query}"),
            Message(role="user", content=json.dumps(context, ensure_ascii=False, separators=(",", ":"))),
        ]
        return base, messages, fmt, schema_sha256(fmt), ids

    def policy(
        self,
        query: str,
        skill: SkillDocument | str,
        state: EpisodeState,
        trajectory: Sequence[StepRecord],
        *,
        scope_id: str,
        option_id: str,
        last_option_id: str | None,
        last_option_status: str | None = None,
    ) -> BuiltOptionContext:
        # As above, suppress the base builder's full-skill system message.  A
        # policy turn gets exactly one complete selected option below; sending
        # the whole catalog as well would defeat progressive disclosure and
        # needlessly inflate every request.
        base = self.base.build(
            query,
            "Selected option policy is supplied below.",
            state,
            trajectory,
            scope_id=scope_id,
        )
        option = self.catalog.get(option_id)
        fmt = policy_schema(
            base.available_action_space,
            option_id=option_id,
            allowed_actions=option.primitive_actions,
        )
        context = self._context_payload(
            query, state, trajectory, base,
            last_option_id=last_option_id,
            last_option_status=last_option_status,
        )
        # The selected option is disclosed once in the system message below;
        # do not copy its full procedure into the JSON state payload.
        protocol = (
            "You are the policy inside the selected option. Re-read the current "
            "evidence and missing information after every observation. Output one "
            "global assessment, option_status, and (only for CONTINUE) one legal "
            "primitive action. COMPLETE or BLOCKED must not include an action. "
            "Only O5_ANSWER may output FINISH; it must cite legal evidence."
        )
        messages = [
            Message(role="system", content=protocol),
            Message(role="system", content="Selected option:\n\n" + option.markdown()),
            Message(role="system", content="Current option catalog hash: " + self.catalog.sha256()),
            Message(role="user", content=f"Question:\n{query}"),
            Message(role="user", content=json.dumps(context, ensure_ascii=False, separators=(",", ":"))),
        ]
        if option_id == "O5_ANSWER":
            # The answer contract is relevant only in the answer option; it
            # should not distract retrieval options or make them act as if
            # they were already finishing the episode.
            messages.insert(
                2,
                Message(
                    role="system",
                    content="Fixed answer contract:\n" + self.catalog.fixed_answer_contract,
                ),
            )
        return BuiltOptionContext(
            messages=messages,
            base=base,
            decision_format=fmt,
            decision_schema_sha256=schema_sha256(fmt),
        )

    def available_option_ids(
        self,
        state: EpisodeState,
        base: BuiltPolicyContext,
        *,
        last_option_id: str | None,
        last_option_status: str | None = None,
    ) -> tuple[str, ...]:
        """Conservative state-conditioned selector affordances.

        The selector still uses the option's natural-language initiation
        conditions for the final choice.  This function only removes options
        that have no legal primitive action in the frozen action space.
        """
        space = base.available_action_space
        ids: list[str] = []
        if not state.visible_entity_ids and not state.visible_sentence_ids and not state.visible_chunk_ids:
            if space.search_options:
                ids.append("O1_START_SEARCH")
        else:
            if space.search_options or space.expand_options or space.read_refs:
                ids.extend(["O2_RESOLVE_FACT", "O3_RESOLVE_BRIDGE"])
            latest = state.newest_observation
            if latest is not None and (
                not latest.results
                or latest.error_code is not None
                or latest.status.value in {"error", "duplicate_action", "invalid_action"}
            ):
                ids.append("O4_RECOVER")
        if space.finish_evidence_refs:
            ids.append("O5_ANSWER")
        # Fallback is always an explicit one-action escape hatch when retrieval
        # is open.  It is intentionally absent at budget finalization.
        if space.search_options or space.expand_options or space.read_refs:
            ids.append("FALLBACK")
        unique = tuple(dict.fromkeys(ids))
        return unique or (("O5_ANSWER",) if space.finish_evidence_refs else ("FALLBACK",))

    def _context_payload(
        self,
        query: str,
        state: EpisodeState,
        trajectory: Sequence[StepRecord],
        base: BuiltPolicyContext,
        *,
        last_option_id: str | None,
        last_option_status: str | None = None,
    ) -> dict[str, Any]:
        latest = base.policy_view.policy_state.latest_attempt
        previous_ids = set(state.semantic_memory_node_ids)
        newly_seen_ids: set[str] = set()
        if trajectory:
            prior = trajectory[-1].state_before
            newly_seen_ids = set(state.semantic_memory_node_ids) - set(prior.semantic_memory_node_ids)
        new_items: list[dict[str, Any]] = []
        old_items: list[dict[str, Any]] = []
        for item in base.policy_view.policy_state.semantic_memory:
            payload = item.model_dump(mode="json")
            stable = state.reference_registry.stable_id_for(payload["ref"])
            if stable in newly_seen_ids:
                new_items.append(payload)
            else:
                # Keep refs and visibility state, but do not repeat old text.
                compact = {key: value for key, value in payload.items() if key not in {"text", "previews"}}
                old_items.append(compact)
        previous_assessment = (
            state.last_assessment.model_dump(mode="json") if state.last_assessment else None
        )
        option_goal = None
        if last_option_id and last_option_id != "FALLBACK":
            try:
                option_goal = self.catalog.get(last_option_id).goal
            except KeyError:
                option_goal = None
        return {
            "previous_assessment": previous_assessment,
            "last_option": last_option_id,
            "last_option_subgoal": option_goal,
            "last_option_status": last_option_status,
            "last_action_and_result": latest,
            "new_text_entering_view": new_items,
            "previously_seen_evidence_refs": old_items,
            "earlier_actions": base.policy_view.policy_state.attempted_actions,
            "available_primitive_actions": base.available_action_space.model_dump(mode="json"),
            "remaining_budget": base.policy_view.policy_state.budget,
        }
