"""Routing-policy telemetry for configuration-dependent skill runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from agentic_rag.agent.models import (
    AvailableActionSpace,
    ExpandAction,
    FinishAction,
    ResolvedAction,
    ResolvedExpandAction,
    ResolvedFinishAction,
    SearchAction,
    StepRecord,
)

RoutingPolicy = Literal["neutral", "configuration-dependent"]


def route_name(action: Any) -> str | None:
    if isinstance(action, SearchAction):
        return "global_search"
    if isinstance(action, (ExpandAction, ResolvedExpandAction)):
        return "entity_navigation"
    if isinstance(action, (FinishAction, ResolvedFinishAction)):
        return "finish"
    return None


def routing_metadata(
    *,
    policy: RoutingPolicy,
    action: ResolvedAction | None,
    space: AvailableActionSpace,
    state,
    trajectory: Sequence[StepRecord] = (),
    exact_duplicate_blocked: bool = False,
) -> dict[str, Any]:
    """Describe the selected route without deciding semantic entity relevance."""

    visible_entity_refs = sorted({
        ref
        for option in space.expand_options
        for ref in option.source_refs
    })
    selected_route = route_name(action)
    missing = bool(state.last_assessment and state.last_assessment.missing_information)
    local_available = bool(visible_entity_refs)
    if not missing:
        trigger = "no_gap"
    elif local_available:
        trigger = "entity_navigation_available"
    else:
        trigger = "global_only"

    query_reformulated: bool | None = None
    if isinstance(action, SearchAction):
        prior_queries = [
            record.resolved_decision.action.query
            for record in trajectory
            if record.resolved_decision is not None
            and isinstance(record.resolved_decision.action, SearchAction)
        ]
        query_reformulated = action.query not in prior_queries

    if policy == "neutral":
        compliance = "not_applicable"
    elif selected_route == "entity_navigation":
        compliance = "local_selected"
    elif selected_route == "global_search" and not local_available:
        compliance = "global_fallback"
    elif selected_route == "global_search" and local_available:
        # The backend cannot decide whether any visible entity is semantically
        # relevant to a model-generated gap. Preserve that uncertainty.
        compliance = "global_selected_with_local_available"
    elif selected_route == "finish":
        compliance = "finish"
    else:
        compliance = "undetermined"

    return {
        "routing_policy": policy,
        "entity_navigation_available": local_available,
        "visible_navigable_entity_count": len(visible_entity_refs),
        "visible_navigable_entity_refs": visible_entity_refs,
        "selected_route": selected_route,
        "selected_entity_ref": (
            (
                action.source_ref
                if isinstance(action, ExpandAction)
                else action.source_id
                if isinstance(action, ResolvedExpandAction)
                else None
            )
        ),
        "route_trigger": trigger,
        "query_reformulated": query_reformulated,
        "exact_duplicate_blocked": exact_duplicate_blocked,
        "route_compliance": compliance,
    }
