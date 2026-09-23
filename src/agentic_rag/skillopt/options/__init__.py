"""Options SkillOpt v1 public API.

This package is independent from the legacy and workflow-rules-v2 trainers.
The runtime option controller may use the same JSON catalogue, while the
optimizer only interacts with the immutable store and constrained edits here.
"""
from .option_store import OptionSpec, OptionStore
from .option_edits import EDIT_OPERATIONS, EditReceipt, apply_edits
from .option_schema import optimizer_edit_schema, option_policy_schema, option_selector_schema
from .option_validation import validate_no_episode_literals, validate_option, validate_store
from .option_trajectory import OptionEvent, OptionTrajectory
from .option_router import RoutedCases, route_episode, route_episodes
from .coordinator import OptionSkillOptCoordinator, ReflectionResult
from .option_renderer import (
    build_optimizer_trajectory,
    build_policy_view,
    build_selection_view,
    build_termination_view,
    render_selected_option,
    render_selector_catalog,
)

__all__ = [
    "OptionSpec",
    "OptionStore",
    "EDIT_OPERATIONS",
    "EditReceipt",
    "apply_edits",
    "option_policy_schema",
    "option_selector_schema",
    "optimizer_edit_schema",
    "validate_no_episode_literals",
    "validate_option",
    "validate_store",
    "OptionEvent",
    "OptionTrajectory",
    "RoutedCases",
    "route_episode",
    "route_episodes",
    "OptionSkillOptCoordinator",
    "ReflectionResult",
    "build_optimizer_trajectory",
    "build_policy_view",
    "build_selection_view",
    "build_termination_view",
    "render_selected_option",
    "render_selector_catalog",
]
