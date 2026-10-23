"""Small, standalone SkillOpt adapter for the official A-RAG runtime.

The package deliberately does not import the previous Agentic-RAG controller,
state schema, substrate, or E/S/C reference system.  It treats an A-RAG
trajectory as a sequence of native tool calls and keeps the optimization
contract independent from the runtime implementation.
"""

from .skill_rules import RuleStore, compile_initial_skill, load_rules, render_formatted_prompt, skill_sha256
from .arag_adapter import normalize_trajectory
from .optimizer_tools import tool_schemas, parse_tool_call
from .checkpoint import load_checkpoint, save_checkpoint
from .archive import JsonlArchive
from .candidate_replay import replay_rejected_candidate
from .validation import accept_candidate
from .meta_optimization import compare_parent_candidate
from .paired_summary import HRO_VERSION, action_sequence, is_informative_pair, summarize_pair
from .action_space import ACTION_SPACE_VERSION, action_space_contract, configured_actions
from .hro_v1 import HRO_V1_VERSION, UNSCORED, transition_group
from .retrieval_diagnostics import DIAGNOSTICS_VERSION, build_retrieval_diagnostics
from .context_rendering import (
    CONTEXT_RENDERER_VERSION,
    render_phase1_merge_context,
    render_phase1_reflection_context,
)

__all__ = [
    "RuleStore",
    "compile_initial_skill",
    "load_rules",
    "render_formatted_prompt",
    "skill_sha256",
    "normalize_trajectory",
    "tool_schemas",
    "parse_tool_call",
    "load_checkpoint",
    "save_checkpoint",
    "JsonlArchive",
    "replay_rejected_candidate",
    "accept_candidate",
    "compare_parent_candidate",
    "HRO_VERSION",
    "action_sequence",
    "is_informative_pair",
    "summarize_pair",
    "ACTION_SPACE_VERSION",
    "action_space_contract",
    "configured_actions",
    "HRO_V1_VERSION",
    "UNSCORED",
    "transition_group",
    "DIAGNOSTICS_VERSION",
    "build_retrieval_diagnostics",
    "CONTEXT_RENDERER_VERSION",
    "render_phase1_merge_context",
    "render_phase1_reflection_context",
]
