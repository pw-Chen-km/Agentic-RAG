"""Small, standalone SkillOpt adapter for the official A-RAG runtime.

The package deliberately does not import the previous Agentic-RAG controller,
state schema, substrate, or E/S/C reference system.  It treats an A-RAG
trajectory as a sequence of native tool calls and keeps the optimization
contract independent from the runtime implementation.
"""

from .skill_rules import RuleStore, compile_initial_skill, load_rules, render_formatted_prompt, skill_sha256
from .arag_adapter import normalize_trajectory
from .optimizer_tools import tool_schemas, parse_tool_call
from .workflow import SkillOptWorkflow, WorkflowConfig
from .checkpoint import load_checkpoint, save_checkpoint
from .archive import JsonlArchive
from .candidate_replay import replay_rejected_candidate
from .validation import accept_candidate
from .meta_optimization import compare_parent_candidate

__all__ = [
    "RuleStore",
    "compile_initial_skill",
    "load_rules",
    "render_formatted_prompt",
    "skill_sha256",
    "normalize_trajectory",
    "tool_schemas",
    "parse_tool_call",
    "SkillOptWorkflow",
    "WorkflowConfig",
    "load_checkpoint",
    "save_checkpoint",
    "JsonlArchive",
    "replay_rejected_candidate",
    "accept_candidate",
    "compare_parent_candidate",
]
