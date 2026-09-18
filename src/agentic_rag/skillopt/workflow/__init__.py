"""Modular, workflow-aware SkillOpt orchestration primitives.

The package is deliberately independent from the legacy SkillOpt adapter.  It
provides small contracts so retrieval, meta, and answer stages can evolve
without changing the Agent runtime.
"""
from .config import WorkflowConfig, StageConfig
from .skill_sections import SkillSections, validate_stage_patch
from .trajectory_views import build_view
from .candidate_replay import ReplayRequest, replay_purpose
from .archive import CandidateArchive, skill_hash
from .checkpoint import WorkflowCheckpoint
from .validation import ValidationMetrics, GateDecision, evaluate_candidate
from .proposals import Proposal
from .runner import WorkflowRunner
from .rule_store import RuleStore, SkillRule, count_tokens

__all__ = ["WorkflowConfig", "StageConfig", "SkillSections", "validate_stage_patch", "build_view", "ReplayRequest", "replay_purpose", "CandidateArchive", "skill_hash", "WorkflowCheckpoint", "ValidationMetrics", "GateDecision", "evaluate_candidate", "Proposal", "WorkflowRunner", "RuleStore", "SkillRule", "count_tokens"]
