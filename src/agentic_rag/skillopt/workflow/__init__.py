"""Modular, workflow-aware SkillOpt orchestration primitives.

The package is deliberately independent from the legacy SkillOpt adapter.  It
provides small contracts so retrieval, meta, and answer stages can evolve
without changing the Agent runtime.
"""
from .config import WorkflowConfig, StageConfig
from .skill_sections import SkillSections, validate_stage_patch
from .trajectory_views import build_view
from .candidate_replay import ReplayRequest, replay_purpose

__all__ = ["WorkflowConfig", "StageConfig", "SkillSections", "validate_stage_patch", "build_view", "ReplayRequest", "replay_purpose"]
