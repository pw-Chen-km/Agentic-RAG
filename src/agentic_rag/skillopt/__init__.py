"""Optional SkillOpt v0.2.0 integration for Agentic RAG experiments."""

from agentic_rag.skillopt.adapter import (
    SKILLOPT_AVAILABLE,
    AgenticRAGSkillOptAdapter,
)
from agentic_rag.skillopt.data import (
    SmokeBenchmarkItem,
    load_smoke_split,
    prepare_hotpotqa_smoke_splits,
    validate_hotpotqa_smoke_lineage,
)
from agentic_rag.skillopt.benchmark import (
    BenchmarkItem,
    load_arag_split,
    prepare_arag_smoke_splits,
    smoke_type_quotas,
    split_manifest_profile,
    validate_arag_smoke_lineage,
)
from agentic_rag.skillopt.dataloader import AgenticRAGSkillOptDataLoader
from agentic_rag.skillopt.rollout import RolloutBatch, run_rollout_batch
from agentic_rag.skillopt.trainer import (
    SkillOptUnavailableError,
    create_skillopt_trainer,
    load_skillopt_config,
    run_skillopt_training,
    summarize_rollout_usage,
)

__all__ = [
    "SKILLOPT_AVAILABLE",
    "AgenticRAGSkillOptAdapter",
    "AgenticRAGSkillOptDataLoader",
    "BenchmarkItem",
    "RolloutBatch",
    "SkillOptUnavailableError",
    "SmokeBenchmarkItem",
    "create_skillopt_trainer",
    "load_skillopt_config",
    "load_smoke_split",
    "load_arag_split",
    "prepare_arag_smoke_splits",
    "prepare_hotpotqa_smoke_splits",
    "validate_hotpotqa_smoke_lineage",
    "run_rollout_batch",
    "run_skillopt_training",
    "summarize_rollout_usage",
    "smoke_type_quotas",
    "split_manifest_profile",
    "validate_arag_smoke_lineage",
]
