"""Typed substrate and retrieval interfaces for Agentic RAG."""

__version__ = "0.2.0"

from agentic_rag.substrate.adapters import (
    HotpotQAAdapter,
    HotpotQABenchmarkExactAdapter,
    SourceAdapter,
)
from agentic_rag.evaluation.profiles import (
    DatasetProfile,
    HOTPOTQA_PROFILE,
    get_dataset_profile,
)
from agentic_rag.agent import AgentConfig, AgentController, AgentHarness
from agentic_rag.substrate.bridge import SubstrateBridge
from agentic_rag.substrate.builder import SubstrateBuilder
from agentic_rag.config import BuildConfig
from agentic_rag.substrate.retrieval import Retriever
from agentic_rag.substrate.storage import Substrate

__all__ = [
    "BuildConfig",
    "AgentConfig",
    "AgentController",
    "AgentHarness",
    "DatasetProfile",
    "HOTPOTQA_PROFILE",
    "HotpotQAAdapter",
    "HotpotQABenchmarkExactAdapter",
    "Retriever",
    "SourceAdapter",
    "Substrate",
    "SubstrateBridge",
    "SubstrateBuilder",
    "get_dataset_profile",
]
