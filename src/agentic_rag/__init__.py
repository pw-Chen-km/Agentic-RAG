"""Typed substrate and retrieval interfaces for Agentic RAG."""

__version__ = "0.1.0"

from agentic_rag.adapters import (
    HotpotQAAdapter,
    HotpotQABenchmarkExactAdapter,
    SourceAdapter,
)
from agentic_rag.agent import AgentConfig, AgentController, AgentHarness
from agentic_rag.bridge import SubstrateBridge
from agentic_rag.builder import SubstrateBuilder
from agentic_rag.config import BuildConfig
from agentic_rag.retrieval import Retriever
from agentic_rag.storage import Substrate

__all__ = [
    "BuildConfig",
    "AgentConfig",
    "AgentController",
    "AgentHarness",
    "HotpotQAAdapter",
    "HotpotQABenchmarkExactAdapter",
    "Retriever",
    "SourceAdapter",
    "Substrate",
    "SubstrateBridge",
    "SubstrateBuilder",
]
