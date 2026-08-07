"""Typed multi-substrate construction, storage, and retrieval interfaces."""

from agentic_rag.substrate.adapters import (
    BenchmarkExactAdapter,
    HotpotQAAdapter,
    HotpotQABenchmarkExactAdapter,
    SourceAdapter,
)
from agentic_rag.substrate.bridge import SubstrateBridge
from agentic_rag.substrate.builder import SubstrateBuilder
from agentic_rag.substrate.retrieval import Retriever
from agentic_rag.substrate.storage import Substrate

__all__ = [
    "BenchmarkExactAdapter",
    "HotpotQAAdapter",
    "HotpotQABenchmarkExactAdapter",
    "Retriever",
    "SourceAdapter",
    "Substrate",
    "SubstrateBridge",
    "SubstrateBuilder",
]
