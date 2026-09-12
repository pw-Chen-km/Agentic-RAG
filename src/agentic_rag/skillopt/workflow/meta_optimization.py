"""Small, replaceable Meta Optimizer interface (GRPO-like comparison layer)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

@dataclass(frozen=True)
class WorkflowComparison:
    question_id: str
    branch_a: dict[str, Any]
    branch_b: dict[str, Any]
    match_level: str = "same_question"

class ComparisonStrategy(Protocol):
    def compare(self, branches: Iterable[dict[str, Any]]) -> list[WorkflowComparison]: ...

class MetaOptimizer:
    """Dispatches comparisons in reflection-sized minibatches; LLM is injected."""
    def __init__(self, llm: Any, minibatch_size: int = 5) -> None:
        if minibatch_size <= 0: raise ValueError("minibatch_size must be positive")
        self.llm, self.minibatch_size = llm, minibatch_size

    def propose(self, comparisons: list[WorkflowComparison], prompt_builder: Any) -> list[Any]:
        proposals = []
        for i in range(0, len(comparisons), self.minibatch_size):
            batch = comparisons[i:i+self.minibatch_size]
            proposals.append(self.llm(prompt_builder(batch)))
        return proposals
