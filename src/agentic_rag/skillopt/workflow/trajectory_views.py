"""Pluggable optimizer-facing trajectory renderers."""
from __future__ import annotations
from typing import Any, Iterable

def _step(row: Any) -> dict[str, Any]:
    if hasattr(row, "model_dump"): return row.model_dump(mode="json")
    return dict(row) if isinstance(row, dict) else {"value": str(row)}

def build_view(trajectory: Iterable[Any], representation: str = "workflow", *, include_text: bool = True) -> dict[str, Any]:
    """Render without mutating audit records; all views share the same steps."""
    steps = [_step(x) for x in trajectory]
    if representation not in {"raw", "result", "progress", "workflow", "answer"}: raise ValueError(f"unknown representation: {representation}")
    if representation == "raw": return {"representation": representation, "steps": steps}
    result = []
    for item in steps:
        x = {k: v for k, v in item.items() if k not in {"raw_text", "conversation"}}
        if not include_text or representation == "progress":
            for key in ("text", "results", "observation", "retrieval_text"): x.pop(key, None)
        result.append(x)
    return {"representation": representation, "steps": result}

def batch_views(trajectories: Iterable[Iterable[Any]], representation: str, *, minibatch_size: int = 5) -> list[dict[str, Any]]:
    if minibatch_size <= 0: raise ValueError("minibatch_size must be positive")
    rows = list(trajectories)
    return [ {"representation": representation, "episodes": [build_view(t, representation) for t in rows[i:i+minibatch_size]]} for i in range(0, len(rows), minibatch_size) ]
