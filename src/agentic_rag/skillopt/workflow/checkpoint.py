"""Atomic, versioned checkpoint for resumable staged training."""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Mapping

class WorkflowCheckpoint:
    version='workflow-skillopt-v1'
    def __init__(self, path: str | Path): self.path=Path(path)
    def save(self, state: Mapping[str, Any]) -> None:
        payload={'version':self.version, **dict(state)}
        tmp=self.path.with_suffix(self.path.suffix+'.tmp'); tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8'); os.replace(tmp,self.path)
    def load(self) -> dict[str, Any] | None:
        if not self.path.exists(): return None
        value=json.loads(self.path.read_text(encoding='utf-8'))
        if value.get('version') != self.version: raise ValueError('checkpoint version mismatch')
        return value
