"""SkillOpt-compatible plain Markdown skill documents."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class SkillDocument(BaseModel):
    """Exact Markdown contents plus a content-addressed version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = Path(path)
        raw = source.read_bytes()
        content = raw.decode("utf-8")
        if not content.strip():
            raise ValueError(f"skill Markdown is empty: {source}")
        return cls(
            content=content,
            sha256=hashlib.sha256(raw).hexdigest(),
            source_path=str(source),
        )

    @classmethod
    def from_text(cls, content: str, *, source_path: str | None = None) -> Self:
        if not content.strip():
            raise ValueError("skill Markdown must not be empty")
        raw = content.encode("utf-8")
        return cls(
            content=content,
            sha256=hashlib.sha256(raw).hexdigest(),
            source_path=source_path,
        )

    @property
    def version(self) -> str:
        return self.sha256

    def write_snapshot(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.content, encoding="utf-8")
        return destination
