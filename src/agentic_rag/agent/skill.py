"""Plain Markdown retrieval skills."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


FIXED_ANSWER_CONTRACT_HEADING = "## Fixed answer contract"


class SkillDocument(BaseModel):
    """Exact Markdown content plus a content-addressed identifier."""

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
            source_path=source.as_posix(),
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

    @property
    def fixed_answer_contract(self) -> str | None:
        """Return the canonical fixed block when the Markdown declares one."""

        index = self.content.find(FIXED_ANSWER_CONTRACT_HEADING)
        if index < 0:
            return None
        return self.content[index:].strip()

    @staticmethod
    def freeze_answer_contract(
        candidate_content: str,
        fixed_answer_contract: str | None,
    ) -> str:
        """Replace a candidate's fixed block with the seed's canonical block."""

        if fixed_answer_contract is None:
            return candidate_content
        if not fixed_answer_contract.startswith(FIXED_ANSWER_CONTRACT_HEADING):
            raise ValueError("fixed answer contract has an invalid heading")
        index = candidate_content.find(FIXED_ANSWER_CONTRACT_HEADING)
        if index < 0:
            raise ValueError("candidate removed the fixed answer contract heading")
        workflow = candidate_content[:index].rstrip()
        if not workflow:
            raise ValueError("candidate removed the retrieval workflow")
        return f"{workflow}\n\n{fixed_answer_contract}\n"

    def write_snapshot(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.content.encode("utf-8"))
        return destination
