"""Build configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_id: str = Field(min_length=1)
    split: str = Field(default="dev", min_length=1)
    dataset: str = "hotpotqa"
    source_format: Literal[
        "hotpotqa_scoped",
        "hotpotqa_benchmark_exact",
    ] = "hotpotqa_scoped"
    benchmark_scope_id: str | None = None
    validate_benchmark_profile: bool = False
    max_chunk_tokens: int = Field(default=256, ge=1)
    spacy_model: str = "en_core_web_sm"
    enable_abbreviations: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=64, ge=1)
    embedding_device: str | None = None
    bm25_stopwords: str | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "BuildConfig":
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if "substrate" in raw:
            substrate = raw.get("substrate", {})
            entity = raw.get("entity", {})
            retrieval = raw.get("retrieval", {})
            dense = retrieval.get("dense", {}) if isinstance(retrieval, dict) else {}
            raw = {
                "corpus_id": substrate["corpus_id"],
                "split": substrate.get("split", "dev"),
                "dataset": substrate.get("dataset", "hotpotqa"),
                "source_format": substrate.get(
                    "source_format", "hotpotqa_scoped"
                ),
                "benchmark_scope_id": substrate.get("benchmark_scope_id"),
                "validate_benchmark_profile": substrate.get(
                    "validate_benchmark_profile", False
                ),
                "max_chunk_tokens": substrate.get("max_chunk_tokens", 256),
                "spacy_model": substrate.get("sentence_model", "en_core_web_sm"),
                "enable_abbreviations": entity.get("abbreviation_detector") is not None,
                "embedding_model": dense.get(
                    "model", "sentence-transformers/all-MiniLM-L6-v2"
                ),
                "embedding_batch_size": dense.get("batch_size", 64),
                "embedding_device": dense.get("device"),
            }
        return cls.model_validate(raw)
