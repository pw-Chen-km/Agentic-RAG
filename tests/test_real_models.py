from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentic_rag.builder import SubstrateBuilder
from agentic_rag.config import BuildConfig
from agentic_rag.storage import Substrate
from agentic_rag.validation import validate_substrate
from conftest import FIXTURE_PATH


@pytest.mark.real_models
@pytest.mark.skipif(
    os.environ.get("RUN_REAL_MODELS") != "1",
    reason="Set RUN_REAL_MODELS=1 to download and exercise real NLP/embedding models",
)
def test_real_spacy_scispacy_minilm_pipeline(tmp_path: Path) -> None:
    output = tmp_path / "real-substrate"
    SubstrateBuilder(
        BuildConfig(corpus_id="real_fixture", split="dev")
    ).build(FIXTURE_PATH, output)
    report = validate_substrate(output)
    assert report.valid, report.errors
    substrate = Substrate.open(output)
    assert any(
        alias.alias == "RAG" and alias.alias_type == "ABBREVIATION"
        for alias in substrate.entity_aliases
    )
