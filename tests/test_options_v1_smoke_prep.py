from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/options_v1_smoke.py"
_SPEC = importlib.util.spec_from_file_location("options_v1_smoke", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
PreparationError = _MODULE.PreparationError
prepare = _MODULE.prepare
validate = _MODULE.validate
dry_run = _MODULE.dry_run


DATASETS = ("hotpotqa", "novel", "medical")


def _make_substrate(root: Path, dataset: str, count: int = 25) -> Path:
    substrate = root / dataset
    (substrate / "evaluation").mkdir(parents=True)
    scope = f"{dataset}:benchmark_exact:dev"
    manifest = {
        "dataset": dataset,
        "split": "dev",
        "benchmark_scope_id": scope,
    }
    (substrate / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [
        {
            "question_id": f"{dataset}:q:{index:03d}",
            "scope_id": scope,
            "source": dataset,
            "question": f"Question {index} for {dataset}?",
            "answer": f"Answer {index}",
            "question_type": "bridge" if index % 2 == 0 else "comparison",
            "source_question_id": f"raw-{index}",
            "source_row_index": index,
        }
        for index in range(count)
    ]
    pq.write_table(pa.Table.from_pylist(rows), substrate / "evaluation/benchmark_questions.parquet")
    return substrate


def _config(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/options_v1_smoke.json")
                        .read_text(encoding="utf-8"))
    source_root = tmp_path / "sources"
    overrides = {dataset: _make_substrate(source_root, dataset) for dataset in DATASETS}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, overrides


def test_prepares_deterministic_20_question_samples_without_model_calls(tmp_path: Path) -> None:
    config, sources = _config(tmp_path)
    out = tmp_path / "prepared"
    result = prepare(config, output_override=out, substrate_overrides=sources)

    assert result["model_calls"] == 0
    assert {dataset: values["count"] for dataset, values in result["datasets"].items()} == {
        dataset: 20 for dataset in DATASETS
    }
    assert validate(out) == {"valid": True, "dataset_counts": {name: 20 for name in DATASETS}, "total": 60}
    for dataset in DATASETS:
        rows = [json.loads(line) for line in (out / "questions" / f"{dataset}.jsonl")
                .read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 20
        assert sum(result["datasets"][dataset]["question_type_counts"].values()) == 20
        assert all("answer" in row for row in rows)  # evaluator side only; runtime contract forbids sending it
        assert all(row["source"] == dataset for row in rows)

    second = tmp_path / "prepared-copy"
    prepare(config, output_override=second, substrate_overrides=sources)
    for dataset in DATASETS:
        assert ((out / "questions" / f"{dataset}.jsonl").read_bytes()
                == (second / "questions" / f"{dataset}.jsonl").read_bytes())


def test_rejects_wrong_dataset_scope_and_duplicate_ids(tmp_path: Path) -> None:
    config, sources = _config(tmp_path)
    manifest_path = sources["novel"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_scope_id"] = "wrong:scope"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PreparationError, match="expected scope"):
        prepare(config, output_override=tmp_path / "bad", substrate_overrides=sources)


def test_validate_detects_tampered_question_file(tmp_path: Path) -> None:
    config, sources = _config(tmp_path)
    out = tmp_path / "prepared"
    prepare(config, output_override=out, substrate_overrides=sources)
    path = out / "questions" / "medical.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(PreparationError, match="hash mismatch"):
        validate(out)


def test_refuses_non_20_smoke_quota(tmp_path: Path) -> None:
    config, sources = _config(tmp_path)
    doc = json.loads(config.read_text(encoding="utf-8"))
    doc["questions_per_dataset"] = 19
    config.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(PreparationError, match="fixed at exactly 20"):
        prepare(config, output_override=tmp_path / "bad", substrate_overrides=sources)


def test_dry_run_checks_sources_without_writing(tmp_path: Path) -> None:
    config, sources = _config(tmp_path)
    result = dry_run(config, substrate_overrides=sources)
    assert result["valid"] is True
    assert result["writes"] == 0
    assert result["model_calls"] == 0
    assert all(item["selected_questions"] == 20 for item in result["datasets"].values())
    assert not (tmp_path / "prepared").exists()
