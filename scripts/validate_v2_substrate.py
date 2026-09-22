"""Validate a cloud substrate before v2 smoke or full runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_rag.substrate.storage import Substrate


REQUIRED_EMBEDDING = "qwen3-embedding:4b"


def validate(path: Path, *, expected_model: str = REQUIRED_EMBEDDING) -> dict[str, Any]:
    substrate = Substrate.open(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    raw_model = manifest.get("embedding_model")
    model = raw_model.get("name") if isinstance(raw_model, dict) else raw_model
    if model != expected_model:
        raise ValueError(
            f"substrate embedding model is {model!r}; expected {expected_model!r}. "
            "Rebuild only this substrate before running v2."
        )
    index_reports = {}
    dimensions = set()
    for target in ("chunk", "sentence", "entity"):
        metadata_path = path / "indexes" / f"dense_{target}" / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"missing dense {target} index metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model") != expected_model:
            raise ValueError(f"dense_{target} uses {metadata.get('model')!r}")
        dimensions.add(int(metadata.get("dimension") or 0))
        index_reports[target] = metadata
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError(f"dense index dimensions disagree: {sorted(dimensions)}")
    return {
        "substrate": path.resolve().as_posix(),
        "embedding_model": model,
        "embedding_dimension": next(iter(dimensions)),
        "scope_count": len(substrate.doc_ids_by_scope),
        "document_count": len(substrate.documents),
        "sentence_count": len(substrate.sentences),
        "entity_count": len(substrate.entities),
        "indexes": index_reports,
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--expected-model", default=REQUIRED_EMBEDDING)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.substrate, expected_model=args.expected_model)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
