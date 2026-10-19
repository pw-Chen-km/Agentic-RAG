#!/usr/bin/env python3
"""Build the two offline indices required by the EntityHop ablation.

Only the configured chunk corpus is read.  In particular this command has no
argument for questions, answers, or evidence annotations and cannot consume
those files accidentally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arag.tools.bm25_index import BM25Index  # noqa: E402
from arag.tools.entity_link_index import (  # noqa: E402
    VERSION,
    build_entity_index,
    load_chunks,
    source_chunks_sha256,
    write_entity_index,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="Novel corpus chunks JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for index artefacts")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument(
        "--no-abbreviations",
        action="store_true",
        help="Disable scispaCy abbreviation detector (must be explicit)",
    )
    args = parser.parse_args(argv)
    chunks_path = Path(args.chunks).resolve()
    output = Path(args.output_dir).resolve()
    rows = load_chunks(chunks_path)
    if not rows:
        raise SystemExit("chunks file is empty")

    # Direct BM25 is intentionally independent from the entity substrate.
    direct_path = output / "direct_bm25.json"
    direct_index = BM25Index(rows)
    direct_index.save(direct_path)

    linked = build_entity_index(
        rows,
        model_name=args.spacy_model,
        enable_abbreviations=not args.no_abbreviations,
        source_path=chunks_path,
    )
    linked_path = output / "entity_linked.json"
    write_entity_index(linked, linked_path)

    manifest = {
        "schema_version": "entityhop-index-manifest-v1",
        "entity_index_version": VERSION,
        "source_chunks_sha256": source_chunks_sha256(chunks_path),
        "source_chunk_count": len(rows),
        "spacy_model": args.spacy_model,
        "enable_abbreviations": not args.no_abbreviations,
        "direct_bm25_sha256": _sha256(direct_path),
        "entity_linked_sha256": _sha256(linked_path),
        "questions_used": False,
        "answers_used": False,
        "gold_evidence_used": False,
        "artifacts": {
            "direct_bm25": str(direct_path),
            "entity_linked": str(linked_path),
        },
    }
    (output / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

