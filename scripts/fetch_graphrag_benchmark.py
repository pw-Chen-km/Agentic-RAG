"""Fetch and normalize the official GraphRAG-Benchmark datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://raw.githubusercontent.com/GraphRAG-Bench/GraphRAG-Benchmark/main/Datasets"
RESOLVED_COMMIT = "fdbab5959b18c96532580877ffe27d112bccc0ec"
FILES = {
    "novel": ("Corpus/novel.json", "Questions/novel_questions.json"),
    "medical": ("Corpus/medical.json", "Questions/medical_questions.json"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunks(text: str, size: int = 5000) -> list[str]:
    words = text.split()
    result, current, length = [], [], 0
    for word in words:
        if current and length + len(word) + 1 > size:
            result.append(" ".join(current)); current, length = [], 0
        current.append(word); length += len(word) + 1
    if current: result.append(" ".join(current))
    return result


def normalize(dataset: str, root: Path) -> dict:
    target = root / dataset
    target.mkdir(parents=True, exist_ok=True)
    corpus_path, questions_path = target / "corpus.json", target / "questions.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if isinstance(corpus, dict):
        corpus = [corpus]
    if not isinstance(corpus, list) or not isinstance(questions, list):
        raise ValueError(f"{dataset}: source files must contain arrays")
    normalized_chunks: list[str] = []
    for item in corpus:
        if not isinstance(item, dict) or not item.get("context"):
            raise ValueError(f"{dataset}: invalid corpus row")
        title = str(item.get("corpus_name") or dataset)
        for part in chunks(str(item["context"])):
            normalized_chunks.append(f"{title} — {part}")
    (target / "chunks.json").write_text(json.dumps([f"{i}:{text}" for i, text in enumerate(normalized_chunks)], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "questions.json").write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "dataset": dataset,
        "source_repository": "https://github.com/GraphRAG-Bench/GraphRAG-Benchmark",
        "resolved_commit": RESOLVED_COMMIT,
        "source_urls": [f"{BASE}/{part}" for part in FILES[dataset]],
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "MIT (repository LICENSE; source datasets retain their upstream notices)",
        "files": {name: {"path": str(target / name), "sha256": sha256(target / name), "size_bytes": (target / name).stat().st_size} for name in ("corpus.json", "questions.json", "chunks.json")},
        "counts": {"corpus_documents": len(corpus), "questions": len(questions), "chunks": len(normalized_chunks)},
        "chunking": {"method": "deterministic whitespace chunks", "max_characters": 5000},
    }
    (target / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/graphrag_benchmark"))
    parser.add_argument("--datasets", nargs="+", choices=tuple(FILES), default=list(FILES))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        target = args.root / dataset
        target.mkdir(parents=True, exist_ok=True)
        corpus_url, questions_url = (f"{BASE}/{part}" for part in FILES[dataset])
        if args.download or not (target / "corpus.json").exists():
            urllib.request.urlretrieve(corpus_url, target / "corpus.json")
        if args.download or not (target / "questions.json").exists():
            urllib.request.urlretrieve(questions_url, target / "questions.json")
        print(json.dumps(normalize(dataset, args.root), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
