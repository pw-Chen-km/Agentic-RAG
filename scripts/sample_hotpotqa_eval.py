"""Create a deterministic held-out HotpotQA evaluation sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


_SPACE = re.compile(r"\s+")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--bridge-count", type=int, default=16)
    parser.add_argument("--comparison-count", type=int, default=4)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_excluded_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            excluded.add(str(row["id"]))
    return excluded


def _selection_key(seed: int, question_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{question_id}".encode()).hexdigest()


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    tokenized = re.sub(r"[^\w]+", " ", without_marks, flags=re.UNICODE)
    return _SPACE.sub(" ", tokenized).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _arguments()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.bridge_count < 0 or args.comparison_count < 0:
        raise ValueError("sample counts must be non-negative")

    questions_path = args.source / "questions.json"
    chunks_path = args.source / "chunks.json"
    questions = _load_json(questions_path)
    chunks = _load_json(chunks_path)
    if not isinstance(questions, list) or not isinstance(chunks, list):
        raise ValueError("source questions.json and chunks.json must be lists")

    excluded = _load_excluded_ids(args.exclude)
    requested = {
        "bridge": args.bridge_count,
        "comparison": args.comparison_count,
    }
    selected: list[tuple[int, dict[str, Any]]] = []
    for question_type, count in requested.items():
        candidates = [
            (index, row)
            for index, row in enumerate(questions)
            if isinstance(row, dict)
            and str(row.get("id")) not in excluded
            and str(row.get("question_type")) == question_type
        ]
        candidates.sort(
            key=lambda item: _selection_key(args.seed, str(item[1]["id"]))
        )
        if len(candidates) < count:
            raise ValueError(
                f"not enough {question_type} candidates: {len(candidates)} < {count}"
            )
        selected.extend(candidates[:count])
    selected.sort(key=lambda item: _selection_key(args.seed, str(item[1]["id"])))

    normalized_corpus = _normalize("\n".join(str(chunk) for chunk in chunks))
    evidence_sentence_count = 0
    fully_matched_evidence_sentence_count = 0
    evidence_group_count = 0
    matched_evidence_group_count = 0
    missing_evidence: list[dict[str, str]] = []
    output_rows: list[dict[str, Any]] = []
    manifest_items: list[dict[str, Any]] = []
    for source_index, row in selected:
        question_id = str(row["id"])
        evidence = row.get("evidence") or []
        for group in evidence:
            if not isinstance(group, list) or len(group) != 2:
                continue
            evidence_group_count += 1
            title_match = bool(
                _normalize(str(group[0]))
                and _normalize(str(group[0])) in normalized_corpus
            )
            sentences = group[1]
            if not isinstance(sentences, list):
                continue
            sentence_match = False
            for sentence in sentences:
                normalized_sentence = _normalize(str(sentence))
                if not normalized_sentence:
                    continue
                evidence_sentence_count += 1
                if normalized_sentence in normalized_corpus:
                    fully_matched_evidence_sentence_count += 1
                    sentence_match = True
                    continue
                prefix = " ".join(normalized_sentence.split()[:8])
                if prefix and prefix in normalized_corpus:
                    sentence_match = True
            if title_match or sentence_match:
                matched_evidence_group_count += 1
            else:
                missing_evidence.append(
                    {"id": question_id, "sentence": f"group:{group[0]}"}
                )

        normalized_answer = _normalize(str(row["answer"]))
        if not normalized_answer or normalized_answer not in normalized_corpus:
            missing_evidence.append(
                {"id": question_id, "sentence": f"answer:{row['answer']}"}
            )

        output_rows.append(
            {
                "answer": str(row["answer"]),
                "id": question_id,
                "question": str(row["question"]),
                "question_type": str(row["question_type"]),
                "scope_id": "hotpotqa:benchmark_exact:dev",
                "source": str(row.get("source") or "hotpotqa"),
            }
        )
        manifest_items.append(
            {
                "id": question_id,
                "question_type": str(row["question_type"]),
                "selection_hash": _selection_key(args.seed, question_id),
                "source_row_index": source_index,
            }
        )

    if missing_evidence:
        preview = missing_evidence[:3]
        raise ValueError(
            f"selected supporting evidence missing from chunks.json: {preview}"
        )

    args.output.mkdir(parents=True)
    questions_output = args.output / "questions.jsonl"
    questions_output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in output_rows
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0",
        "purpose": "held_out_skill_ablation",
        "selection": {
            "algorithm": "sha256(f'{seed}\\0{question_id}')-v1",
            "seed": args.seed,
            "bridge_count": args.bridge_count,
            "comparison_count": args.comparison_count,
            "total_count": len(output_rows),
            "excluded_id_count": len(excluded),
            "excluded_files": [path.as_posix() for path in args.exclude],
        },
        "source": {
            "directory": args.source.as_posix(),
            "questions_sha256": _sha256(questions_path),
            "chunks_sha256": _sha256(chunks_path),
            "question_count": len(questions),
            "chunk_count": len(chunks),
        },
        "evidence_validation": {
            "evidence_group_count": evidence_group_count,
            "matched_evidence_group_count": matched_evidence_group_count,
            "evidence_sentence_count": evidence_sentence_count,
            "fully_matched_evidence_sentence_count": (
                fully_matched_evidence_sentence_count
            ),
            "all_supporting_groups_locatable": True,
            "all_gold_answers_present": True,
        },
        "output": {
            "path": questions_output.as_posix(),
            "sha256": _sha256(questions_output),
            "size_bytes": questions_output.stat().st_size,
        },
        "items": manifest_items,
    }
    _write_json(args.output / "manifest.json", manifest)
    print(
        f"sampled={len(output_rows)} bridge={args.bridge_count} "
        f"comparison={args.comparison_count} excluded={len(excluded)} "
        f"evidence_groups={matched_evidence_group_count}/{evidence_group_count}"
    )


if __name__ == "__main__":
    main()
