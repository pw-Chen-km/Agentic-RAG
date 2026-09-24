"""Rebuild evaluation-only gold evidence sidecars without rebuilding indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_rag.substrate.models import GoldSupport
from agentic_rag.substrate.storage import EvaluationSidecars, Substrate, write_records


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain a JSON array of objects")
    return value


def align_evidence(substrate: Substrate, text: str) -> dict[str, Any]:
    needle = normalize(text)
    if not needle:
        return {"status": "not_evaluable", "reason": "empty evidence", "sentence_ids": [], "chunk_ids": []}
    sentence_ids = [item.sentence_id for item in substrate.sentences if needle in normalize(item.text)]
    chunk_ids = [item.chunk_id for item in substrate.chunks if needle in normalize(item.text)]
    if not sentence_ids and not chunk_ids:
        return {"status": "not_evaluable", "reason": "evidence text not found in substrate", "sentence_ids": [], "chunk_ids": []}
    return {"status": "aligned", "reason": None, "sentence_ids": sentence_ids, "chunk_ids": chunk_ids}


def repair_hotpot(source_rows: list[dict[str, Any]], substrate: Substrate, sidecars: EvaluationSidecars) -> list[dict[str, Any]]:
    provenance = {
        (item.original_title, item.original_sentence_id): item
        for item in sidecars.source_sentence_provenance
    }
    records: list[dict[str, Any]] = []
    supports: list[GoldSupport] = []
    benchmark_ids = {item.question_id for item in sidecars.benchmark_questions}
    source_id_to_question_id = {
        str(item.source_question_id): item.question_id
        for item in sidecars.benchmark_questions
        if item.source_question_id
    }
    for row in source_rows:
        question_id = str(row.get("_id") or row.get("id") or row.get("qid") or "")
        target_question_id = source_id_to_question_id.get(question_id, question_id)
        if target_question_id not in benchmark_ids:
            continue
        aligned: list[dict[str, Any]] = []
        for index, fact in enumerate(row.get("supporting_facts") or []):
            if not isinstance(fact, (list, tuple)) or len(fact) < 2:
                raise ValueError(f"HotpotQA {question_id}: invalid supporting fact")
            title, position = str(fact[0]), int(fact[1])
            source = provenance.get((title, position))
            if source is None:
                raise ValueError(f"HotpotQA {question_id}: missing provenance for {title}[{position}]")
            supports.append(GoldSupport(
                scope_id=source.scope_id,
                doc_id=source.doc_id,
                title=title,
                source_sentence_pos=position,
                source_sentence_text=source.original_sentence_text,
                sentence_id=source.sentence_id,
                question_id=target_question_id,
                fact_id=f"sf:{index:04d}",
            ))
            aligned.append({
                "fact_id": f"sf:{index:04d}",
                "title": title,
                "source_sentence_pos": position,
                "source_sentence_text": source.original_sentence_text,
                "sentence_id": source.sentence_id,
                "status": "aligned" if source.sentence_id else "not_evaluable",
                "reason": None if source.sentence_id else "blank source sentence",
            })
        records.append({"question_id": target_question_id, "source_question_id": question_id,
                        "evidence": [], "alignment": aligned})
    if len(records) != len(benchmark_ids):
        raise ValueError("HotpotQA source and substrate question IDs do not match")
    write_records(substrate.root / "evaluation" / "gold_support.parquet", supports, "gold_support")
    return records


def repair_benchmark(source_rows: list[dict[str, Any]], substrate: Substrate, sidecars: EvaluationSidecars) -> list[dict[str, Any]]:
    questions = list(sidecars.benchmark_questions)
    by_source: dict[str, list[int]] = {}
    for index, item in enumerate(questions):
        key = item.source_question_id or item.question_id
        by_source.setdefault(str(key), []).append(index)
    updated = list(questions)
    records: list[dict[str, Any]] = []
    seen_source_rows: dict[str, int] = {}
    for row_index, row in enumerate(source_rows):
        source_id = str(row.get("id") or row.get("_id") or "")
        candidates = by_source.get(source_id, [])
        occurrence = seen_source_rows.get(source_id, 0)
        seen_source_rows[source_id] = occurrence + 1
        if candidates and occurrence < len(candidates):
            candidates = [candidates[occurrence]]
        elif not candidates and row_index < len(updated):
            candidates = [row_index]
        if not candidates:
            raise ValueError(f"Benchmark row {row_index} cannot be matched to substrate question")
        target_index = candidates[0]
        item = updated[target_index]
        evidence = row.get("evidence")
        if isinstance(evidence, str):
            evidence = [part.strip() for part in evidence.splitlines() if part.strip()]
        elif isinstance(evidence, list):
            evidence = [str(part).strip() for part in evidence if str(part).strip()]
        else:
            evidence = []
        relations = row.get("evidence_relations")
        if isinstance(relations, str):
            relations = [part.strip() for part in relations.splitlines() if part.strip()]
        elif isinstance(relations, list):
            relations = [str(part).strip() for part in relations if str(part).strip()]
        else:
            relations = []
        updated[target_index] = replace(
            item,
            evidence=tuple(evidence),
            evidence_triple=(str(row["evidence_triple"]) if row.get("evidence_triple") is not None else item.evidence_triple),
            evidence_relations=tuple(relations),
        )
        aligned = [align_evidence(substrate, text) for text in evidence]
        records.append({
            "question_id": item.question_id,
            "source_question_id": source_id,
            "evidence": evidence,
            "evidence_triple": row.get("evidence_triple"),
            "evidence_relations": relations,
            "alignment": aligned,
        })
    write_records(substrate.root / "evaluation" / "benchmark_questions.parquet", updated, "benchmark_questions")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("hotpotqa", "novel", "medical"), required=True)
    parser.add_argument("--source-questions", type=Path, required=True)
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output already exists; refusing to overwrite: {args.output}")
    shutil.copytree(args.substrate, args.output)
    substrate = Substrate.open(args.output)
    sidecars = EvaluationSidecars.open(args.output)
    source_rows = load_rows(args.source_questions)
    records = repair_hotpot(source_rows, substrate, sidecars) if args.dataset == "hotpotqa" else repair_benchmark(source_rows, substrate, sidecars)
    evaluation = args.output / "evaluation"
    payload = {"version": "gold-evidence-sidecar-v1", "dataset": args.dataset, "records": records}
    payload_path = evaluation / "gold_evidence.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = args.output / "manifest.json"
    substrate_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.dataset == "hotpotqa":
        substrate_manifest.setdefault("record_counts", {})["gold_support"] = sum(
            len(item.get("alignment") or []) for item in records
        )
    manifest_path.write_text(
        json.dumps(substrate_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "version": "gold-evidence-lineage-v1",
        "dataset": args.dataset,
        "source_questions": args.source_questions.resolve().as_posix(),
        "source_questions_sha256": digest(args.source_questions),
        "substrate_manifest_sha256": digest(manifest_path),
        "gold_evidence_sha256": digest(payload_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_count": len(records),
        "raw_evidence_count": sum(len(item.get("evidence") or []) for item in records),
        "not_evaluable_count": sum(
            1 for item in records for alignment in item.get("alignment", []) if alignment.get("status") == "not_evaluable"
        ),
    }
    (evaluation / "gold_evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
