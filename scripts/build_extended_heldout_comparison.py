#!/usr/bin/env python3
"""Build the Initial + four optimized Skills + A-RAG comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initial-summary", type=Path, required=True)
    parser.add_argument("--optimized-summary", type=Path, required=True)
    parser.add_argument("--arag-summary", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def agentic_metrics(label: str, condition: str, summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["results"]
    count = len(rows)
    return {
        "condition": condition,
        "label": label,
        "question_count": count,
        "llm_acc_correct": sum(int(row.get("llm_acc") or 0) for row in rows),
        "llm_accuracy": sum(int(row.get("llm_acc") or 0) for row in rows) / count,
        "normalized_exact_correct": sum(
            int(row.get("normalized_exact") or 0) for row in rows
        ),
        "normalized_exact_accuracy": sum(
            int(row.get("normalized_exact") or 0) for row in rows
        ) / count,
        "contain_correct": sum(int(row.get("contain_acc") or 0) for row in rows),
        "contain_accuracy": sum(int(row.get("contain_acc") or 0) for row in rows) / count,
        "blank_answers": sum(
            not str(row.get("predicted_answer") or "").strip() for row in rows
        ),
        "errors": sum(bool(row.get("error_code")) for row in rows),
        "invalid_attempts_total": sum(
            int(row.get("invalid_attempts") or 0) for row in rows
        ),
        "policy_calls_total": sum(
            int((row.get("usage") or {}).get("policy_calls") or 0) for row in rows
        ),
        "total_tokens": sum(
            int((row.get("usage") or {}).get("total_tokens") or 0) for row in rows
        ),
    }


def arag_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["results"]
    count = len(rows)
    return {
        "condition": "a_rag",
        "label": "A-RAG baseline",
        "question_count": count,
        "llm_acc_correct": int(summary["llm_acc_correct"]),
        "llm_accuracy": int(summary["llm_acc_correct"]) / count,
        "normalized_exact_correct": int(summary["normalized_exact_correct"]),
        "normalized_exact_accuracy": int(summary["normalized_exact_correct"]) / count,
        "contain_correct": int(summary["contain_correct"]),
        "contain_accuracy": int(summary["contain_correct"]) / count,
        "blank_answers": int(summary["blank_answers"]),
        "errors": int(summary["errors"]),
        "invalid_attempts_total": None,
        "policy_calls_total": sum(
            int((row.get("usage") or {}).get("policy_calls") or 0) for row in rows
        ),
        "total_tokens": sum(
            int((row.get("usage") or {}).get("total_tokens") or 0) for row in rows
        ),
    }


def main() -> None:
    args = arguments()
    initial = load(args.initial_summary)
    optimized = load(args.optimized_summary)
    arag = load(args.arag_summary)
    initial_ids = [str(row["id"]) for row in initial["results"]]
    arag_ids = [str(row["id"]) for row in arag["results"]]
    optimized_ids = [str(row["id"]) for row in optimized["per_question"]]
    if not (initial_ids == optimized_ids == arag_ids):
        raise ValueError("the six systems do not share the same ordered IDs")

    systems = [agentic_metrics("Initial skill", "initial", initial)]
    systems.extend(optimized["systems"])
    systems.append(arag_metrics(arag))
    payload = {
        "schema_version": "1.0",
        "study": "initial_four_skillopt_conditions_and_arag_external_heldout100",
        "question_ids": initial_ids,
        "systems": systems,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "extended_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Extended External Held-out-100 Comparison",
        "",
        "| System | Qwen judge | Exact | Contain | Blank | Errors | Invalid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system in systems:
        invalid = system.get("invalid_attempts_total")
        lines.append(
            f"| {system['label']} | {system['llm_accuracy']:.1%} | "
            f"{system['normalized_exact_accuracy']:.1%} | "
            f"{system['contain_accuracy']:.1%} | {system['blank_answers']} | "
            f"{system['errors']} | {'N/A' if invalid is None else invalid} |"
        )
    lines.extend(
        [
            "",
            "All rows use the same ordered external HotpotQA Sample-100 and the same Qwen 3.6 judge.",
            "A-RAG is a whole-system baseline; its action interface and budget are not identical to Agentic-RAG.",
            "",
        ]
    )
    (args.root / "extended_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "systems": systems}, indent=2))


if __name__ == "__main__":
    main()
