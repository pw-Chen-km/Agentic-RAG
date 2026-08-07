"""Run the same V3.2 evaluation contract across all benchmark datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from agentic_rag.evaluation.profiles import DATASET_PROFILES


DATASET_ORDER = (
    "2wikimultihop",
    "hotpotqa",
    "medical",
    "musique",
    "novel",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--split-root", type=Path, default=Path("data/skillopt"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=list(DATASET_ORDER),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all paths and print commands without running inference.",
    )
    return parser.parse_args()


def _load_jsonl_count(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark split does not exist: {path}")
    count = sum(
        bool(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
    )
    if count < 1:
        raise ValueError(f"benchmark split is empty: {path}")
    return count


def _dataset_paths(
    args: argparse.Namespace, dataset: str
) -> tuple[Path, Path, Path]:
    artifact = args.artifact_root / f"{dataset}_benchmark_exact"
    split = args.split_root / f"{dataset}_smoke" / f"{args.split_name}.jsonl"
    output = args.output / dataset
    return artifact, split, output


def _command(args: argparse.Namespace, dataset: str) -> list[str]:
    artifact, split, output = _dataset_paths(args, dataset)
    if not artifact.is_dir():
        raise FileNotFoundError(f"substrate does not exist: {artifact}")
    count = _load_jsonl_count(split)
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_benchmark_eval.py")),
        "--substrate",
        str(artifact),
        "--split",
        str(split),
        "--config",
        str(args.config),
        "--skill",
        str(args.skill),
        "--output",
        str(output),
        "--expected-count",
        str(count),
        "--dataset",
        dataset,
    ]
    if args.resume and (output / "progress.json").is_file():
        command.append("--resume")
    return command


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _matrix_summary(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        summary_path = args.output / dataset / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": dataset,
                "answer_mode": DATASET_PROFILES[dataset].answer_mode.value,
                "reported_metrics": [
                    metric.value
                    for metric in DATASET_PROFILES[dataset].reported_metrics
                ],
                "question_count": summary["question_count"],
                "contain_correct": summary["contain_correct"],
                "normalized_exact_correct": summary[
                    "normalized_exact_correct"
                ],
                "total_invalid_attempts": summary["total_invalid_attempts"],
                "action_counts": summary["action_counts"],
                "usage": summary["usage"],
                "summary": summary_path.as_posix(),
            }
        )
    return {
        "architecture": "v3.2",
        "datasets": rows,
        "dataset_count": len(rows),
        "note": (
            "Medical and Novel semantic accuracy must be added with "
            "judge_benchmark.py; contain is not a reported metric for them."
        ),
    }


def main() -> None:
    args = _arguments()
    if args.dry_run:
        for dataset in args.datasets:
            print(json.dumps(_command(args, dataset), ensure_ascii=True))
        return
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    for index, dataset in enumerate(args.datasets, start=1):
        print(
            f"DATASET {index}/{len(args.datasets)} START {dataset}",
            flush=True,
        )
        subprocess.run(_command(args, dataset), check=True)
        print(
            f"DATASET {index}/{len(args.datasets)} DONE {dataset}",
            flush=True,
        )
    summary = _matrix_summary(args)
    _write_json(args.output / "matrix_summary.json", summary)
    print(
        f"MATRIX DONE datasets={summary['dataset_count']} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
