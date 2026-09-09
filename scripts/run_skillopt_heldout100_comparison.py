"""Evaluate four completed SkillOpt best skills on one external Sample-100."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentic_rag.skillopt.heldout100 import (
    build_heldout100_plan,
    execute_heldout100_plan,
)


def arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run Raw, Result-Driven, Organized + Supporting-Fact Labels, then "
            "Progress-Abstracted best skills "
            "sequentially on the fixed external HotpotQA Sample-100."
        )
    )
    parser.add_argument(
        "--pilot-root",
        type=Path,
        required=True,
        help="Completed output root from run_skillopt_trajectory_ablation.py.",
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=root / "data" / "evaluations" / "hotpotqa_sample100_seed42",
    )
    parser.add_argument(
        "--substrate",
        type=Path,
        default=Path(
            "/home/jj/Large_Space_A/chiu/PW/Agentic-RAG-data/"
            "hotpotqa_provenance/substrate"
        ),
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=root / "configs" / "hotpotqa_qwen36_amd_nothink.yaml",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted serial run using each condition's progress.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate all four completed runs and print the exact commands; "
            "do not contact Ollama or create outputs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    root = Path(__file__).resolve().parents[1]
    try:
        plan = build_heldout100_plan(
            repo_root=root,
            pilot_root=args.pilot_root,
            sample_dir=args.sample_dir,
            substrate=args.substrate,
            agent_config=args.agent_config,
            output_root=args.output_root,
            python_executable=sys.executable,
        )
        if args.dry_run:
            print(
                json.dumps(
                    plan.manifest(mode="dry_run", status="validated"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        summary = execute_heldout100_plan(plan, resume=args.resume)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": plan.output_root.as_posix(),
                    "systems": summary["systems"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"PRECONDITION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
