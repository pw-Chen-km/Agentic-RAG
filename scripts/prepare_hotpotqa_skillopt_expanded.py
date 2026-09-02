"""Prepare a larger HotpotQA SkillOpt split while preserving the pilot split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.skillopt.expanded_split import prepare_expanded_hotpotqa_splits
from agentic_rag.skillopt.provenance import prepare_hotpotqa_provenance_splits


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--base-split-dir", type=Path, required=True)
    parser.add_argument("--excluded-questions", type=Path, required=True)
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--validation-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    base_manifest = prepare_expanded_hotpotqa_splits(
        questions_path=args.questions,
        base_split_dir=args.base_split_dir,
        excluded_questions_path=args.excluded_questions,
        output_dir=args.base_output,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    provenance_manifest = prepare_hotpotqa_provenance_splits(
        source_path=(args.substrate.parent / "source" / "hotpotqa.json"),
        base_split_dir=args.base_output,
        substrate_path=args.substrate,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "base_split": base_manifest,
                "provenance_split": provenance_manifest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
