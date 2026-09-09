"""Dry-run by default; final tests require the bundle's sealed training runs.

Legacy bundles require four completions; cross-dataset bundles require all twelve.

After successful training, seal each arm without model calls:
  python scripts/run_prepared_skillopt_test.py --manifest ... --dataset ... \
      --arm organized --seal-completion

Inspect the final test (no model calls):
  python scripts/run_prepared_skillopt_test.py --manifest ... --dataset ... --arm initial

Only an explicit --execute runs Target Agent and the existing answer evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.skillopt.multidataset_configs import (
    DATASETS,
    REPRESENTATIONS,
    aggregate_final_test_results,
    run_prepared_final_test,
    seal_training_completion,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--arm", choices=("initial", *REPRESENTATIONS))
    parser.add_argument("--source-dataset", choices=("hotpotqa", "medical", "novel"),
                        help="For cross-dataset testing, the dataset used to train this Skill")
    parser.add_argument("--lane-id", help="Pinned runtime lane, selected by the resource-aware scheduler")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: validate only, no models or output writes")
    mode.add_argument("--execute", action="store_true", help="FINAL phase: requires all training completions specified by the bundle")
    mode.add_argument("--seal-completion", action="store_true", help="Offline: verify completed training artifacts and bind best_skill.md")
    mode.add_argument("--aggregate", action="store_true", help="Offline: compare saved binary results only after all 65 final tests complete")
    args = parser.parse_args(argv)
    if args.aggregate:
        if any((args.dataset, args.arm, args.source_dataset, args.lane_id)):
            parser.error("--aggregate compares the entire bundle and accepts no per-task selectors")
        print(json.dumps(aggregate_final_test_results(args.manifest), ensure_ascii=False, allow_nan=False,
                         sort_keys=True, indent=2))
        return
    if args.dataset is None or args.arm is None:
        parser.error("--dataset and --arm are required except with --aggregate")
    if args.seal_completion and (args.source_dataset is not None or args.lane_id is not None):
        parser.error("--seal-completion uses --dataset as source; lane is verified from the startup contract")
    result = (seal_training_completion(args.manifest, args.dataset, args.arm) if args.seal_completion
              else run_prepared_final_test(args.manifest, args.dataset, args.arm, execute=args.execute,
                                           source_dataset=args.source_dataset, lane_id=args.lane_id))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
