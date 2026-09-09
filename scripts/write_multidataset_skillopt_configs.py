"""Write immutable legacy or cross-dataset configs; never starts training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.skillopt.multidataset_configs import (
    CROSS_DATASET_DESIGN, LEGACY_DESIGN, MODEL_DIGEST, write_experiment_configs,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--initial-skill", type=Path, required=True)
    parser.add_argument("--optimizer-config", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design", choices=(LEGACY_DESIGN, CROSS_DATASET_DESIGN), default=LEGACY_DESIGN)
    parser.add_argument("--lanes", type=Path, help="JSON file containing an ordered list of lane records")
    parser.add_argument("--model-digest", default=MODEL_DIGEST)
    parser.add_argument("--reflection-minibatch-size", type=int, default=8,
                        help="Cross-dataset reflection trajectories per group (1–40); default 8 preserves existing bundles")
    parser.add_argument("--reflection-token-budget-enabled", action="store_true",
                        help="Explicitly split oversized reflection groups without truncation; requires minibatch size 5")
    args = parser.parse_args(argv)
    lanes = json.loads(args.lanes.read_text(encoding="utf-8")) if args.lanes else None
    report = write_experiment_configs(
        args.prepared_root, args.initial_skill, args.optimizer_config, args.agent_config, args.output,
        design=args.design, lanes=lanes, model_digest=args.model_digest,
        reflection_minibatch_size=args.reflection_minibatch_size,
        reflection_token_budget_enabled=args.reflection_token_budget_enabled,
    )
    print(json.dumps({"execution_manifest": str(args.output.resolve() / "execution_manifest.json"),
                      "experiment_count": len(report["experiments"]),
                      "ready_experiment_count": sum(entry["ready"] for entry in report["experiments"]),
                      "reflection_minibatch_size": report["reflection_minibatch_size"],
                      "reflection_grouping": report.get("reflection_grouping", "fixed"),
                      "datasets": report["datasets"], "model_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
