"""Recompute configuration summaries from immutable pilot progress."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from agentic_rag.evaluation.interface_study import aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/interface-study-v1"))
    args = parser.parse_args()
    progress = args.run / "progress.jsonl"
    rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("condition", "unknown")].append(row)
    summary = {
        "run_manifest": json.loads((args.run / "run_manifest.json").read_text(encoding="utf-8")),
        "conditions": {name: aggregate(values) for name, values in sorted(groups.items())},
        "episode_count": len(rows),
    }
    (args.run / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["conditions"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
