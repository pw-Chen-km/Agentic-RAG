#!/usr/bin/env python3
"""Prepare full-pool SkillOpt splits without calling models or rebuilding indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.skillopt.multidataset_prepare import prepare_multidataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_multidataset(args.spec, args.output)
    print(json.dumps({"output": args.output.resolve().as_posix(),
                      "all_four_arms_ready": report["all_four_arms_ready"],
                      "datasets": {name: {key: item.get(key) for key in (
                          "split_status", "eligible_count", "excluded_count", "reference_ready_count",
                          "mapping_issue_count", "all_four_arms_ready", "error", "split_error")}
                                   for name, item in report["datasets"].items()}},
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["all_four_arms_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
