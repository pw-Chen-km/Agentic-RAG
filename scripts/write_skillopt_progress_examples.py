#!/usr/bin/env python3
"""Generate synthetic progress examples without models or benchmark episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.skillopt.offline_examples import write_offline_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory, or byte-identical previously generated examples.")
    args = parser.parse_args()
    summary = write_offline_examples(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "synthetic": True,
                      "case_count": summary["case_count"], "arm_count": summary["arm_count"],
                      "model_calls": summary["model_calls"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
