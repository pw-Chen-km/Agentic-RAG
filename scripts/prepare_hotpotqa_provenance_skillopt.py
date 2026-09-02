"""Prepare the fixed 20/6/6 HotpotQA provenance split for SkillOpt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.skillopt.provenance import (
    HOTPOTQA_PROVENANCE_SHA256,
    prepare_hotpotqa_provenance_splits,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-split-dir", type=Path, required=True)
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-source-sha256",
        default=HOTPOTQA_PROVENANCE_SHA256,
        help="Pinned SHA-256 of the exact 1,000-question raw HotpotQA source.",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    manifest = prepare_hotpotqa_provenance_splits(
        source_path=args.source,
        base_split_dir=args.base_split_dir,
        substrate_path=args.substrate,
        output_dir=args.output,
        expected_source_sha256=args.expected_source_sha256,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
