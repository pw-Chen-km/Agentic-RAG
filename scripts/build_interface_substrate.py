"""Build the title-deduplicated global HotpotQA substrate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.config import BuildConfig
from agentic_rag.substrate.builder import SubstrateBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/interface_study/hotpot_dev_distractor_v1.selected.json"))
    parser.add_argument("--output", type=Path, default=Path("data/interface_study/substrate"))
    parser.add_argument("--config", type=Path, default=Path("configs/hotpotqa_global_provenance.yaml"))
    args = parser.parse_args()
    config = BuildConfig.from_yaml(args.config)
    manifest = SubstrateBuilder(config).build(args.source, args.output)
    print(json.dumps({"corpus_id": manifest.corpus_id, "records": manifest.record_counts}, ensure_ascii=True))


if __name__ == "__main__":
    main()
