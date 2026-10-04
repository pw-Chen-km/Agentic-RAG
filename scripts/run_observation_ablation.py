"""Run the four observation representations with identical benchmark inputs.

This wrapper intentionally runs each arm in its own output directory.  The
underlying benchmark runner keeps per-question checkpoints, so an interrupted
arm can be resumed without changing the other arms.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODES = ("raw", "program", "reader", "reader_assessed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    modes = MODES if args.mode == "all" else (args.mode,)
    runner = Path(__file__).with_name("run_benchmark_eval.py")
    for mode in modes:
        output = args.output_root / mode
        command = [
            sys.executable,
            str(runner),
            "--substrate", str(args.substrate),
            "--split", str(args.split),
            "--config", str(args.config_dir / f"hotpotqa_observation_{mode}.yaml"),
            "--skill", str(args.skill),
            "--output", str(output),
            "--expected-count", str(args.expected_count),
            "--dataset", "hotpotqa",
        ]
        if args.resume or (output / "progress.json").exists():
            command.append("--resume")
        print("RUN", mode, flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
