"""Probe existing AMD lanes; defaults to no network/model calls.

--execute is a real capacity experiment. It never changes a service, creates a
tunnel, unloads a foreign model, or reads validation/test examples. The caller
must authorize each GPU explicitly; default authorization is GPU1 only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.skillopt.amd_capacity import run_capacity_test, ssh_snapshot


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allowed-gpu", action="append", type=int, dest="gpus")
    parser.add_argument("--ssh-host", default="root@140.116.240.181")
    parser.add_argument("--ssh-port", type=int, default=45026)
    parser.add_argument("--ssh-key", type=Path, default=Path("/home/jj/.ssh/amd_root_key"))
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--sample-interval", type=float, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("Use a new output file; existing capacity evidence is never overwritten")
    prefix = ["ssh", "-i", str(args.ssh_key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
              "-o", "StrictHostKeyChecking=yes", "-p", str(args.ssh_port), args.ssh_host]
    report = run_capacity_test(args.manifest, args.workload, execute=args.execute,
                              allowed_gpu_ids=args.gpus if args.gpus is not None else [1],
                              snapshot_provider=lambda: ssh_snapshot(prefix),
                              timeout_seconds=args.timeout_seconds, sample_interval=args.sample_interval)
    text = json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
