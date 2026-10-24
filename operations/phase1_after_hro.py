"""Wait for the HRO wrapper receipt before launching a Phase1-only control.

This operational helper lives outside the experiment's source-hash directories;
repairing queue coordination does not change the running optimizer or Agent.
"""

import argparse
import fcntl
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def upstream_state(upstream):
    """The wrapper puts run.exit beside results/, not in its grandparent."""
    upstream = Path(upstream)
    exit_file = upstream.parent / "run.exit"
    checkpoint_path = upstream / "checkpoint.json"
    exit_text = exit_file.read_text().strip() if exit_file.exists() else None
    details = {"upstream_exit_file": str(exit_file), "upstream_exit": exit_text}
    if exit_text is not None and exit_text != "EXIT=0":
        return "blocked_upstream_failure", details
    try:
        checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
    except (ValueError, OSError) as exc:
        return "upstream_checkpoint_invalid", {**details, "error": str(exc)}
    if (checkpoint and checkpoint.get("stage") == "complete"
            and checkpoint.get("status") == "complete" and exit_text == "EXIT=0"):
        return "ready", details
    return "waiting_for_hro_complete", details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True, help="HRO results directory")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="Phase1-only run root")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    metadata = {"upstream_output": str(args.upstream), "config": str(args.config),
                "output": str(args.root / "results")}

    def write(status, **extra):
        payload = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(),
                   **metadata, **extra}
        (args.root / "queue.status.json").write_text(json.dumps(payload, indent=2) + "\n")

    with (args.root / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A restarted waiting queue must never overwrite a previously launched run.
        if ((args.root / "run.exit").exists()
                or any((args.root / "results").glob("*"))):
            raise RuntimeError("Phase1-only output already exists; inspect before resuming")
        while True:
            status, details = upstream_state(args.upstream)
            if status == "ready":
                break
            write(status, **details)
            if status != "waiting_for_hro_complete":
                return 2
            time.sleep(60)
        write("running", **details)
        with (args.root / "phase1_only.log").open("w") as log:
            proc = subprocess.Popen(
                [str(args.repo / ".venv/bin/python"), str(args.repo / "scripts/run_phase1.py"),
                 "--config", str(args.config), "--execute"], cwd=args.repo,
                stdout=log, stderr=subprocess.STDOUT,
            )
            rc = proc.wait()
        (args.root / "run.exit").write_text(f"EXIT={rc}\n")
        write("complete" if rc == 0 else "failed", exit_code=rc)
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
