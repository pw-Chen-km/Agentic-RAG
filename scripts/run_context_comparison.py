"""Sequential, detached-friendly 21+21 context/assessment smoke comparison."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name("run_jj_v2_smoke.py")
    status = {"status": "running", "stages": {}}
    for stage, mode in (("stage1", "off"), ("stage2", "on")):
        status["active_stage"] = stage
        (a.output / "comparison_status.json").write_text(json.dumps(status, indent=2))
        cmd = [sys.executable, "-u", str(script), "--data-root", str(a.data_root),
               "--output", str(a.output / stage), "--assessment", mode]
        with (a.output / f"{stage}.launch.log").open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        status["stages"][stage] = {"returncode": result.returncode}
        if result.returncode:
            status["status"] = "error"
            break
    else:
        status["status"] = "completed_requires_audit"
    (a.output / "comparison_status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
