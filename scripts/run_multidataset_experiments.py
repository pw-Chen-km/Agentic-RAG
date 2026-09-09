#!/usr/bin/env python3
"""Plan or explicitly execute the durable, capacity-limited experiment queue."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag.skillopt.experiment_scheduler import main


if __name__ == "__main__":
    raise SystemExit(main())
