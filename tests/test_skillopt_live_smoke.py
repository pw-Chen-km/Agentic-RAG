from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_rag.cli import app


pytestmark = pytest.mark.skillopt_smoke


@pytest.mark.skipif(
    os.getenv("RUN_SKILLOPT_SMOKE") != "1",
    reason="set RUN_SKILLOPT_SMOKE=1 for the paid 18-question workflow",
)
def test_live_hotpotqa_skillopt_workflow(tmp_path: Path) -> None:
    """Opt-in paid smoke against an already prepared corpus and split."""

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required")
    substrate = os.getenv("SKILLOPT_SMOKE_SUBSTRATE")
    split_dir = os.getenv("SKILLOPT_SMOKE_SPLIT_DIR")
    if not substrate or not split_dir:
        pytest.skip(
            "set SKILLOPT_SMOKE_SUBSTRATE and SKILLOPT_SMOKE_SPLIT_DIR"
        )
    project_root = Path(__file__).resolve().parents[1]
    output = Path(
        os.getenv("SKILLOPT_SMOKE_OUTPUT", str(tmp_path / "run"))
    )
    result = CliRunner().invoke(
        app,
        [
            "skillopt-train",
            substrate,
            "--split-dir",
            split_dir,
            "--agent-config",
            str(
                project_root
                / "configs"
                / "agentic_hotpotqa_skillopt_smoke.yaml"
            ),
            "--skillopt-config",
            str(
                project_root / "configs" / "skillopt_hotpotqa_smoke.yaml"
            ),
            "--skill-file",
            str(
                project_root
                / "skills"
                / "hotpotqa_skillopt_initial.md"
            ),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "best_skill.md").is_file()
    workflow = json.loads(
        (output / "workflow_summary.json").read_text(encoding="utf-8")
    )
    assert workflow["run_kind"] == "workflow_smoke"
    assert workflow["paper_parity"] is False
    assert len(list(output.glob("steps/step_*"))) == 2
    assert list(output.rglob("io_trace.json"))
    assert list(output.rglob("evaluation.json"))
    assert list(output.rglob("rollout_result.json"))

    api_key = os.environ["OPENAI_API_KEY"]
    for path in output.rglob("*"):
        if path.is_file():
            assert api_key.encode("utf-8") not in path.read_bytes()
