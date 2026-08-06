from __future__ import annotations

from typer.testing import CliRunner

from agentic_rag.cli import app


def test_cli_exposes_canonical_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "build",
        "validate",
        "search",
        "read",
        "bridge",
        "run",
        "skillopt-prepare",
        "skillopt-train",
    ):
        assert command in result.stdout
