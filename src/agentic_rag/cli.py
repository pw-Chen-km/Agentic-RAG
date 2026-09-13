"""Small CLI for the interface study runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.config import BuildConfig
from agentic_rag.substrate.builder import SubstrateBuilder
from agentic_rag.substrate.retrieval import Retriever
from agentic_rag.substrate.storage import Substrate
from agentic_rag.substrate.validation import validate_substrate


app = typer.Typer(name="agentic-rag", no_args_is_help=True)


def _emit(value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


@app.command("build")
def build(
    config: Annotated[Path, typer.Option("--config")],
    source: Annotated[Path, typer.Option("--source")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    _emit(SubstrateBuilder(BuildConfig.from_yaml(config)).build(source, output))


@app.command("validate")
def validate(substrate: Annotated[Path, typer.Option("--substrate")]) -> None:
    _emit(validate_substrate(substrate))


@app.command("search")
def search(
    substrate: Annotated[Path, typer.Option("--substrate")],
    query: Annotated[str, typer.Option("--query")],
    method: Annotated[str, typer.Option("--method")],
    target: Annotated[str, typer.Option("--target")],
    scope: Annotated[str, typer.Option("--scope")],
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
) -> None:
    _emit([
        item.model_dump(mode="json")
        for item in Retriever(substrate).search(query, method, target, scope, top_k)
    ])


@app.command("read")
def read(
    substrate: Annotated[Path, typer.Option("--substrate")],
    chunk: Annotated[str, typer.Option("--chunk")],
) -> None:
    _emit(Substrate.open(substrate).read_chunk(chunk))


@app.command("run")
def run(
    substrate: Annotated[Path, typer.Option("--substrate")],
    config: Annotated[Path, typer.Option("--config")],
    skill: Annotated[Path, typer.Option("--skill")],
    output: Annotated[Path, typer.Option("--output")],
    question: Annotated[str, typer.Option("--question")],
    scope: Annotated[str, typer.Option("--scope")],
    episode_id: Annotated[str | None, typer.Option("--episode-id")] = None,
) -> None:
    harness = AgentHarness.from_config(substrate, config, skill, output)
    _emit(harness.run(question, scope, episode_id=episode_id))


if __name__ == "__main__":
    app()
