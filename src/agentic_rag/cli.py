"""Command-line interface for building and inspecting a substrate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from agentic_rag.agent.answer import AnswerGenerationError
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.policy import PolicyError
from agentic_rag.bridge import SubstrateBridge
from agentic_rag.builder import SubstrateBuilder
from agentic_rag.config import BuildConfig
from agentic_rag.errors import AgenticRAGError
from agentic_rag.retrieval import Retriever
from agentic_rag.storage import Substrate
from agentic_rag.validation import validate_substrate

app = typer.Typer(
    name="agentic-rag",
    help="Build a typed substrate and run query-time Agentic RAG.",
    no_args_is_help=True,
)


def _emit(value: object) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _fail(exc: AgenticRAGError) -> None:
    typer.echo(
        json.dumps(
            {"error": exc.code, "message": exc.message},
            ensure_ascii=False,
            sort_keys=True,
        ),
        err=True,
    )
    raise typer.Exit(code=2)


def _fail_typed(code: str, message: str) -> None:
    typer.echo(
        json.dumps(
            {"error": code, "message": message},
            ensure_ascii=False,
            sort_keys=True,
        ),
        err=True,
    )
    raise typer.Exit(code=2)


@app.command("build")
def build_command(
    source: Annotated[
        Path, typer.Argument(exists=True, readable=True)
    ],
    output: Annotated[Path, typer.Argument(file_okay=False)],
    config_path: Annotated[
        Path | None,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = None,
    corpus_id: Annotated[str | None, typer.Option("--corpus-id")] = None,
    split: Annotated[str | None, typer.Option("--split")] = None,
    source_format: Annotated[
        str | None, typer.Option("--source-format")
    ] = None,
    benchmark_scope_id: Annotated[
        str | None, typer.Option("--benchmark-scope-id")
    ] = None,
    max_chunk_tokens: Annotated[
        int | None, typer.Option("--max-chunk-tokens", min=1)
    ] = None,
    spacy_model: Annotated[str | None, typer.Option("--spacy-model")] = None,
    embedding_model: Annotated[
        str | None, typer.Option("--embedding-model")
    ] = None,
    embedding_device: Annotated[
        str | None, typer.Option("--embedding-device")
    ] = None,
    disable_abbreviations: Annotated[
        bool,
        typer.Option(
            "--disable-abbreviations",
            help="Disable scispaCy AbbreviationDetector.",
        ),
    ] = False,
) -> None:
    try:
        if config_path is not None:
            config = BuildConfig.from_yaml(config_path)
            updates = {
                key: value
                for key, value in {
                    "corpus_id": corpus_id,
                    "split": split,
                    "source_format": source_format,
                    "benchmark_scope_id": benchmark_scope_id,
                    "max_chunk_tokens": max_chunk_tokens,
                    "spacy_model": spacy_model,
                    "embedding_model": embedding_model,
                    "embedding_device": embedding_device,
                }.items()
                if value is not None
            }
            if disable_abbreviations:
                updates["enable_abbreviations"] = False
            config = config.model_copy(update=updates)
        else:
            if corpus_id is None:
                raise AgenticRAGError(
                    "--corpus-id is required when --config is not provided"
                )
            config = BuildConfig(
                corpus_id=corpus_id,
                split=split or "dev",
                source_format=source_format or "hotpotqa_scoped",
                benchmark_scope_id=benchmark_scope_id,
                max_chunk_tokens=max_chunk_tokens or 256,
                spacy_model=spacy_model or "en_core_web_sm",
                embedding_model=embedding_model
                or "sentence-transformers/all-MiniLM-L6-v2",
                embedding_device=embedding_device,
                enable_abbreviations=not disable_abbreviations,
            )
        manifest = SubstrateBuilder(config).build(source, output)
        _emit(manifest)
    except AgenticRAGError as exc:
        _fail(exc)


@app.command("validate")
def validate_command(
    substrate_path: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, readable=True)
    ],
) -> None:
    try:
        report = validate_substrate(substrate_path)
        _emit(report)
        if not report.valid:
            raise typer.Exit(code=1)
    except AgenticRAGError as exc:
        _fail(exc)


@app.command("search")
def search_command(
    substrate_path: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, readable=True)
    ],
    query: Annotated[str, typer.Argument()],
    scope_id: Annotated[str, typer.Option("--scope-id")],
    method: Annotated[str, typer.Option("--method")],
    target: Annotated[str, typer.Option("--target")],
    top_k: Annotated[int, typer.Option("--top-k", min=1)] = 5,
) -> None:
    try:
        hits = Retriever(substrate_path).search(
            query=query,
            method=method,
            target=target,
            scope_id=scope_id,
            top_k=top_k,
        )
        _emit([hit.model_dump(mode="json") for hit in hits])
    except AgenticRAGError as exc:
        _fail(exc)


@app.command("read")
def read_command(
    substrate_path: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, readable=True)
    ],
    chunk_id: Annotated[str, typer.Argument()],
) -> None:
    try:
        _emit(Substrate.open(substrate_path).read_chunk(chunk_id))
    except AgenticRAGError as exc:
        _fail(exc)


@app.command("bridge")
def bridge_command(
    substrate_path: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, readable=True)
    ],
    entity_id: Annotated[str, typer.Argument()],
    scope_id: Annotated[str, typer.Option("--scope-id")],
    top_k: Annotated[int | None, typer.Option("--top-k", min=1)] = None,
) -> None:
    try:
        hits = SubstrateBridge(substrate_path).entity_sentence_entity(
            entity_id,
            scope_id,
            top_k=top_k,
        )
        _emit([hit.model_dump(mode="json") for hit in hits])
    except AgenticRAGError as exc:
        _fail(exc)


@app.command("run")
def run_command(
    substrate_path: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, readable=True)
    ],
    question: Annotated[str, typer.Argument()],
    scope_id: Annotated[str, typer.Option("--scope-id")],
    skill_file: Annotated[
        Path,
        typer.Option(
            "--skill-file",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", file_okay=False)
    ] = Path("runs"),
    episode_id: Annotated[
        str | None, typer.Option("--episode-id")
    ] = None,
) -> None:
    """Run one scoped agent episode and persist its complete trajectory."""

    try:
        harness = AgentHarness.from_config(
            substrate_path=substrate_path,
            config=config_path,
            skill_file=skill_file,
            output_root=output,
        )
        result = harness.run(
            question, scope_id, episode_id=episode_id
        )
        _emit(result)
        if result.termination_reason.value != "finish":
            raise typer.Exit(code=1)
    except AgenticRAGError as exc:
        _fail(exc)
    except PolicyError as exc:
        _fail_typed("policy_error", str(exc))
    except AnswerGenerationError as exc:
        _fail_typed("answer_generation_error", str(exc))
    except ValidationError as exc:
        _fail_typed("agent_configuration_error", str(exc))
    except (OSError, ValueError) as exc:
        _fail_typed("agent_run_error", str(exc))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
