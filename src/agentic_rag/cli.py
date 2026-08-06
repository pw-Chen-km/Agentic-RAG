"""Command-line interface for building and inspecting a substrate."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Iterator

import typer
from pydantic import ValidationError

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.policy import PolicyError
from agentic_rag.substrate.bridge import SubstrateBridge
from agentic_rag.substrate.builder import SubstrateBuilder
from agentic_rag.evaluation.profiles import get_dataset_profile
from agentic_rag.config import BuildConfig
from agentic_rag.errors import AgenticRAGError
from agentic_rag.evaluation import (
    EpisodeEvaluator,
    EvaluationError,
    OpenAIResponsesJudge,
)
from agentic_rag.substrate.retrieval import Retriever
from agentic_rag.skillopt.adapter import AgenticRAGSkillOptAdapter
from agentic_rag.substrate.storage import Substrate
from agentic_rag.skillopt.data import (
    HOTPOTQA_BENCHMARK_SCOPE_ID,
    prepare_hotpotqa_smoke_splits,
    split_manifest_profile,
    validate_hotpotqa_smoke_lineage,
)
from agentic_rag.skillopt.trainer import (
    SkillOptUnavailableError,
    load_skillopt_config,
    run_skillopt_training,
)
from agentic_rag.substrate.validation import validate_substrate

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


@contextmanager
def _skillopt_openai_environment(
    *, endpoint: str, auth_mode: str
) -> Iterator[None]:
    """Bridge the target OPENAI_API_KEY into SkillOpt without persisting it."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SkillOptUnavailableError(
            "OPENAI_API_KEY is required for the SkillOpt workflow"
        )
    updates = {
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_AUTH_MODE": auth_mode,
        "AZURE_OPENAI_API_KEY": api_key,
    }
    provider_prefixes = (
        "AZURE_OPENAI_",
        "OPTIMIZER_AZURE_OPENAI_",
        "TARGET_AZURE_OPENAI_",
    )

    def is_provider_key(key: str) -> bool:
        return key.startswith(provider_prefixes) or key in {
            "OPTIMIZER_DEPLOYMENT",
            "TARGET_DEPLOYMENT",
        }

    # SkillOpt derives role-specific variables while configuring its model
    # backend.  Snapshot the whole provider namespace so those derived API-key
    # variables cannot escape this command's context.
    previous = {
        key: value
        for key, value in os.environ.items()
        if is_provider_key(key)
    }
    provider_module = sys.modules.get("skillopt.model.azure_openai")
    module_state: dict[str, object] | None = None
    module_state_names = (
        "ENDPOINT",
        "API_VERSION",
        "API_KEY",
        "AUTH_MODE",
        "AD_SCOPE",
        "MANAGED_IDENTITY_CLIENT_ID",
        "OPTIMIZER_ENDPOINT",
        "OPTIMIZER_API_VERSION",
        "OPTIMIZER_API_KEY",
        "OPTIMIZER_AUTH_MODE",
        "OPTIMIZER_AD_SCOPE",
        "OPTIMIZER_MANAGED_IDENTITY_CLIENT_ID",
        "TARGET_ENDPOINT",
        "TARGET_API_VERSION",
        "TARGET_API_KEY",
        "TARGET_AUTH_MODE",
        "TARGET_AD_SCOPE",
        "TARGET_MANAGED_IDENTITY_CLIENT_ID",
        "OPTIMIZER_DEPLOYMENT",
        "TARGET_DEPLOYMENT",
        "_optimizer_client",
        "_target_client",
        "_AZ_CLI_TOKEN_CACHE",
    )
    if provider_module is not None:
        module_state = {}
        for name in module_state_names:
            if hasattr(provider_module, name):
                value = getattr(provider_module, name)
                module_state[name] = (
                    dict(value) if isinstance(value, dict) else value
                )
    os.environ.update(updates)
    try:
        yield
    finally:
        for key in tuple(os.environ):
            if is_provider_key(key):
                os.environ.pop(key, None)
        os.environ.update(previous)

        # configure_azure_openai also stores credentials and clients in module
        # globals.  Restore a pre-existing module exactly; otherwise scrub the
        # key-bearing globals and caches created during this command.
        active_module = sys.modules.get("skillopt.model.azure_openai")
        if active_module is not None:
            if module_state is not None and active_module is provider_module:
                for name, value in module_state.items():
                    setattr(active_module, name, value)
            else:
                shared_key = previous.get("AZURE_OPENAI_API_KEY", "")
                optimizer_key = (
                    previous.get("OPTIMIZER_AZURE_OPENAI_API_KEY")
                    or previous.get("AZURE_OPENAI_OPTIMIZER_API_KEY")
                    or shared_key
                )
                target_key = (
                    previous.get("TARGET_AZURE_OPENAI_API_KEY")
                    or previous.get("AZURE_OPENAI_TARGET_API_KEY")
                    or shared_key
                )
                for name, value in {
                    "API_KEY": shared_key,
                    "OPTIMIZER_API_KEY": optimizer_key,
                    "TARGET_API_KEY": target_key,
                    "_optimizer_client": None,
                    "_target_client": None,
                }.items():
                    if hasattr(active_module, name):
                        setattr(active_module, name, value)
                token_cache = getattr(
                    active_module, "_AZ_CLI_TOKEN_CACHE", None
                )
                if isinstance(token_cache, dict):
                    token_cache.clear()


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
    dataset: Annotated[str | None, typer.Option("--dataset")] = None,
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
    validate_benchmark_profile: Annotated[
        bool,
        typer.Option(
            "--validate-benchmark-profile",
            help="Require the pinned HotpotQA reference counts.",
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
                    "dataset": dataset,
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
            if validate_benchmark_profile:
                updates["validate_benchmark_profile"] = True
            config = BuildConfig.model_validate(
                {**config.model_dump(), **updates}
            )
        else:
            if corpus_id is None:
                raise AgenticRAGError(
                    "--corpus-id is required when --config is not provided"
                )
            config = BuildConfig(
                corpus_id=corpus_id,
                split=split or "dev",
                dataset=dataset or "hotpotqa",
                source_format=source_format or "hotpotqa_scoped",
                benchmark_scope_id=benchmark_scope_id,
                validate_benchmark_profile=validate_benchmark_profile,
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
            help="Skill Markdown file loaded into every Policy context.",
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
    except ValidationError as exc:
        _fail_typed("agent_configuration_error", str(exc))
    except (OSError, ValueError) as exc:
        _fail_typed("agent_run_error", str(exc))


@app.command("skillopt-prepare")
def skillopt_prepare_command(
    dataset_dir: Annotated[
        Path,
        typer.Option(
            "--dataset-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help=(
                "Pinned Ayanami0730/rag_test checkout, or one subset "
                "directory containing chunks.json and questions.json."
            ),
        ),
    ],
    split_dir: Annotated[
        Path,
        typer.Option(
            "--split-dir",
            file_okay=False,
            help="Destination for deterministic train/validation/test files.",
        ),
    ],
    dataset: Annotated[
        str,
        typer.Option(
            "--dataset",
            help="Dataset profile; only hotpotqa is supported.",
        ),
    ] = "hotpotqa",
    seed: Annotated[int, typer.Option("--seed")] = 42,
    split_size: Annotated[
        int, typer.Option("--split-size", min=1)
    ] = 6,
    train_size: Annotated[
        int | None,
        typer.Option(
            "--train-size",
            min=6,
            help=(
                "HotpotQA-only training size. Keeps validation/test at "
                "the fixed six-item evaluation contract."
            ),
        ),
    ] = None,
    allow_subset: Annotated[
        bool,
        typer.Option(
            "--allow-subset",
            help="Skip official reference-count validation for local fixtures.",
        ),
    ] = False,
) -> None:
    """Prepare deterministic HotpotQA SkillOpt splits."""

    try:
        profile = get_dataset_profile(dataset)
        if seed != 42 or split_size != 6:
            raise ValueError("the canonical HotpotQA split uses seed 42 and size 6")
        if train_size is not None and allow_subset:
            raise ValueError(
                "--train-size cannot be combined with --allow-subset"
            )
        prepare_kwargs = {
            "dataset_dir": dataset_dir,
            "split_dir": split_dir,
            "expected_question_count": None if allow_subset else profile.reference_question_count,
            "expected_chunk_count": None if allow_subset else profile.reference_chunk_count,
        }
        if train_size is not None:
            prepare_kwargs["train_size"] = train_size
        manifest = prepare_hotpotqa_smoke_splits(**prepare_kwargs)
        _emit(manifest)
    except AgenticRAGError as exc:
        _fail(exc)
    except (OSError, ValueError) as exc:
        _fail_typed("skillopt_prepare_error", str(exc))


@app.command("skillopt-train")
def skillopt_train_command(
    substrate_path: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    split_dir: Annotated[
        Path,
        typer.Option(
            "--split-dir",
            exists=True,
            file_okay=False,
            readable=True,
        ),
    ],
    agent_config_path: Annotated[
        Path,
        typer.Option(
            "--agent-config",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    skillopt_config_path: Annotated[
        Path,
        typer.Option(
            "--skillopt-config",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    skill_file: Annotated[
        Path,
        typer.Option(
            "--skill-file",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", file_okay=False),
    ] = Path("runs/skillopt_hotpotqa"),
) -> None:
    """Run the native SkillOpt v0.2.0 workflow for HotpotQA."""

    try:
        substrate = Substrate.open(substrate_path)
        profile = split_manifest_profile(split_dir)
        split_manifest = json.loads(
            (split_dir / "split_manifest.json").read_text(encoding="utf-8")
        )
        scope_id = HOTPOTQA_BENCHMARK_SCOPE_ID
        substrate.require_scope(scope_id)
        if substrate.manifest.source_format != "hotpotqa_benchmark_exact":
            raise ValueError(
                "HotpotQA SkillOpt requires a hotpotqa_benchmark_exact substrate"
            )
        validate_hotpotqa_smoke_lineage(split_dir, substrate.manifest)
        split_metadata = split_manifest.get("splits")
        if not isinstance(split_metadata, dict):
            raise ValueError("split manifest has no splits mapping")

        def split_count(split_name: str) -> int:
            metadata = split_metadata.get(split_name)
            count = metadata.get("count") if isinstance(metadata, dict) else None
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 1
            ):
                raise ValueError(
                    f"split manifest {split_name} count must be positive"
                )
            return count

        workflow_train_size = split_count("train")
        workflow_validation_size = split_count("validation")
        workflow_test_size = split_count("test")

        agent_config = AgentConfig.from_yaml(agent_config_path)
        if (
            agent_config.policy.provider != "openai"
            or agent_config.policy.model != "gpt-5.6-luna"
        ):
            raise ValueError(
                "SkillOpt workflow requires OpenAI gpt-5.6-luna Policy"
            )
        if not skill_file.read_text(encoding="utf-8").strip():
            raise ValueError("initial SkillOpt Markdown must not be blank")
        skillopt_config = load_skillopt_config(skillopt_config_path)
        output = output.resolve()
        skillopt_config.update(
            {
                "out_root": str(output),
                "split_dir": str(split_dir.resolve()),
                "skill_init": str(skill_file.resolve()),
                "dataset": profile.key,
            }
        )
        required = (
            "optimizer_model",
            "target_model",
            "batch_size",
            "num_epochs",
            "accumulation",
            "seed",
            "merge_batch_size",
            "edit_budget",
            "analyst_workers",
            "sel_env_num",
            "test_env_num",
            "eval_test",
        )
        missing = [key for key in required if key not in skillopt_config]
        if missing:
            raise ValueError(
                "SkillOpt config is missing required flattened keys: "
                + ", ".join(missing)
            )
        expected_smoke_values = {
            "optimizer_model": "gpt-5.6-luna",
            "target_model": "gpt-5.6-luna",
            "num_epochs": 1,
            "train_size": workflow_train_size,
            "batch_size": 3,
            "accumulation": 1,
            "seed": 42,
            "minibatch_size": 3,
            "merge_batch_size": 3,
            "analyst_workers": 1,
            "max_analyst_rounds": 1,
            "failure_only": False,
            "edit_budget": 1,
            "min_edit_budget": 1,
            "lr_scheduler": "constant",
            "skill_update_mode": "patch",
            "use_slow_update": False,
            "use_meta_skill": False,
            "use_gate": True,
            "sel_env_num": workflow_validation_size,
            "test_env_num": workflow_test_size,
            "eval_test": True,
            "workers": 1,
            "judge_model": "gpt-5.6-luna",
        }
        mismatches = {
            key: {
                "expected": expected,
                "actual": skillopt_config.get(key),
            }
            for key, expected in expected_smoke_values.items()
            if skillopt_config.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "SkillOpt workflow-smoke config mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )

        def harness_factory(
            *, skill_content: str, output_root: Path
        ) -> AgentHarness:
            return AgentHarness.from_skill_content(
                substrate_path=substrate_path,
                config=agent_config,
                skill_content=skill_content,
                skill_source_path="skillopt:candidate",
                output_root=output_root,
            )

        judge_model = str(
            skillopt_config.get("judge_model") or "gpt-5.6-luna"
        )
        evaluator = EpisodeEvaluator(
            OpenAIResponsesJudge(model=judge_model),
            profile=profile,
        )
        adapter = AgenticRAGSkillOptAdapter(
            split_dir=split_dir,
            harness_factory=harness_factory,
            evaluator=evaluator,
            dataset=profile,
            workers=int(skillopt_config.get("workers", 1)),
            analyst_workers=int(skillopt_config["analyst_workers"]),
            failure_only=bool(
                skillopt_config.get("failure_only", False)
            ),
            minibatch_size=int(
                skillopt_config.get("minibatch_size", 3)
            ),
            edit_budget=int(skillopt_config["edit_budget"]),
            seed=int(skillopt_config["seed"]),
        )
        endpoint = str(
            skillopt_config.get("azure_openai_endpoint")
            or "https://api.openai.com/v1"
        )
        auth_mode = str(
            skillopt_config.get("azure_openai_auth_mode")
            or "openai_compatible"
        )
        with _skillopt_openai_environment(
            endpoint=endpoint,
            auth_mode=auth_mode,
        ):
            summary = run_skillopt_training(skillopt_config, adapter)
        _emit(
            {
                "run_kind": "workflow_smoke",
                "paper_parity": False,
                "dataset": profile.key,
                "scope_id": scope_id,
                "metric_contract": {
                    "reported": [
                        metric.value for metric in profile.reported_metrics
                    ],
                    "hard": profile.skillopt_hard_metric.value,
                    "soft": profile.skillopt_soft_metric.value,
                },
                "output": str(output),
                "summary": summary,
            }
        )
    except AgenticRAGError as exc:
        _fail(exc)
    except SkillOptUnavailableError as exc:
        _fail_typed("skillopt_unavailable", str(exc))
    except EvaluationError as exc:
        _fail_typed(exc.code, str(exc))
    except ValidationError as exc:
        _fail_typed("skillopt_configuration_error", str(exc))
    except (ImportError, OSError, TypeError, ValueError) as exc:
        _fail_typed("skillopt_training_error", str(exc))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
