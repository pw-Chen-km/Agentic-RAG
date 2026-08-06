"""Lazy SkillOpt v0.2.0 trainer orchestration.

Nothing in this module imports SkillOpt until training is explicitly requested,
so ordinary substrate and query-time commands keep their existing dependency
surface.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agentic_rag.skillopt.adapter import AgenticRAGSkillOptAdapter
from agentic_rag.skillopt.rollout import _write_json_once


class SkillOptUnavailableError(RuntimeError):
    """The optional pinned SkillOpt runtime is not installed."""


_PINNED_PROMPT_NAMES = (
    "analyst_error",
    "analyst_success",
    "merge_failure",
    "merge_success",
    "merge_final",
    "ranking",
)
_SKILLOPT_SOURCE_COMMIT = "e4ea6a6771e797ef820cdd8bfea64c57e0481065"


def load_skillopt_config(path: str | Path) -> dict[str, Any]:
    """Load and flatten a SkillOpt YAML using SkillOpt's own config parser."""

    try:
        from skillopt.config import flatten_config, is_structured, load_config
    except ImportError as exc:
        raise SkillOptUnavailableError(
            "SkillOpt v0.2.0 is required for training; install the project's "
            "SkillOpt optional dependency group"
        ) from exc
    config = load_config(str(path))
    return flatten_config(config) if is_structured(config) else dict(config)


def create_skillopt_trainer(
    config: Mapping[str, Any],
    adapter: AgenticRAGSkillOptAdapter,
    *,
    trainer_cls: Callable[[dict[str, Any], Any], Any] | None = None,
) -> Any:
    """Construct ``ReflACTTrainer`` without using SkillOpt's static registry."""

    if trainer_cls is None:
        try:
            from skillopt.engine.trainer import ReflACTTrainer
        except ImportError as exc:
            raise SkillOptUnavailableError(
                "SkillOpt v0.2.0 is required for training; install the "
                "project's SkillOpt optional dependency group"
            ) from exc
        trainer_cls = ReflACTTrainer
    return trainer_cls(dict(config), adapter)


def run_skillopt_training(
    config: Mapping[str, Any],
    adapter: AgenticRAGSkillOptAdapter,
    *,
    trainer_cls: Callable[[dict[str, Any], Any], Any] | None = None,
) -> dict[str, Any]:
    """Run one SkillOpt workflow and add inspectable workflow-level metadata."""

    cfg = dict(config)
    out_root_value = cfg.get("out_root")
    if not isinstance(out_root_value, str) or not out_root_value.strip():
        raise ValueError("flattened SkillOpt config requires a non-empty out_root")
    out_root = Path(out_root_value)
    out_root.mkdir(parents=True, exist_ok=True)
    _copy_split_manifest(adapter, out_root)

    trainer = create_skillopt_trainer(
        cfg,
        adapter,
        trainer_cls=trainer_cls,
    )
    with _use_pinned_skillopt_prompts(out_root):
        raw_summary = trainer.train()
    if not isinstance(raw_summary, Mapping):
        raise TypeError("SkillOpt trainer.train() must return a summary mapping")
    summary = dict(raw_summary)
    target_and_judge_usage = summarize_rollout_usage(out_root)
    optimizer_usage = summary.get("token_summary", {})
    profile = adapter.profile
    hard_metric = profile.skillopt_hard_metric.value
    soft_metric = profile.skillopt_soft_metric.value
    hard_key = f"hard_{hard_metric}"
    soft_key = f"soft_{soft_metric}"
    initial_metrics = {
        "selection": {
            hard_key: summary.get("baseline_selection_hard"),
        },
        "test": {
            hard_key: summary.get("baseline_test_hard"),
            soft_key: summary.get("baseline_test_soft"),
        },
    }
    final_metrics = {
        "selection": {
            f"best_{hard_key}": summary.get("best_selection_hard"),
            f"final_{hard_key}": summary.get("final_selection_hard"),
            f"final_{soft_key}": summary.get("final_selection_soft"),
        },
        "test": {
            f"best_{hard_key}": summary.get("test_hard"),
            f"best_{soft_key}": summary.get("test_soft"),
            f"final_{hard_key}": summary.get("final_test_hard"),
            f"final_{soft_key}": summary.get("final_test_soft"),
        },
        "best_step": summary.get("best_step"),
    }
    token_summary = {
        "target_and_judge": target_and_judge_usage,
        "optimizer": optimizer_usage,
    }
    workflow_summary = {
        "run_kind": "workflow_smoke",
        "paper_parity": False,
        "dataset": profile.key,
        "metric_contract": {
            "reported": [metric.value for metric in profile.reported_metrics],
            "hard": hard_metric,
            "soft": soft_metric,
        },
        "unique_question_count": sum(
            len(adapter.dataloader.get_split_items(split))
            for split in ("train", "val", "test")
        ),
        "episode_execution_count": target_and_judge_usage["episodes"],
        "trainer_summary": summary,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "target_and_judge_usage": target_and_judge_usage,
        # SkillOpt's native token tracker records reflection/merge/update
        # stages. Preserve it here beside our Policy/Judge counters
        # so one file contains the complete stage-level cost ledger.
        "optimizer_usage": optimizer_usage,
    }
    _write_json_once(out_root / "initial_metrics.json", initial_metrics)
    _write_json_once(out_root / "final_metrics.json", final_metrics)
    _write_json_once(out_root / "token_summary.json", token_summary)
    _write_json_once(out_root / "workflow_summary.json", workflow_summary)
    return summary


def summarize_rollout_usage(out_root: str | Path) -> dict[str, Any]:
    """Aggregate exact target/judge counters from persisted per-task results."""

    totals = {
        "episodes": 0,
        "policy": {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        },
        "judge": {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        },
        "retrieved_tokens": 0,
    }
    for path in sorted(Path(out_root).rglob("rollout_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(row, dict):
            continue
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        judge = (
            row.get("judge_usage")
            if isinstance(row.get("judge_usage"), dict)
            else {}
        )
        totals["episodes"] += 1
        totals["policy"]["calls"] += int(usage.get("policy_calls", 0))
        totals["policy"]["input_tokens"] += int(
            usage.get("input_tokens", 0)
        )
        totals["policy"]["output_tokens"] += int(
            usage.get("output_tokens", 0)
        )
        totals["policy"]["reasoning_tokens"] += int(
            usage.get("reasoning_tokens", 0)
        )
        totals["policy"]["total_tokens"] += int(
            usage.get("total_tokens", 0)
        )
        totals["retrieved_tokens"] += int(usage.get("retrieved_tokens", 0))
        for key in (
            "calls",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        ):
            totals["judge"][key] += int(judge.get(key, 0))
    return totals


def _copy_split_manifest(
    adapter: AgenticRAGSkillOptAdapter, out_root: Path
) -> None:
    source = adapter.dataloader.split_dir / "split_manifest.json"
    if not source.is_file():
        return
    destination = out_root / "split_manifest.json"
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise FileExistsError(
                f"run root contains a different split manifest: {destination}"
            )
        return
    shutil.copyfile(source, destination)


@contextmanager
def _use_pinned_skillopt_prompts(out_root: Path) -> Iterator[None]:
    """Work around prompt files omitted from the SkillOpt 0.2.0 wheel.

    The native prompt loader and all native reflection/update stages remain in
    use. Only its generic prompt directory is temporarily redirected to the
    verbatim Markdown bundle from the matching Microsoft release tag.
    """

    try:
        import skillopt.prompts as skillopt_prompts
    except ImportError:
        # Deterministic fake-trainer CI intentionally works without installing
        # the optional SkillOpt runtime.
        yield
        return

    prompt_dir = Path(__file__).with_name("prompts")
    prompt_files = {
        name: prompt_dir / f"{name}.md" for name in _PINNED_PROMPT_NAMES
    }
    missing = [str(path) for path in prompt_files.values() if not path.is_file()]
    if missing:
        raise SkillOptUnavailableError(
            "Pinned SkillOpt prompt bundle is incomplete: " + ", ".join(missing)
        )
    prompt_metadata = {
        "source": "microsoft/SkillOpt",
        "tag": "v0.2.0",
        "commit": _SKILLOPT_SOURCE_COMMIT,
        "reason": "PyPI wheel omits Markdown package data",
        "files": {
            path.name: _sha256_file(path)
            for path in prompt_files.values()
        },
    }
    _write_json_once(
        out_root / "skillopt_prompt_bundle.json", prompt_metadata
    )

    previous_prompt_dir = skillopt_prompts._PROMPTS_DIR
    skillopt_prompts._PROMPTS_DIR = str(prompt_dir)
    skillopt_prompts.clear_cache()
    try:
        yield
    finally:
        skillopt_prompts._PROMPTS_DIR = previous_prompt_dir
        skillopt_prompts.clear_cache()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "SkillOptUnavailableError",
    "create_skillopt_trainer",
    "load_skillopt_config",
    "run_skillopt_training",
    "summarize_rollout_usage",
]
