"""Run completed Agentic RAG episodes as SkillOpt rollout samples."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from pydantic import BaseModel

from agentic_rag.agent.models import EpisodeResult
from agentic_rag.evaluation import EpisodeEvaluator, EvaluationResult
from agentic_rag.paths import portable_path_component
from agentic_rag.skillopt.dataloader import item_to_dict


class EpisodeHarness(Protocol):
    """The narrow existing-harness surface required by SkillOpt."""

    def run(
        self,
        question: str,
        scope_id: str,
        *,
        episode_id: str | None = None,
    ) -> EpisodeResult: ...

    def build_io_trace(self, result: EpisodeResult) -> dict[str, Any]: ...


class HarnessFactory(Protocol):
    """Build one candidate-skill harness rooted at a prediction directory."""

    def __call__(
        self, *, skill_content: str, output_root: Path
    ) -> EpisodeHarness: ...


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """Items plus phase metadata that must not be exposed to the target."""

    items: tuple[dict[str, Any], ...]
    phase: str
    split: str


def run_rollout_batch(
    *,
    batch: RolloutBatch | Sequence[object],
    out_root: str | Path,
    skill_content: str,
    harness_factory: HarnessFactory,
    evaluator: EpisodeEvaluator,
    workers: int = 1,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Run and score a batch, persisting SkillOpt's per-task contract.

    A gold answer is passed only to the post-episode evaluator.  It is attached
    as ``reference_text`` only for training rollouts so validation/test labels
    cannot enter reflection patches.
    """

    if workers != 1:
        raise ValueError(
            "the inspectable smoke workflow requires workers=1"
        )
    if not skill_content.strip():
        raise ValueError("skill_content must not be blank")

    if isinstance(batch, RolloutBatch):
        phase = batch.phase
        split = batch.split
        raw_items: Sequence[object] = batch.items
    else:
        phase = "train"
        split = "train"
        raw_items = batch
    if phase not in {"train", "eval"}:
        raise ValueError(f"unsupported rollout phase: {phase}")

    root = Path(out_root)
    prediction_root = root / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    harness = harness_factory(
        skill_content=skill_content,
        output_root=prediction_root,
    )

    results: list[dict[str, Any]] = []
    for raw_item in raw_items:
        item = item_to_dict(raw_item)
        _validate_rollout_item(item)
        task_id = str(item["id"])
        task_dir = prediction_root / portable_path_component(task_id)
        persisted_result = task_dir / "rollout_result.json"
        if resume and persisted_result.is_file():
            _require_matching_skill(task_dir, skill_content)
            loaded = json.loads(persisted_result.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"invalid persisted rollout: {persisted_result}")
            results.append(loaded)
            continue

        episode_path = task_dir / "episode.json"
        if resume and episode_path.is_file():
            _require_matching_skill(task_dir, skill_content)
            episode = EpisodeResult.model_validate_json(
                episode_path.read_text(encoding="utf-8")
            )
            if (
                episode.episode_id != task_id
                or episode.query != str(item["question"])
                or episode.scope_id != str(item["scope_id"])
            ):
                raise ValueError(
                    f"persisted Episode does not match rollout item: {task_dir}"
                )
            episode.artifact_dir = task_dir.as_posix()
        else:
            # Only these two target inputs cross the benchmark boundary.  In
            # particular, answer, source, split, and evidence stay outside
            # Harness.
            episode = harness.run(
                str(item["question"]),
                str(item["scope_id"]),
                episode_id=task_id,
            )
        actual_task_dir = Path(episode.artifact_dir or task_dir)
        if actual_task_dir != task_dir:
            raise RuntimeError(
                "Harness artifact_dir must equal the portable task directory"
            )
        # Normalize the adapter boundary even when an injected Harness returns
        # native Windows separators. Fresh and resumed rollout records must be
        # byte-for-byte comparable on every supported operating system.
        episode.artifact_dir = actual_task_dir.as_posix()

        # Commit the complete, provider-free trace before the judge boundary.
        # If the judge transport fails, the six Harness artifacts plus this
        # trace are a durable Episode checkpoint and the target Agent must not
        # be run a second time on retry.
        io_trace_path = actual_task_dir / "io_trace.json"
        if not (resume and io_trace_path.is_file()):
            _write_json_once(
                io_trace_path,
                harness.build_io_trace(episode),
            )

        evaluation_path = actual_task_dir / "evaluation.json"
        if resume and evaluation_path.is_file():
            evaluation = EvaluationResult.model_validate_json(
                evaluation_path.read_text(encoding="utf-8")
            )
            if (
                evaluation.question != str(item["question"])
                or evaluation.predicted_answer != (episode.answer or "")
                or evaluation.gold_answer != str(item["answer"])
            ):
                raise ValueError(
                    "persisted evaluation does not match the Episode or label: "
                    f"{evaluation_path}"
                )
        else:
            evaluation = evaluator.evaluate(
                question=str(item["question"]),
                predicted_answer=episode.answer,
                gold_answer=str(item["answer"]),
            )
        rollout_result = _build_rollout_result(
            item=item,
            episode=episode,
            evaluation=evaluation,
            phase=phase,
            split=split,
        )
        _write_json_once(
            actual_task_dir / "evaluation.json",
            evaluation.model_dump(mode="json"),
        )
        _write_json_once(
            actual_task_dir / "rollout_result.json",
            rollout_result,
        )
        results.append(rollout_result)
    return results


def _build_rollout_result(
    *,
    item: Mapping[str, Any],
    episode: EpisodeResult,
    evaluation: EvaluationResult,
    phase: str,
    split: str,
) -> dict[str, Any]:
    termination = episode.termination_reason.value
    if episode.error_code:
        fail_reason = episode.error_code
    elif evaluation.hard == 0:
        # SkillOpt's shared reflector reads this compact header plus the
        # conversation trajectory; the final Answer Generator output is not
        # part of conversation.json.  Include it here so train reflection can
        # compare the actual failure with its hidden reference_text.
        prediction = (episode.answer or "<empty answer>").strip()
        fail_reason = (
            f"{evaluation.hard_metric.value}=0; "
            f"{evaluation.soft_metric.value}={evaluation.soft:g}; "
            f"predicted_answer={prediction[:500]}"
        )
    else:
        fail_reason = None

    task_type = item.get("question_type") or item.get("task_type") or "qa"
    all_metrics = {
        "llm_acc": evaluation.llm_acc,
        "contain_acc": evaluation.contain_acc,
    }
    result: dict[str, Any] = {
        "id": str(item["id"]),
        "hard": int(evaluation.hard),
        "soft": float(evaluation.soft),
        "predicted_answer": episode.answer or "",
        "question": str(item["question"]),
        "task_description": str(item["question"]),
        "task_type": str(task_type),
        "dataset": evaluation.dataset,
        "n_turns": len(episode.trajectory),
        "fail_reason": fail_reason,
        "termination_reason": termination,
        "rollout_phase": phase,
        "rollout_split": split,
        "artifact_dir": str(episode.artifact_dir or ""),
        "metrics": {
            metric.value: all_metrics[metric.value]
            for metric in evaluation.reported_metrics
        },
        "metric_contract": {
            "hard": evaluation.hard_metric.value,
            "soft": evaluation.soft_metric.value,
        },
        "usage": episode.usage.model_dump(mode="json"),
        "judge_usage": evaluation.judge_usage.model_dump(mode="json"),
    }
    if phase == "train":
        result["reference_text"] = str(item["answer"])
    return result


def _validate_rollout_item(item: Mapping[str, Any]) -> None:
    missing = [
        field
        for field in ("id", "question", "scope_id", "answer")
        if not isinstance(item.get(field), str)
        or not str(item[field]).strip()
    ]
    if missing:
        raise ValueError(f"rollout item has invalid required fields: {missing}")
    task_id = str(item["id"])
    if (
        task_id in {".", ".."}
        or "/" in task_id
        or "\\" in task_id
        or "\x00" in task_id
    ):
        raise ValueError("rollout item id must be one safe path component")


def _require_matching_skill(task_dir: Path, skill_content: str) -> None:
    skill_path = task_dir / "skill.md"
    if not skill_path.is_file():
        raise FileNotFoundError(
            f"resumed rollout is missing its skill snapshot: {skill_path}"
        )
    if skill_path.read_text(encoding="utf-8") != skill_content:
        raise FileExistsError(
            f"persisted rollout uses a different candidate skill: {task_dir}"
        )


def _write_json_once(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            _to_jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"workflow artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _to_jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, BaseModel):
        return _to_jsonable(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    raise TypeError(f"unsupported workflow artifact value: {type(value).__name__}")


__all__ = [
    "EpisodeHarness",
    "HarnessFactory",
    "RolloutBatch",
    "run_rollout_batch",
]
