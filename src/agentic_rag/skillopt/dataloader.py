"""SkillOpt-compatible batch planning for profile-driven A-RAG splits.

The module keeps SkillOpt as an optional dependency.  When SkillOpt v0.2.0 is
installed, :class:`AgenticRAGSkillOptDataLoader` is a real ``BaseDataLoader``
subclass.  The small fallback types let the repository exercise the complete
adapter contract in normal CI without installing SkillOpt's research runtime.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from agentic_rag.benchmark_profiles import AragDatasetProfile
from agentic_rag.skillopt.benchmark import split_manifest_profile
from agentic_rag.skillopt.lineage import load_json

try:  # pragma: no cover - exercised by the opt-in SkillOpt installation
    from skillopt.datasets.base import BaseDataLoader as _BaseDataLoader
    from skillopt.datasets.base import BatchSpec
except ImportError:  # pragma: no cover - fallback behavior is tested instead

    @dataclass(slots=True)
    class BatchSpec:  # type: ignore[no-redef]
        """Minimal v0.2.0-compatible batch record used without SkillOpt."""

        phase: str
        split: str
        seed: int
        batch_size: int
        payload: object | None = None
        metadata: dict[str, Any] | None = None

    class _BaseDataLoader:
        """Import-time fallback matching SkillOpt's relevant abstract surface."""

        def setup(self, cfg: dict[str, Any]) -> None:
            del cfg

        def set_out_root(self, out_root: str) -> None:
            del out_root


_SPLIT_FILES = {
    "train": "train.jsonl",
    "val": "validation.jsonl",
    "validation": "validation.jsonl",
    "valid_seen": "validation.jsonl",
    "selection": "validation.jsonl",
    "test": "test.jsonl",
    "valid_unseen": "test.jsonl",
}


class AgenticRAGSkillOptDataLoader(_BaseDataLoader):
    """Load deterministic profile-bound splits and plan SkillOpt batches."""

    def __init__(
        self,
        split_dir: str | Path,
        *,
        dataset: str | None = None,
        seed: int = 42,
        limit: int = 0,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.dataset = dataset
        self.seed = int(seed)
        self.limit = int(limit)
        if self.limit < 0:
            raise ValueError("limit must be non-negative")
        self._splits: dict[str, list[dict[str, Any]]] = {}
        self.profile: AragDatasetProfile | None = None
        self._scope_id: str | None = None

    def setup(self, cfg: dict[str, Any]) -> None:
        configured_dir = cfg.get("split_dir")
        if configured_dir and not str(self.split_dir):
            self.split_dir = Path(str(configured_dir))
        configured_dataset = cfg.get("dataset") or self.dataset
        self.profile = split_manifest_profile(
            self.split_dir,
            expected_dataset=(
                str(configured_dataset) if configured_dataset else None
            ),
        )
        self.dataset = self.profile.key
        manifest = load_json(self.split_dir / "split_manifest.json")
        dataset_metadata = (
            manifest.get("dataset") if isinstance(manifest, Mapping) else None
        )
        declared_scope = (
            dataset_metadata.get("scope_id")
            if isinstance(dataset_metadata, Mapping)
            else None
        )
        self._scope_id = (
            str(declared_scope)
            if isinstance(declared_scope, str) and declared_scope.strip()
            else self.profile.scope_id("dev")
        )
        self._splits = {
            "train": self._load("train"),
            "val": self._load("val"),
            "test": self._load("test"),
        }
        all_ids = [
            str(item["id"])
            for values in self._splits.values()
            for item in values
        ]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("SkillOpt train/validation/test IDs must be disjoint")

    def get_train_size(self) -> int:
        return len(self._require_setup("train"))

    @property
    def train_items(self) -> list[dict[str, Any]]:
        return list(self._require_setup("train"))

    @property
    def val_items(self) -> list[dict[str, Any]]:
        return list(self._require_setup("val"))

    @property
    def test_items(self) -> list[dict[str, Any]]:
        return list(self._require_setup("test"))

    def get_split_items(self, split: str) -> list[dict[str, Any]]:
        filename = _SPLIT_FILES.get(split)
        if filename == "train.jsonl":
            canonical = "train"
        elif filename == "validation.jsonl":
            canonical = "val"
        elif filename == "test.jsonl":
            canonical = "test"
        else:
            raise ValueError(f"unsupported SkillOpt split: {split}")
        return list(self._require_setup(canonical))

    def plan_train_epoch(
        self,
        *,
        epoch: int,
        steps_per_epoch: int,
        accumulation: int,
        batch_size: int,
        seed: int,
        **kwargs: Any,
    ) -> list[BatchSpec]:
        del kwargs
        if min(steps_per_epoch, accumulation, batch_size) <= 0:
            return []
        items = self.train_items
        rng = random.Random(seed + epoch * 1000)
        rng.shuffle(items)
        total_batches = steps_per_epoch * accumulation
        batches: list[BatchSpec] = []
        cursor = 0
        for batch_index in range(total_batches):
            selected = items[cursor : cursor + batch_size]
            cursor += len(selected)
            if not selected and items:
                selected = list(items[:batch_size])
            batches.append(
                BatchSpec(
                    phase="train",
                    split="train",
                    seed=seed + epoch * 1000 + batch_index + 1,
                    batch_size=len(selected),
                    payload=selected,
                )
            )
        return batches

    def build_train_batch(
        self, batch_size: int, seed: int, **kwargs: Any
    ) -> BatchSpec:
        del kwargs
        items = self.train_items
        random.Random(seed).shuffle(items)
        selected = items[:batch_size]
        return BatchSpec(
            phase="train",
            split="train",
            seed=seed,
            batch_size=len(selected),
            payload=selected,
        )

    def build_eval_batch(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs: Any,
    ) -> BatchSpec:
        del kwargs
        items = self.get_split_items(split)
        if env_num > 0:
            items = items[:env_num]
        return BatchSpec(
            phase="eval",
            split=split,
            seed=seed,
            batch_size=len(items),
            payload=items,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "split_dir": self.split_dir.as_posix(),
            "dataset": self.dataset,
            "seed": self.seed,
            "limit": self.limit,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "split_dir" in state:
            self.split_dir = Path(str(state["split_dir"]))
        if "dataset" in state and state["dataset"] is not None:
            self.dataset = str(state["dataset"])
        if "seed" in state:
            self.seed = int(state["seed"])
        if "limit" in state:
            self.limit = int(state["limit"])

    def _load(self, split: str) -> list[dict[str, Any]]:
        filename = _SPLIT_FILES[split]
        path = self.split_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing SkillOpt split file: {path}")
        items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise ValueError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                items.append(
                    _validate_item(
                        raw,
                        path=path,
                        line=line_number,
                        profile=self.profile,
                        scope_id=self._scope_id,
                    )
                )
        if self.limit:
            items = items[: self.limit]
        if not items:
            raise ValueError(f"SkillOpt split is empty: {path}")
        return items

    def _require_setup(self, split: str) -> list[dict[str, Any]]:
        if split not in self._splits:
            raise RuntimeError("SkillOpt dataloader.setup(cfg) must be called first")
        return self._splits[split]


def item_to_dict(item: object) -> dict[str, Any]:
    """Convert smoke dataset records to a detached plain dictionary."""

    if isinstance(item, Mapping):
        return dict(item)
    if is_dataclass(item) and not isinstance(item, type):
        return asdict(item)
    model_dump = getattr(item, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="python"))
    raise TypeError(
        "SkillOpt items must be mappings, dataclasses, or Pydantic models"
    )


def _validate_item(
    item: Mapping[str, Any],
    *,
    path: Path,
    line: int,
    profile: AragDatasetProfile | None,
    scope_id: str | None,
) -> dict[str, Any]:
    result = dict(item)
    required = ("id", "question", "scope_id", "answer")
    missing = [
        key
        for key in required
        if not isinstance(result.get(key), str) or not str(result[key]).strip()
    ]
    if missing:
        raise ValueError(f"{path}:{line} has invalid required fields: {missing}")
    if scope_id is not None and result["scope_id"] != scope_id:
        raise ValueError(
            f"{path}:{line} has scope_id {result['scope_id']!r}; "
            f"expected {scope_id!r}"
        )
    question_type = result.get("question_type") or result.get("task_type")
    if not isinstance(question_type, str) or not question_type.strip():
        raise ValueError(f"{path}:{line} has no question_type")
    if profile is not None and question_type not in profile.allowed_task_types:
        raise ValueError(
            f"{path}:{line} has unsupported question_type {question_type!r}; "
            f"expected one of {profile.allowed_task_types}"
        )
    result["question_type"] = question_type
    source = result.get("source")
    if profile is not None and source != profile.key:
        raise ValueError(
            f"{path}:{line} has source {source!r}; expected {profile.key!r}"
        )
    return result


__all__ = [
    "AgenticRAGSkillOptDataLoader",
    "BatchSpec",
    "item_to_dict",
]
