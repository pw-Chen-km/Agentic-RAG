"""Replay rejected Skill candidates only on their original train questions."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .skill_rules import skill_sha256
from .support import digest


def replay_rejected_candidate(
    *,
    candidate_skill_hash: str,
    parent_skill_hash: str,
    train_question_ids: Iterable[str],
    rollout: Callable[[str, str], Mapping[str, Any]],
    seen_keys: set[tuple[str, tuple[str, ...]]] | None = None,
) -> list[Mapping[str, Any]]:
    ids = tuple(str(item) for item in train_question_ids)
    key = (candidate_skill_hash, ids)
    if seen_keys is not None:
        if key in seen_keys:
            return []
        seen_keys.add(key)
    return [rollout(question_id, candidate_skill_hash) for question_id in ids]


def collect_train_branches(
    engine, directory, batch, questions, parent, candidate, gate=None
):
    """Durable train-only replay using Phase1's per-Skill/per-question cache.

    Active Skill adoption is handled by the caller. This function only executes
    and records branches, irrespective of the validation decision.
    """
    parent_hash, candidate_hash = skill_sha256(parent), skill_sha256(candidate)
    inputs = {
        "batch": batch,
        "question_rows": digest(questions),
        "parent_skill_hash": parent_hash,
        "candidate_skill_hash": candidate_hash,
        "runtime_contract_hash": engine.replay_contract_hash(batch),
    }

    def reference(question, skill, version):
        episode = engine.episode(question, skill, "train", version)
        qid = str(question["id"])
        folder = (
            engine.root
            / "episodes"
            / "train"
            / skill_sha256(skill)
            / digest({"id": qid})[:20]
        )
        return {
            "question_id": qid,
            "split": "train",
            "skill_hash": skill_sha256(skill),
            "episode_directory": str(folder.relative_to(engine.root)),
            "raw_hash": digest(episode["raw"]),
            "evaluation_hash": digest(episode["evaluation"]),
        }

    def collect():
        if parent_hash == candidate_hash:
            return {
                **inputs,
                "status": "no_changed_candidate",
                "purpose": "meta_analysis_only",
                "pairs": [],
            }

        def pair(question):
            # Completed parent rollouts are loaded, not generated again. If this
            # process is interrupted, completed candidate episodes are also reused.
            a = reference(question, parent, f"batch {batch} parent")
            b = reference(question, candidate, f"batch {batch} candidate train replay")
            return {"question_id": str(question["id"]), "parent": a, "candidate": b}

        pairs = engine.map_ordered(pair, questions, engine.cfg.episode_workers)
        return {
            **inputs,
            "status": "completed",
            "purpose": "meta_analysis_only",
            "candidate_gate_accepted": gate["accepted"] if gate is not None else None,
            "validation_status": "completed" if gate is not None else "not_run_yet",
            "pairs": pairs,
        }

    return engine.cached(Path(directory) / "train_branches.json", inputs, collect)
