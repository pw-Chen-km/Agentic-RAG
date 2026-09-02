"""Scoped SkillOpt ranking fixes; never edits the installed dependency.

An empty selection is a successful decision, not a parse error. Every
non-empty edit pool is checked, including pools smaller than the edit budget.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator


RANKING_COMPAT_VERSION = "agentic-rag-skillopt-ranking-v1"
MAX_RANKING_ATTEMPTS = 3


def build_safe_ranker(native: Any) -> Callable[..., dict[str, Any]]:
    original_ranker = native.rank_and_select

    def rank_and_select(
        skill_content: str,
        patch: dict,
        max_edits: int,
        meta_skill_context: str = "",
        update_mode: str = "patch",
    ) -> dict[str, Any]:
        mode = native.normalize_update_mode(update_mode)
        if mode != "patch":
            return original_ranker(
                skill_content, patch, max_edits,
                meta_skill_context=meta_skill_context, update_mode=update_mode,
            )
        if type(max_edits) is not int or max_edits < 0:
            raise ValueError("max_edits must be a non-negative integer")
        edits = native.get_payload_items(patch, mode)
        attempts: list[dict[str, Any]] = []

        def result(status: str, indices: list[int], parsed: dict | None = None) -> dict:
            return {
                **patch,
                "edits": [edits[index] for index in indices],
                "ranking_details": {
                    "version": RANKING_COMPAT_VERSION,
                    "status": status,
                    "selected_indices": indices,
                    "reasoning": (parsed or {}).get("reasoning", ""),
                    "attempts": attempts,
                },
            }

        if not edits:
            return result("no_candidates", [])
        if max_edits == 0:
            return result("no_budget", [])

        descriptions = "\n".join(
            f"[{index}] {native.describe_item(edit, mode)}"
            for index, edit in enumerate(edits)
        )
        user = (
            f"## Current Skill\n{skill_content}\n\n"
            f"## Edits Pool ({len(edits)} edits, budget={max_edits})\n"
            f"{descriptions}\n\n"
            f"Select at most {max_edits} edits. Select none when no edit is "
            "appropriate; selected_indices: [] is a valid final decision. "
            "Return distinct valid 0-based indices in priority order."
        )
        meta = native.format_meta_skill_context(meta_skill_context)
        if meta:
            user = f"{meta}\n\n{user}"
        system = native.load_prompt("ranking")
        for number in range(1, MAX_RANKING_ATTEMPTS + 1):
            attempt: dict[str, Any] = {"attempt": number}
            attempts.append(attempt)
            try:
                response, _ = native.chat_optimizer(
                    system=system, user=user, max_completion_tokens=16384,
                    # This loop owns retries: never multiply three attempts
                    # by the native chat wrapper's own retry count.
                    retries=1, stage="ranking",
                )
                attempt["response"] = response
                parsed = native.extract_json(response)
                if not isinstance(parsed, dict):
                    raise ValueError("ranking response must be a JSON object")
                indices = parsed.get("selected_indices")
                if not isinstance(indices, list):
                    raise ValueError("selected_indices must be a list")
                if len(indices) > max_edits:
                    raise ValueError("selected_indices exceeds the edit budget")
                if any(type(index) is not int or not 0 <= index < len(edits) for index in indices):
                    raise ValueError("selected_indices contains an invalid index")
                if len(set(indices)) != len(indices):
                    raise ValueError("selected_indices contains duplicate indices")
                attempt["status"] = "valid"
                return result("selected" if indices else "abstained", indices, parsed)
            except Exception as exc:
                attempt["status"] = "error"
                attempt["error_type"] = type(exc).__name__
                attempt["error"] = str(exc)
        # An infrastructure/format error is distinguishable from abstention,
        # but neither is permission to apply an unselected edit.
        return result("ranking_error", [])

    return rank_and_select


@contextmanager
def use_skillopt_ranking_compatibility() -> Iterator[None]:
    try:
        import skillopt.engine.trainer as native_trainer
        import skillopt.optimizer.clip as native_clip
    except ImportError:
        # Fake-trainer CI can run without the optional SkillOpt dependency.
        yield
        return
    original_clip = native_clip.rank_and_select
    original_trainer = native_trainer.rank_and_select
    safe_ranker = build_safe_ranker(native_clip)
    native_clip.rank_and_select = safe_ranker
    native_trainer.rank_and_select = safe_ranker
    try:
        yield
    finally:
        native_clip.rank_and_select = original_clip
        native_trainer.rank_and_select = original_trainer
