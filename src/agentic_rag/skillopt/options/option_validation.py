"""Validation rules for Options SkillOpt candidates.

This validator is intentionally stricter than a Markdown parser.  It checks
the option contract and rejects edits that try to smuggle episode-specific
facts into a reusable policy.  A caller can pass the current question and
reference ids as ``forbidden_literals`` for an additional per-episode check.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .option_store import ACTION_NAMES, OPTION_FIELDS, OptionSpec, OptionStore


REFERENCE_RE = re.compile(r"(?<![A-Za-z])[ESC]\d+(?![A-Za-z])")
QUESTION_LITERAL_RE = re.compile(r"\b(?:question|episode|document|answer)\s*[:#]?\s*[0-9a-f]{6,}\b", re.I)


def _walk_text(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_text(key)
            yield from _walk_text(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_text(item)


def validate_no_episode_literals(value: Any, *, forbidden_literals: Iterable[str] = ()) -> None:
    """Reject references and caller-supplied question-specific strings.

    ``E1``/``S2``/``C3`` are useful in an audit trajectory but never belong in
    a reusable option rule.  The validator does not attempt to guess names or
    ordinary English words; callers may pass exact episode strings when they
    need that stronger check.
    """
    forbidden = tuple(item for item in forbidden_literals if isinstance(item, str) and item)
    for text in _walk_text(value):
        match = REFERENCE_RE.search(text)
        if match:
            raise ValueError(f"episode-specific reference is not allowed: {match.group(0)}")
        match = QUESTION_LITERAL_RE.search(text)
        if match:
            raise ValueError(f"episode-specific literal is not allowed: {match.group(0)}")
        for literal in forbidden:
            if literal in text:
                raise ValueError(f"episode-specific literal is not allowed: {literal}")


def validate_option(option: OptionSpec, *, forbidden_literals: Iterable[str] = ()) -> None:
    """Validate one option independently of the store."""
    # Parsing already validates field names and action names.  These checks
    # make the invariants explicit at the candidate boundary.
    if not option.primitive_actions:
        raise ValueError(f"{option.option_id} needs a primitive action")
    if option.option_id == "O5_ANSWER" and "FINISH" not in option.primitive_actions:
        raise ValueError("O5_ANSWER must allow FINISH")
    if option.option_id != "O5_ANSWER" and "FINISH" in option.primitive_actions:
        raise ValueError(f"only O5_ANSWER may allow FINISH: {option.option_id}")
    if option.option_id == "FALLBACK" and "FINISH" in option.primitive_actions:
        raise ValueError("FALLBACK may not finish an episode")
    validate_no_episode_literals(option.to_mapping(), forbidden_literals=forbidden_literals)


def validate_store(store: OptionStore, *, forbidden_literals: Iterable[str] = ()) -> None:
    ids = [item.option_id for item in store.options]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate option ids")
    if "FALLBACK" not in ids:
        raise ValueError("FALLBACK is required")
    for option in store.options:
        validate_option(option, forbidden_literals=forbidden_literals)
    # The fixed guidance and answer contract may mention the protocol, but
    # neither block may be changed through an option edit.  We only check
    # non-emptiness here; immutability is enforced by apply_edits.
    if not store.fixed_runtime_guidance.strip() or not store.fixed_answer_contract.strip():
        raise ValueError("fixed blocks must be non-empty")

