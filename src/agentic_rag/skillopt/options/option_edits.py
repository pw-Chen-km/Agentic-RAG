"""Restricted, auditable edits for Options SkillOpt v1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .option_store import OPTION_FIELDS, OptionSpec, OptionStore
from .option_validation import validate_option, validate_store


EDIT_OPERATIONS = {"refine", "replace", "add_policy_case", "add_option", "delete_option", "no_change"}
STAGES = {"selection", "policy", "termination", "meta"}
FIELD_ROOTS = {"name", "goal", "initiation", "primitive_actions", "policy", "termination", "interrupt_when"}


def _field_value(option: OptionSpec, field: str) -> Any:
    parts = field.split(".")
    if parts[0] not in FIELD_ROOTS or parts[0] not in OPTION_FIELDS:
        raise ValueError(f"invalid editable field: {field}")
    value: Any = option.to_mapping()
    for part in parts:
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"unknown editable field: {field}")
        value = value[part]
    return value


def _set_field(data: dict[str, Any], field: str, value: Any) -> None:
    parts = field.split(".")
    if not parts or parts[0] not in FIELD_ROOTS:
        raise ValueError(f"invalid editable field: {field}")
    cursor: dict[str, Any] = data
    for part in parts[:-1]:
        nested = cursor.get(part)
        if not isinstance(nested, dict):
            raise ValueError(f"unknown editable field: {field}")
        cursor = nested
    cursor[parts[-1]] = value


def _stage_allows(stage: str, field: str) -> bool:
    root = field.split(".", 1)[0]
    if stage == "selection":
        return root in {"name", "goal", "initiation"}
    if stage == "policy":
        return root in {"primitive_actions", "policy"}
    if stage == "termination":
        return root in {"termination", "interrupt_when"}
    if stage == "meta":
        return root in FIELD_ROOTS
    return False


def _token_count(store: OptionStore) -> int:
    """Stable, dependency-free approximation used for candidate size guards."""
    text = "\n".join(
        f"{item.option_id} {item.name} {item.goal} {' '.join(item.initiation)} "
        f"{' '.join(item.primitive_actions)} {' '.join(item.policy.values())} "
        f"{' '.join(item.termination.values())} {item.interrupt_when}"
        for item in store.options
    )
    return len(text.split())


def _candidate_option(value: Any, option_id: str | None = None) -> OptionSpec:
    if not isinstance(value, Mapping):
        raise ValueError("option value must be an object")
    payload = dict(value)
    if option_id is not None:
        payload["option_id"] = option_id
    return OptionSpec.from_mapping(payload)


@dataclass(frozen=True)
class EditReceipt:
    applied: tuple[dict[str, Any], ...]
    changed_option_ids: tuple[str, ...]
    no_change: bool
    before_hash: str
    after_hash: str
    option_count_before: int
    option_count_after: int
    skill_tokens_before: int = 0
    skill_tokens_after: int = 0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "applied": [dict(item) for item in self.applied],
            "changed_option_ids": list(self.changed_option_ids),
            "no_change": self.no_change,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "option_count_before": self.option_count_before,
            "option_count_after": self.option_count_after,
            "skill_tokens_before": self.skill_tokens_before,
            "skill_tokens_after": self.skill_tokens_after,
        }


def apply_edits(
    store: OptionStore,
    edits: Sequence[Mapping[str, Any]] | None,
    *,
    stage: str,
    max_edits: int = 2,
    max_delta_tokens: int = 300,
    max_total_tokens: int = 2500,
    forbidden_literals: Iterable[str] = (),
) -> tuple[OptionStore, EditReceipt]:
    """Apply at most a small number of edits and return a new catalogue.

    ``refine`` and ``replace`` are intentionally aliases at this layer: the
    former is the preferred human-facing operation, while the latter is
    useful when an analyst explicitly wants to replace a field.  Neither can
    create a new key or mutate fixed guidance/answer contract.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    edits = list(edits or [])
    if len(edits) > max_edits:
        raise ValueError(f"too many edits: {len(edits)} > {max_edits}")
    validate_store(store, forbidden_literals=forbidden_literals)
    if not edits:
        tokens = _token_count(store)
        return store, EditReceipt((), (), True, store.json_hash(), store.json_hash(), len(store.options), len(store.options), tokens, tokens)
    index = store.option_index()
    options = [item.to_mapping() for item in store.options]
    touched: set[str] = set()
    receipts: list[dict[str, Any]] = []
    for raw in edits:
        if not isinstance(raw, Mapping):
            raise ValueError("each edit must be an object")
        operation = raw.get("operation")
        if operation == "no_change":
            if len(edits) != 1:
                raise ValueError("no_change cannot be combined with edits")
            tokens = _token_count(store)
            return store, EditReceipt((), (), True, store.json_hash(), store.json_hash(), len(store.options), len(store.options), tokens, tokens)
        if operation not in EDIT_OPERATIONS - {"no_change"}:
            raise ValueError(f"invalid option edit operation: {operation}")
        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("an option edit needs a non-empty reason")
        support = raw.get("supporting_case_ids")
        if not isinstance(support, list) or len(set(support)) < 2 or any(not isinstance(item, str) or not item for item in support):
            raise ValueError("an option edit needs at least two distinct supporting training cases")
        option_id = raw.get("option_id")
        if operation in {"refine", "replace", "add_policy_case", "delete_option"}:
            if not isinstance(option_id, str) or option_id not in index:
                raise ValueError(f"edit targets unknown option: {option_id}")
        if option_id in touched:
            raise ValueError(f"option modified more than once: {option_id}")
        if operation in {"refine", "replace"}:
            field = raw.get("field")
            if not isinstance(field, str) or not _stage_allows(stage, field):
                raise ValueError(f"stage {stage} cannot edit field: {field}")
            old = index[option_id]
            _field_value(old, field)
            value = raw.get("new_value")
            if value is None:
                raise ValueError("new_value is required")
            candidate = old.to_mapping()
            _set_field(candidate, field, value)
            option = _candidate_option(candidate, option_id)
            validate_option(option, forbidden_literals=forbidden_literals)
            options = [option.to_mapping() if item.get("option_id") == option_id else item for item in options]
            receipts.append({"operation": operation, "option_id": option_id, "field": field})
        elif operation == "add_policy_case":
            if stage not in {"policy", "meta"}:
                raise ValueError("add_policy_case is only allowed in policy or meta stage")
            observation = raw.get("observation")
            response = raw.get("response")
            if not isinstance(observation, str) or not observation.strip() or not isinstance(response, str) or not response.strip():
                raise ValueError("add_policy_case requires observation and response")
            old = index[option_id]
            candidate = old.to_mapping()
            policy = dict(candidate["policy"])
            if observation in policy:
                raise ValueError(f"policy case already exists: {observation}")
            policy[observation] = response
            candidate["policy"] = policy
            option = _candidate_option(candidate, option_id)
            validate_option(option, forbidden_literals=forbidden_literals)
            options = [option.to_mapping() if item.get("option_id") == option_id else item for item in options]
            receipts.append({"operation": operation, "option_id": option_id, "field": f"policy.{observation}"})
        elif operation == "add_option":
            if stage not in {"selection", "meta"}:
                raise ValueError("add_option is only allowed in selection or meta stage")
            option = _candidate_option(raw.get("option"))
            if option.option_id in index or option.option_id in touched:
                raise ValueError(f"option id already exists: {option.option_id}")
            validate_option(option, forbidden_literals=forbidden_literals)
            options.append(option.to_mapping())
            index[option.option_id] = option
            option_id = option.option_id
            receipts.append({"operation": operation, "option_id": option_id})
        elif operation == "delete_option":
            if stage not in {"selection", "meta"}:
                raise ValueError("delete_option is only allowed in selection or meta stage")
            if option_id in {"O1_START_SEARCH", "O5_ANSWER", "FALLBACK"}:
                raise ValueError(f"protected option cannot be deleted: {option_id}")
            options = [item for item in options if item.get("option_id") != option_id]
            receipts.append({"operation": operation, "option_id": option_id})
        touched.add(option_id)
        # Index must reflect a replacement before the next edit.  This also
        # makes duplicate references within one response deterministic.
        index = {item["option_id"]: OptionSpec.from_mapping(item) for item in options}
    candidate_store = OptionStore.from_mapping({**store.to_mapping(), "options": options})
    validate_store(candidate_store, forbidden_literals=forbidden_literals)
    before_tokens = _token_count(store)
    after_tokens = _token_count(candidate_store)
    if after_tokens > max_total_tokens:
        raise ValueError(f"option Skill token limit exceeded: {after_tokens} > {max_total_tokens}")
    if after_tokens - before_tokens > max_delta_tokens:
        raise ValueError(f"option edit token delta exceeds limit: {after_tokens - before_tokens} > {max_delta_tokens}")
    receipt = EditReceipt(tuple(receipts), tuple(sorted(touched)), False, store.json_hash(), candidate_store.json_hash(), len(store.options), len(candidate_store.options), before_tokens, after_tokens)
    return candidate_store, receipt
