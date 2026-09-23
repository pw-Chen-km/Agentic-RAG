"""The structured option catalogue used by Options SkillOpt v1.

The runtime receives Markdown, but the optimiser never edits that Markdown
directly.  This module is deliberately small and deterministic: a JSON
catalogue is the source of truth and the Markdown rendering is derived from
it.  Keeping the two representations separate makes a candidate auditable
and makes it impossible for an optimiser response to silently add a new
field.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


OPTION_IDS = (
    "O1_START_SEARCH",
    "O2_RESOLVE_FACT",
    "O3_RESOLVE_BRIDGE",
    "O4_RECOVER",
    "O5_ANSWER",
    "FALLBACK",
)
OPTION_FIELDS = {
    "name",
    "goal",
    "initiation",
    "primitive_actions",
    "policy",
    "termination",
    "interrupt_when",
}
ACTION_NAMES = {"SEARCH", "EXPAND", "READ", "FINISH"}


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of strings")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must contain non-empty strings")
    return tuple(item.strip() for item in value)


def _text_map(value: Any, field: str, *, allow_empty: bool = False) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    result: dict[str, str] = {}
    for key, item in value.items():
        key = _required_text(key, f"{field} key")
        result[key] = _required_text(item, f"{field}.{key}")
    return result


@dataclass(frozen=True)
class OptionSpec:
    """One option, with the fields defined by the option contract."""

    option_id: str
    name: str
    goal: str
    initiation: tuple[str, ...]
    primitive_actions: tuple[str, ...]
    policy: dict[str, str]
    termination: dict[str, str]
    interrupt_when: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OptionSpec":
        if not isinstance(value, Mapping):
            raise ValueError("option must be an object")
        unknown = set(value) - {"option_id", *OPTION_FIELDS}
        if unknown:
            raise ValueError(f"unknown option fields: {sorted(unknown)}")
        option_id = _required_text(value.get("option_id"), "option_id")
        if not re.fullmatch(r"(?:O[1-9][0-9]*_[A-Z0-9_]+|FALLBACK)", option_id):
            raise ValueError(f"invalid option_id: {option_id}")
        actions = _string_list(value.get("primitive_actions"), "primitive_actions")
        invalid_actions = set(actions) - ACTION_NAMES
        if invalid_actions:
            raise ValueError(f"unknown primitive actions: {sorted(invalid_actions)}")
        if len(set(actions)) != len(actions):
            raise ValueError("primitive_actions must not contain duplicates")
        initiation = _string_list(value.get("initiation"), "initiation")
        policy = _text_map(value.get("policy"), "policy")
        termination = _text_map(value.get("termination"), "termination")
        return cls(
            option_id=option_id,
            name=_required_text(value.get("name"), "name"),
            goal=_required_text(value.get("goal"), "goal"),
            initiation=initiation,
            primitive_actions=actions,
            policy=policy,
            termination=termination,
            interrupt_when=_required_text(value.get("interrupt_when"), "interrupt_when"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "name": self.name,
            "goal": self.goal,
            "initiation": list(self.initiation),
            "primitive_actions": list(self.primitive_actions),
            "policy": dict(self.policy),
            "termination": dict(self.termination),
            "interrupt_when": self.interrupt_when,
        }

    def render(self, *, full: bool = True) -> str:
        lines = [f"### {self.option_id}: {self.name}", f"Goal: {self.goal}"]
        if not full:
            lines.append("Use this option when: " + "; ".join(self.initiation))
            return "\n".join(lines)
        lines.append("Initiation conditions:")
        lines.extend(f"- {item}" for item in self.initiation)
        lines.append("Primitive actions: " + ", ".join(self.primitive_actions))
        lines.append("Observation-driven policy:")
        lines.extend(f"- If {key}: {value}" for key, value in self.policy.items())
        lines.append("Termination:")
        lines.extend(f"- {key}: {value}" for key, value in self.termination.items())
        lines.append(f"Interrupt when: {self.interrupt_when}")
        return "\n".join(lines)


@dataclass(frozen=True)
class OptionStore:
    """Immutable option source that can be safely copied for a candidate."""

    version: str
    fixed_runtime_guidance: str
    fixed_answer_contract: str
    options: tuple[OptionSpec, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OptionStore":
        if not isinstance(value, Mapping):
            raise ValueError("option catalogue must be an object")
        unknown = set(value) - {"version", "fixed_runtime_guidance", "fixed_answer_contract", "options"}
        if unknown:
            raise ValueError(f"unknown catalogue fields: {sorted(unknown)}")
        raw_options = value.get("options")
        if not isinstance(raw_options, list) or not raw_options:
            raise ValueError("options must be a non-empty list")
        options = tuple(OptionSpec.from_mapping(item) for item in raw_options)
        ids = [item.option_id for item in options]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate option_id")
        if "FALLBACK" not in ids:
            raise ValueError("catalogue must contain FALLBACK")
        return cls(
            version=_required_text(value.get("version"), "version"),
            fixed_runtime_guidance=_required_text(value.get("fixed_runtime_guidance"), "fixed_runtime_guidance"),
            fixed_answer_contract=_required_text(value.get("fixed_answer_contract"), "fixed_answer_contract"),
            options=options,
        )

    @classmethod
    def from_json(cls, path: Path) -> "OptionStore":
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fixed_runtime_guidance": self.fixed_runtime_guidance,
            "fixed_answer_contract": self.fixed_answer_contract,
            "options": [item.to_mapping() for item in self.options],
        }

    def json_hash(self) -> str:
        payload = json.dumps(self.to_mapping(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def option_index(self) -> dict[str, OptionSpec]:
        return {item.option_id: item for item in self.options}

    def get(self, option_id: str) -> OptionSpec:
        try:
            return self.option_index()[option_id]
        except KeyError as exc:
            raise KeyError(f"unknown option: {option_id}") from exc

    def selectable_ids(self, *, include_fallback: bool = True) -> tuple[str, ...]:
        ids = tuple(item.option_id for item in self.options)
        if include_fallback:
            return ids
        return tuple(item for item in ids if item != "FALLBACK")

    def markdown(self, *, selected_option: str | None = None) -> str:
        """Render a progressive-disclosure Skill for the Agent.

        The selector gets short summaries.  Once an option is selected, the
        caller can render it again with ``selected_option`` to reveal only
        that option's full procedure.
        """
        lines = [f"# Agentic-RAG Options Skill ({self.version})", "", "## Fixed runtime guidance", self.fixed_runtime_guidance, "", "## Option catalogue"]
        if selected_option is None:
            lines.append("Choose one option before selecting a primitive action:")
            for item in self.options:
                lines.append(f"- {item.option_id}: {item.name} — {item.goal}")
                lines.append("  Start when: " + "; ".join(item.initiation))
        else:
            option = self.get(selected_option)
            lines.append(f"The selected option is **{option.option_id}**. Follow its procedure until it is COMPLETE or BLOCKED.")
            lines.append(option.render(full=True))
        lines.extend(["", "## Fixed answer contract", self.fixed_answer_contract, ""])
        return "\n".join(lines)

    def optimizer_view(self, stage: str = "policy") -> dict[str, Any]:
        """Return only editable structured data, never Markdown."""
        return {
            "version": self.version,
            "stage": stage,
            "options": [item.to_mapping() for item in self.options],
            "fixed_fields": ["fixed_runtime_guidance", "fixed_answer_contract"],
        }

    def with_options(self, options: Sequence[OptionSpec]) -> "OptionStore":
        return replace(self, options=tuple(options))
