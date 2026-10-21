"""Canonical formatted Skill for A-RAG.

The source of the initial skill is the two strategy sentences in A-RAG's
official prompt.  The JSON representation is intentionally small: ``when``
states when a rule applies and ``action_sequence`` transcribes what the
original prompt asks the model to do.  There is no hidden recovery or stopping
field in the initial version.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping
from .tokens import count, rule_text

RULE_KEYS = ("rule_id", "title", "when", "action_sequence")
ALLOWED_SECTION = "retrieval_policy"
FORBIDDEN_KEYS = {"stop_or_recovery", "after_observation", "completion_condition"}


def load_rules(path: str | Path) -> dict[str, Any]:
    """Load and validate the canonical A-RAG skill JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "arag-skill-v1":
        raise ValueError("unsupported A-RAG skill schema")
    if set(payload) - {"schema_version", "retrieval_policy", "source_prompt_sha256"}:
        raise ValueError("unexpected top-level skill field")
    rules = payload.get("retrieval_policy")
    if not isinstance(rules, list):
        raise ValueError("retrieval_policy must be a list")
    seen: set[str] = set()
    for rule in rules:
        if rule["rule_id"] in seen:
            raise ValueError("duplicate rule_id")
        validate_rule(rule, existing_ids=seen, require_new=False)
        seen.add(rule["rule_id"])
    return payload


def validate_rule(
    rule: Mapping[str, Any], *, existing_ids: Iterable[str] = (), require_new: bool = False
) -> None:
    keys = set(rule)
    if keys != set(RULE_KEYS):
        raise ValueError(f"rule fields must be exactly {RULE_KEYS}; got {sorted(keys)}")
    for key in FORBIDDEN_KEYS:
        if key in rule:
            raise ValueError(f"{key} is not part of arag-skill-v1")
    rid = rule["rule_id"]
    if not isinstance(rid, str) or not rid.startswith("R") or not rid[1:].isdigit():
        raise ValueError("rule_id must look like R01")
    if require_new and rid in set(existing_ids):
        raise ValueError(f"rule_id already exists: {rid}")
    if not isinstance(rule["title"], str) or not rule["title"].strip():
        raise ValueError("title must be non-empty")
    if not isinstance(rule["when"], str) or not rule["when"].strip():
        raise ValueError("when must be non-empty")
    sequence = rule["action_sequence"]
    if not isinstance(sequence, list) or not sequence or not all(
        isinstance(item, str) and item.strip() for item in sequence
    ):
        raise ValueError("action_sequence must be a non-empty list of strings")


def rule_store(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {rule["rule_id"]: copy.deepcopy(rule) for rule in payload["retrieval_policy"]}


def _estimate_tokens(value: Any) -> int:
    """Compatibility name: now a real, explicitly named tokenizer."""
    return count(value)


class RuleStore:
    """Apply bounded optimizer edits to the formatted A-RAG rules."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        max_edits: int = 2,
        max_tokens: int = 300,
        max_skill_tokens: int = 1500,
    ):
        self.payload = copy.deepcopy(dict(payload))
        self.max_edits = max_edits
        self.max_tokens = max_tokens
        self.max_skill_tokens = max_skill_tokens
        self._validate_payload()

    def _validate_payload(self) -> None:
        ids: set[str] = set()
        for rule in self.payload.get("retrieval_policy", []):
            if rule["rule_id"] in ids:
                raise ValueError("duplicate rule_id")
            validate_rule(rule, existing_ids=ids)
            ids.add(rule["rule_id"])

    @property
    def rules(self) -> list[dict[str, Any]]:
        return self.payload["retrieval_policy"]

    def apply(self, edits: list[Mapping[str, Any]]) -> dict[str, Any]:
        if len(edits) > self.max_edits:
            raise ValueError(f"at most {self.max_edits} edits are allowed")
        for edit in edits:
            if edit.get('operation') in {'add', 'replace'}:
                if not isinstance(edit.get('rule'), Mapping):
                    raise ValueError('add/replace requires a rule object')
                validate_rule(edit['rule'])
        if any(count(rule_text(edit.get("rule", {}))) > self.max_tokens
               for edit in edits if edit.get("operation") in {"add", "replace"}):
            raise ValueError(f"a rule body exceeds {self.max_tokens} tokens (o200k_base)")
        touched = [e.get("rule_id") for e in edits if e.get("operation") != "no_change"]
        if len(touched) != len(set(touched)):
            raise ValueError("multiple edits to the same rule in one candidate")
        result = copy.deepcopy(self.payload)
        ids = {item["rule_id"] for item in result["retrieval_policy"]}
        for edit in edits:
            operation = edit.get("operation")
            rid = edit.get("rule_id")
            if operation == "add":
                if rid in ids:
                    raise ValueError(f"add uses existing rule_id: {rid}")
                rule = edit.get("rule")
                if not isinstance(rule, Mapping):
                    raise ValueError("add requires rule")
                if rule.get("rule_id") != rid:
                    raise ValueError("outer and inner rule_id differ")
                validate_rule(rule, existing_ids=ids, require_new=True)
                result["retrieval_policy"].append(copy.deepcopy(dict(rule)))
                ids.add(rid)
            elif operation == "replace":
                if rid not in ids:
                    raise ValueError(f"replace uses unknown rule_id: {rid}")
                rule = edit.get("rule")
                if not isinstance(rule, Mapping) or rule.get("rule_id") != rid:
                    raise ValueError("replace requires the existing rule_id")
                validate_rule(rule, existing_ids=ids)
                idx = next(i for i, item in enumerate(result["retrieval_policy"]) if item["rule_id"] == rid)
                result["retrieval_policy"][idx] = copy.deepcopy(dict(rule))
            elif operation == "delete":
                if rid not in ids:
                    raise ValueError(f"delete uses unknown rule_id: {rid}")
                result["retrieval_policy"] = [
                    item for item in result["retrieval_policy"] if item["rule_id"] != rid
                ]
                ids.remove(rid)
            elif operation == "no_change":
                if len(edits) != 1:
                    raise ValueError("no_change cannot be combined with an edit")
                if not isinstance(edit.get("reason"), str) or not edit["reason"].strip():
                    raise ValueError("no_change requires a reason")
            else:
                raise ValueError(f"unsupported operation: {operation}")
        signatures = [re.sub(r"\s+", " ", json.dumps({k: r[k] for k in ("when", "action_sequence")},
                      sort_keys=True, ensure_ascii=False)).casefold() for r in result["retrieval_policy"]]
        if len(signatures) != len(set(signatures)):
            raise ValueError("duplicate condition/action rule; use replace or no_change")
        if count(compile_initial_skill(result)) > self.max_skill_tokens:
            raise ValueError(f"editable Skill exceeds {self.max_skill_tokens} tokens")
        # Validate the candidate before mutating the active store.  A rejected
        # proposal must not leave a partially applied rule behind.
        RuleStore(result, max_edits=self.max_edits, max_tokens=self.max_tokens,
                  max_skill_tokens=self.max_skill_tokens)._validate_payload()
        self.payload = result
        return copy.deepcopy(result)

    def apply_with_receipt(self, edits: list[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply edits and return an audit receipt alongside the candidate.

        The receipt keeps parent/candidate hashes and cost metadata separate
        from the runtime Skill, so prompts cannot accidentally receive audit
        labels or case-specific explanations.
        """
        before = copy.deepcopy(self.payload)
        before_hash = skill_sha256(before)
        candidate = self.apply(edits)
        changed = tuple(
            str(edit.get("rule_id")) for edit in edits
            if edit.get("operation") != "no_change" and edit.get("rule_id") is not None
        )
        receipt = {
            "parent_skill_hash": before_hash,
            "candidate_skill_hash": skill_sha256(candidate),
            "changed_rule_ids": list(changed),
            "edits": copy.deepcopy(edits),
            "skill_tokens_before": count(compile_initial_skill(before)),
            "skill_tokens_after": count(compile_initial_skill(candidate)),
            "budget_tokenizer": "o200k_base",
        }
        return candidate, receipt


def compile_initial_skill(payload: Mapping[str, Any]) -> str:
    """Render rules as a readable Markdown Strategy replacement."""

    entityhop_variant = any(
        action in str(rule["action_sequence"])
        for rule in payload["retrieval_policy"]
        for action in ("entity_hop", "linear_hop")
    )
    lines = [
        "## Strategy (formatted A-RAG skill)",
        ("These rules extend A-RAG's strategy with evidence-conditioned entity hopping." if entityhop_variant
         else "These rules are a formatted transcription of A-RAG's original Strategy instructions."),
        "",
    ]
    for rule in payload["retrieval_policy"]:
        lines.extend(
            [
                f"### {rule['rule_id']}: {rule['title']}",
                f"When: {rule['when']}",
                "Action sequence:",
            ]
        )
        lines.extend(f"{i}. {item}" for i, item in enumerate(rule["action_sequence"], start=1))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_formatted_prompt(original_prompt: str, payload: Mapping[str, Any]) -> str:
    """Replace only the official Strategy body; preserve tools and answer rules."""

    marker = "## Strategy"
    answer_marker = "## When Answering"
    start = original_prompt.find(marker)
    end = original_prompt.find(answer_marker)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("A-RAG prompt must contain Strategy and When Answering headings")
    return original_prompt[:start] + compile_initial_skill(payload) + "\n" + original_prompt[end:]


def skill_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
