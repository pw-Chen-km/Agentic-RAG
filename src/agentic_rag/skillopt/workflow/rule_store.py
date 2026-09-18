"""Structured Skill rules used by Workflow-Aware SkillOpt v2.

The Agent still receives ordinary Markdown.  The optimizer edits this small
structured representation so that every change has a stable rule id and can be
audited without rewriting an entire Skill section.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SECTIONS = (
    "retrieval_policy",
    "recovery_policy",
    "answer_policy",
)
EDITABLE_BY_STAGE = {
    "retrieval": {"retrieval_policy", "recovery_policy"},
    "meta": {"retrieval_policy", "recovery_policy"},
    "answer": {"answer_policy"},
}
RULE_ID_RE = re.compile(r"^[RA][0-9]{2,}$")


def count_tokens(text: str, tokenizer: Callable[[str], int] | None = None) -> int:
    """Count tokens through an injected model tokenizer when available.

    The deterministic fallback is deliberately conservative and is used by
    offline tests.  Production runners may inject the configured Qwen
    tokenizer without changing the rule application API.
    """
    if tokenizer is not None:
        return int(tokenizer(text))
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"rule {field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class SkillRule:
    rule_id: str
    title: str
    when: str
    action_sequence: tuple[str, ...]
    stop_or_recovery: str
    exceptions: tuple[str, ...] = ()
    section: str = "retrieval_policy"
    editable_stages: tuple[str, ...] = ("retrieval", "meta")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, section: str | None = None) -> "SkillRule":
        rule_id = _text(value.get("rule_id"), "rule_id")
        if not RULE_ID_RE.fullmatch(rule_id):
            raise ValueError(f"invalid rule_id: {rule_id}")
        resolved_section = str(value.get("section") or section or "")
        if resolved_section not in SECTIONS:
            raise ValueError(f"invalid rule section: {resolved_section}")
        actions = value.get("action_sequence")
        if not isinstance(actions, list) or not actions or any(not isinstance(x, str) or not x.strip() for x in actions):
            raise ValueError("action_sequence must be a non-empty list of strings")
        exceptions = value.get("exceptions", [])
        editable = value.get("editable_stages")
        if editable is None:
            editable = ["answer"] if resolved_section == "answer_policy" else ["retrieval", "meta"]
        elif isinstance(editable, tuple):
            editable = list(editable)
        if not isinstance(exceptions, list) or any(not isinstance(x, str) or not x.strip() for x in exceptions):
            raise ValueError("exceptions must be a list of strings")
        if not isinstance(editable, list) or any(x not in {"retrieval", "meta", "answer"} for x in editable):
            raise ValueError("editable_stages contains an invalid stage")
        if resolved_section == "answer_policy" and not editable:
            editable = ["answer"]
        if resolved_section == "answer_policy" and set(editable) != {"answer"}:
            raise ValueError("answer rules are editable only in the answer stage")
        if resolved_section != "answer_policy" and "answer" in editable:
            raise ValueError("retrieval rules cannot be edited in the answer stage")
        return cls(
            rule_id=rule_id,
            title=_text(value.get("title"), "title"),
            when=_text(value.get("when"), "when"),
            action_sequence=tuple(_text(x, "action_sequence item") for x in actions),
            stop_or_recovery=_text(value.get("stop_or_recovery"), "stop_or_recovery"),
            exceptions=tuple(_text(x, "exception") for x in exceptions),
            section=resolved_section,
            editable_stages=tuple(str(x) for x in editable),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "when": self.when,
            "action_sequence": list(self.action_sequence),
            "stop_or_recovery": self.stop_or_recovery,
            "exceptions": list(self.exceptions),
            "section": self.section,
            "editable_stages": list(self.editable_stages),
        }

    def normalized_signature(self) -> tuple[Any, ...]:
        normalize = lambda value: " ".join(str(value).casefold().split())
        return (
            normalize(self.title),
            normalize(self.when),
            tuple(normalize(x) for x in self.action_sequence),
            normalize(self.stop_or_recovery),
            tuple(normalize(x) for x in self.exceptions),
            self.section,
        )

    def render(self) -> str:
        lines = [f"### [{self.rule_id}] {self.title}", f"- When: {self.when}", "- Action sequence:"]
        lines.extend(f"  {index}. {action}" for index, action in enumerate(self.action_sequence, 1))
        lines.append(f"- Stop or recovery: {self.stop_or_recovery}")
        if self.exceptions:
            lines.append("- Exceptions:")
            lines.extend(f"  - {item}" for item in self.exceptions)
        return "\n".join(lines)


@dataclass(frozen=True)
class RuleStore:
    version: str
    fixed_runtime_guidance: str
    fixed_answer_contract: str
    blocks: dict[str, tuple[SkillRule, ...]]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuleStore":
        version = _text(value.get("version"), "version")
        fixed_runtime = _text(value.get("fixed_runtime_guidance"), "fixed_runtime_guidance")
        fixed_contract = _text(value.get("fixed_answer_contract"), "fixed_answer_contract")
        raw_blocks = value.get("blocks")
        if not isinstance(raw_blocks, Mapping):
            raise ValueError("blocks must be an object")
        blocks: dict[str, tuple[SkillRule, ...]] = {}
        seen: set[str] = set()
        for section in SECTIONS:
            raw_rules = raw_blocks.get(section, [])
            if not isinstance(raw_rules, list):
                raise ValueError(f"{section} must be a list")
            parsed = tuple(SkillRule.from_mapping(item, section=section) for item in raw_rules)
            for rule in parsed:
                if rule.rule_id in seen:
                    raise ValueError(f"duplicate rule_id: {rule.rule_id}")
                seen.add(rule.rule_id)
                if section == "answer_policy" and rule.rule_id[:1] != "A":
                    raise ValueError(f"answer rule must use A id: {rule.rule_id}")
                if section != "answer_policy" and rule.rule_id[:1] != "R":
                    raise ValueError(f"retrieval rule must use R id: {rule.rule_id}")
            blocks[section] = parsed
        return cls(version, fixed_runtime, fixed_contract, blocks)

    @classmethod
    def from_json(cls, path: Path) -> "RuleStore":
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_markdown(cls, text: str, *, version: str = "workflow-skill-v2") -> "RuleStore":
        """Read the deterministic Markdown emitted by :meth:`render_markdown`.

        This is intentionally a strict reader for the generated format, not a
        general Markdown parser.  It provides an offline round-trip check that
        the Agent-facing file still represents the JSON rule source.
        """
        def block(name: str, heading: str) -> tuple[str, list[dict[str, Any]]]:
            start = f"<!-- {name.upper()}_START -->"
            end = f"<!-- {name.upper()}_END -->"
            if start not in text or end not in text:
                raise ValueError(f"missing Markdown block: {name}")
            body = text.split(start, 1)[1].split(end, 1)[0]
            marker = f"## {heading}\n"
            if marker not in body:
                raise ValueError(f"missing Markdown heading: {heading}")
            content = body.split(marker, 1)[1].strip()
            if not content:
                return "", []
            chunks = re.split(r"\n\n(?=### \[[RA]\d+\] )", content)
            parsed: list[dict[str, Any]] = []
            for chunk in chunks:
                lines = chunk.splitlines()
                match = re.fullmatch(r"### \[([RA]\d+)\] (.+)", lines[0]) if lines else None
                if not match:
                    raise ValueError(f"invalid generated rule heading: {lines[:1]}")
                rule_id, title = match.groups()
                if len(lines) < 4 or not lines[1].startswith("- When: ") or lines[2] != "- Action sequence:":
                    raise ValueError(f"invalid generated rule: {rule_id}")
                when = lines[1][len("- When: "):].strip()
                actions: list[str] = []
                cursor = 3
                while cursor < len(lines) and re.match(r"  \d+\. .+", lines[cursor]):
                    actions.append(re.sub(r"^  \d+\. ", "", lines[cursor]))
                    cursor += 1
                if not actions or cursor >= len(lines) or not lines[cursor].startswith("- Stop or recovery: "):
                    raise ValueError(f"invalid action sequence: {rule_id}")
                stop = lines[cursor][len("- Stop or recovery: "):].strip()
                cursor += 1
                exceptions: list[str] = []
                if cursor < len(lines):
                    if lines[cursor] != "- Exceptions:":
                        raise ValueError(f"invalid exceptions block: {rule_id}")
                    exceptions = [line[4:].strip() for line in lines[cursor + 1:]
                                  if line.startswith("  - ")]
                    if len(exceptions) != len(lines) - cursor - 1:
                        raise ValueError(f"invalid exception line: {rule_id}")
                parsed.append({"rule_id": rule_id, "title": title, "when": when,
                               "action_sequence": actions, "stop_or_recovery": stop,
                               "exceptions": exceptions, "section": name,
                               "editable_stages": ["answer"] if name == "answer_policy"
                               else ["retrieval", "meta"]})
            return "", parsed

        fixed_runtime_start = "<!-- FIXED_RUNTIME_GUIDANCE_START -->"
        fixed_runtime_end = "<!-- FIXED_RUNTIME_GUIDANCE_END -->"
        fixed_answer_start = "<!-- FIXED_ANSWER_CONTRACT_START -->"
        fixed_answer_end = "<!-- FIXED_ANSWER_CONTRACT_END -->"
        if fixed_runtime_start not in text or fixed_runtime_end not in text:
            raise ValueError("missing fixed runtime block")
        if fixed_answer_start not in text or fixed_answer_end not in text:
            raise ValueError("missing fixed answer block")
        runtime_body = text.split(fixed_runtime_start, 1)[1].split(fixed_runtime_end, 1)[0]
        answer_body = text.split(fixed_answer_start, 1)[1].split(fixed_answer_end, 1)[0]
        fixed_runtime = runtime_body.split("## Fixed runtime guidance\n", 1)[1].strip()
        fixed_answer = answer_body.split("## Fixed answer contract\n", 1)[1].strip()
        blocks = {section: block(section, heading)[1] for section, heading in (
            ("retrieval_policy", "Retrieval policy"),
            ("recovery_policy", "Recovery policy"),
            ("answer_policy", "Answer policy"),
        )}
        return cls.from_mapping({"version": version,
                                 "fixed_runtime_guidance": fixed_runtime,
                                 "fixed_answer_contract": fixed_answer,
                                 "blocks": blocks})

    def to_mapping(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fixed_runtime_guidance": self.fixed_runtime_guidance,
            "fixed_answer_contract": self.fixed_answer_contract,
            "blocks": {
                section: [rule.to_mapping() for rule in self.blocks.get(section, ())]
                for section in SECTIONS
            },
        }

    def json_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_mapping(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def all_rules(self) -> tuple[SkillRule, ...]:
        return tuple(rule for section in SECTIONS for rule in self.blocks.get(section, ()))

    def rule_index(self) -> dict[str, SkillRule]:
        return {rule.rule_id: rule for rule in self.all_rules()}

    def optimizer_view(self, stage: str) -> dict[str, Any]:
        allowed = EDITABLE_BY_STAGE.get(stage, set())
        return {
            "version": self.version,
            "editable_sections": sorted(allowed),
            "rules": [
                rule.to_mapping()
                for section in SECTIONS
                for rule in self.blocks.get(section, ())
                if section in allowed
            ],
        }

    def trainable_token_count(self, tokenizer: Callable[[str], int] | None = None) -> int:
        return count_tokens(
            "\n".join(rule.render() for section in ("retrieval_policy", "recovery_policy") for rule in self.blocks.get(section, ())),
            tokenizer,
        )

    def render_markdown(self) -> str:
        def block(name: str, heading: str) -> str:
            rules = self.blocks.get(name, ())
            body = "\n\n".join(rule.render() for rule in rules)
            return f"<!-- {name.upper()}_START -->\n## {heading}\n{body}\n<!-- {name.upper()}_END -->"

        return "\n\n".join((
            f"# Agentic-RAG Workflow Skill ({self.version})",
            f"<!-- FIXED_RUNTIME_GUIDANCE_START -->\n## Fixed runtime guidance\n{self.fixed_runtime_guidance}\n<!-- FIXED_RUNTIME_GUIDANCE_END -->",
            block("retrieval_policy", "Retrieval policy"),
            block("recovery_policy", "Recovery policy"),
            block("answer_policy", "Answer policy"),
            f"<!-- FIXED_ANSWER_CONTRACT_START -->\n## Fixed answer contract\n{self.fixed_answer_contract}\n<!-- FIXED_ANSWER_CONTRACT_END -->",
        )) + "\n"

    def apply_edits(
        self,
        edits: Sequence[Mapping[str, Any]],
        *,
        stage: str,
        max_edits: int = 2,
        max_delta_tokens: int = 250,
        max_trainable_tokens: int = 1500,
        tokenizer: Callable[[str], int] | None = None,
    ) -> tuple["RuleStore", dict[str, Any]]:
        if stage not in EDITABLE_BY_STAGE:
            raise ValueError(f"unknown stage: {stage}")
        if len(edits) > max_edits:
            raise ValueError(f"too many edits: {len(edits)} > {max_edits}")
        if not edits:
            return self, {"applied": [], "changed_rule_ids": [], "no_change": True}
        index = self.rule_index()
        blocks = {name: list(rules) for name, rules in self.blocks.items()}
        applied: list[dict[str, Any]] = []
        touched: set[str] = set()
        next_number = max([int(rule.rule_id[1:]) for rule in self.all_rules() if rule.rule_id[1:].isdigit()] or [0]) + 1
        for raw in edits:
            if not isinstance(raw, Mapping):
                raise ValueError("edit must be an object")
            operation = raw.get("operation")
            if operation not in {"add", "replace", "delete"}:
                raise ValueError(f"invalid operation: {operation}")
            rule_id = raw.get("rule_id")
            if rule_id is not None and (not isinstance(rule_id, str) or not RULE_ID_RE.fullmatch(rule_id)):
                raise ValueError("invalid edit rule_id")
            if operation == "add":
                section = raw.get("section")
                if section not in EDITABLE_BY_STAGE[stage]:
                    raise ValueError("add targets a forbidden section")
                if rule_id is None:
                    prefix = "A" if section == "answer_policy" else "R"
                    rule_id = f"{prefix}{next_number:02d}"
                    next_number += 1
                if rule_id in index or rule_id in touched:
                    raise ValueError(f"add uses existing rule_id: {rule_id}")
                if (section == "answer_policy" and not rule_id.startswith("A")) or \
                   (section != "answer_policy" and not rule_id.startswith("R")):
                    raise ValueError("rule_id prefix does not match section")
                rule_value = dict(raw.get("rule") or {})
                rule_value["rule_id"] = rule_id
                rule_value["section"] = section
                rule = SkillRule.from_mapping(rule_value, section=section)
                if stage not in rule.editable_stages:
                    raise ValueError("rule is not editable in this stage")
                blocks[section].append(rule)
                index[rule_id] = rule
                section_name = section
            elif operation == "replace":
                if rule_id not in index:
                    raise ValueError(f"replace target not found: {rule_id}")
                old = index[rule_id]
                if old.section not in EDITABLE_BY_STAGE[stage] or stage not in old.editable_stages:
                    raise ValueError("replace targets a forbidden rule")
                rule_value = dict(raw.get("rule") or {})
                rule_value["rule_id"] = rule_id
                rule_value["section"] = old.section
                rule = SkillRule.from_mapping(rule_value, section=old.section)
                if stage not in rule.editable_stages:
                    raise ValueError("rule is not editable in this stage")
                rules = blocks[old.section]
                blocks[old.section] = [rule if item.rule_id == rule_id else item for item in rules]
                index[rule_id] = rule
                section_name = old.section
            else:
                if rule_id not in index:
                    raise ValueError(f"delete target not found: {rule_id}")
                old = index[rule_id]
                if old.section not in EDITABLE_BY_STAGE[stage] or stage not in old.editable_stages:
                    raise ValueError("delete targets a forbidden rule")
                blocks[old.section] = [item for item in blocks[old.section] if item.rule_id != rule_id]
                del index[rule_id]
                section_name = old.section
            if rule_id in touched:
                raise ValueError(f"rule modified more than once: {rule_id}")
            touched.add(rule_id)
            applied.append({"operation": operation, "rule_id": rule_id, "section": section_name})

        candidate = replace(self, blocks={name: tuple(rules) for name, rules in blocks.items()})
        signatures: dict[tuple[Any, ...], str] = {}
        for rule in candidate.all_rules():
            signature = rule.normalized_signature()
            if signature in signatures and signatures[signature] != rule.rule_id:
                raise ValueError(f"duplicate semantic rule: {signatures[signature]} and {rule.rule_id}")
            signatures[signature] = rule.rule_id
        before_tokens = self.trainable_token_count(tokenizer)
        after_tokens = candidate.trainable_token_count(tokenizer)
        delta = max(0, after_tokens - before_tokens)
        if delta > max_delta_tokens:
            raise ValueError(f"edit token delta exceeds limit: {delta} > {max_delta_tokens}")
        if after_tokens > max_trainable_tokens:
            raise ValueError(f"trainable Skill token limit exceeded: {after_tokens} > {max_trainable_tokens}")
        return candidate, {
            "applied": applied,
            "changed_rule_ids": sorted(touched),
            "no_change": False,
            "skill_tokens_before": before_tokens,
            "skill_tokens_after": after_tokens,
            "skill_token_delta": after_tokens - before_tokens,
        }
