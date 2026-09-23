"""Option definitions and the built-in v1 seed catalog.

The catalog is the only source for option descriptions.  Markdown is rendered
for the model only after an option has been selected; the selector receives
only a compact summary (progressive disclosure).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class OptionSpec:
    option_id: str
    name: str
    goal: str
    initiation: tuple[str, ...]
    primitive_actions: tuple[str, ...]
    policy: dict[str, str]
    termination: dict[str, str]
    interrupt_when: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "OptionSpec":
        required = {
            "option_id", "name", "goal", "initiation", "primitive_actions",
            "policy", "termination", "interrupt_when",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"option is missing required field(s): {', '.join(missing)}")
        option_id = str(value["option_id"]).strip()
        if not option_id or option_id == "FALLBACK" or " " in option_id:
            raise ValueError(f"invalid option_id: {option_id!r}")
        actions = tuple(str(item).upper() for item in value["primitive_actions"])
        allowed = {"SEARCH", "EXPAND", "READ", "FINISH"}
        if not actions or not set(actions) <= allowed:
            raise ValueError(f"invalid primitive_actions for {option_id}")
        if option_id == "O5_ANSWER" and "FINISH" not in actions:
            raise ValueError("O5_ANSWER must allow FINISH")
        if option_id != "O5_ANSWER" and "FINISH" in actions:
            raise ValueError(f"only O5_ANSWER may allow FINISH: {option_id}")
        return cls(
            option_id=option_id,
            name=str(value["name"]),
            goal=str(value["goal"]),
            initiation=tuple(str(item) for item in value["initiation"]),
            primitive_actions=actions,
            policy={str(k): str(v) for k, v in dict(value["policy"]).items()},
            termination={str(k): str(v) for k, v in dict(value["termination"]).items()},
            interrupt_when=str(value["interrupt_when"]),
        )

    def as_dict(self) -> dict[str, Any]:
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

    def summary(self) -> dict[str, Any]:
        return {
            "option_id": self.option_id,
            "name": self.name,
            "goal": self.goal,
            "initiation": list(self.initiation),
            "primitive_actions": list(self.primitive_actions),
        }

    def markdown(self) -> str:
        lines = [
            f"### {self.option_id}: {self.name}",
            f"Goal: {self.goal}",
            "Use this option when:",
            *[f"- {item}" for item in self.initiation],
            "Primitive actions allowed: " + ", ".join(self.primitive_actions),
            "Policy after each observation:",
            *[f"- If {key}: {value}" for key, value in self.policy.items()],
            "End this option when:",
            *[f"- {key}: {value}" for key, value in self.termination.items()],
            f"Interrupt when: {self.interrupt_when}",
        ]
        return "\n".join(lines)


class OptionCatalog:
    """Immutable-ish catalog with deterministic rendering and hashing."""

    FALLBACK_ID = "FALLBACK"

    def __init__(
        self,
        options: tuple[OptionSpec, ...],
        *,
        source_path: str | None = None,
        version: str = "options-v1",
        fixed_runtime_guidance: str = "Choose the option whose initiation conditions match the current state.",
        fixed_answer_contract: str = "Answer only with legal evidence and do not guess.",
    ) -> None:
        ids = [item.option_id for item in options]
        if len(ids) != len(set(ids)):
            raise ValueError("option_id values must be unique")
        if any(item.option_id == self.FALLBACK_ID for item in options):
            raise ValueError("FALLBACK is reserved for the runtime")
        self.options = options
        self.source_path = source_path
        self.version = version
        self.fixed_runtime_guidance = fixed_runtime_guidance
        self.fixed_answer_contract = fixed_answer_contract
        self._by_id = {item.option_id: item for item in options}

    @classmethod
    def load(cls, path: str | Path) -> "OptionCatalog":
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        metadata: dict[str, Any] = raw if isinstance(raw, dict) else {}
        if isinstance(raw, dict) and "options" in raw:
            raw = raw["options"]
        if not isinstance(raw, list):
            raise ValueError("options file must contain a JSON list or {options: [...]} ")
        # FALLBACK is a controller-owned pseudo-option.  Accept it in the
        # shared JSON file for SkillOpt provenance, but do not let it become a
        # mutable procedural option in the runtime catalog.
        raw = [item for item in raw if str(item.get("option_id", "")) != cls.FALLBACK_ID]
        return cls(
            tuple(OptionSpec.from_mapping(item) for item in raw),
            source_path=str(source),
            version=str(metadata.get("version", "options-v1")),
            fixed_runtime_guidance=str(metadata.get("fixed_runtime_guidance", "Choose the option whose initiation conditions match the current state.")),
            fixed_answer_contract=str(metadata.get("fixed_answer_contract", "Answer only with legal evidence and do not guess.")),
        )

    @classmethod
    def default(cls) -> "OptionCatalog":
        return cls(tuple(OptionSpec.from_mapping(item) for item in _DEFAULT_OPTIONS))

    def get(self, option_id: str) -> OptionSpec:
        try:
            return self._by_id[option_id]
        except KeyError as exc:
            raise KeyError(f"unknown option: {option_id}") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(item.option_id for item in self.options)

    def summaries(self, option_ids: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        selected = option_ids or self.ids()
        return [self.get(item).summary() for item in selected]

    def render_markdown(self) -> str:
        return (
            f"# Agentic-RAG Options Skill ({self.version})\n\n"
            "## Fixed runtime guidance\n"
            + self.fixed_runtime_guidance
            + "\n\n## Options\n"
            + "\n\n".join(item.markdown() for item in self.options)
            + "\n\n## Fixed answer contract\n"
            + self.fixed_answer_contract
        )

    def sha256(self) -> str:
        payload = json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fixed_runtime_guidance": self.fixed_runtime_guidance,
            "fixed_answer_contract": self.fixed_answer_contract,
            "options": [item.as_dict() for item in self.options],
        }


_DEFAULT_OPTIONS: tuple[dict[str, Any], ...] = (
    {
        "option_id": "O1_START_SEARCH", "name": "Start search",
        "goal": "在幾乎沒有可用證據時找到第一個可追查線索",
        "initiation": ["目前沒有可用的句子或 Chunk", "問題仍有未解決部分"],
        "primitive_actions": ["SEARCH"],
        "policy": {
            "direct_evidence_found": "回報 COMPLETE，交回 Option Selector",
            "new_bridge_found": "保留線索，必要時繼續搜尋",
            "unread_chunk_available": "回報 COMPLETE，讓 Option Selector 選處理線索的 option",
            "no_progress": "回報 BLOCKED，不要重複相似的搜尋",
        },
        "termination": {
            "subgoal_complete": "已有可追查的資料或合法直接證據",
            "blocked": "搜尋沒有新增可用資料",
            "budget_low": "剩餘預算不足以安全繼續",
        },
        "interrupt_when": "已有更適合處理現有證據的 option",
    },
    {
        "option_id": "O2_RESOLVE_FACT", "name": "Resolve fact",
        "goal": "補足問題目前缺少的直接事實",
        "initiation": ["已有可用線索或證據", "仍缺少可直接回答的事實"],
        "primitive_actions": ["SEARCH", "EXPAND", "READ"],
        "policy": {
            "direct_evidence_found": "回報 COMPLETE",
            "new_bridge_found": "把線索交回 Option Selector，不把它當成已完成事實",
            "unread_chunk_available": "若 Chunk 可能包含缺口，先 READ",
            "no_progress": "回報 BLOCKED，換一條合法路徑",
        },
        "termination": {
            "subgoal_complete": "缺少的直接事實已有合法證據",
            "blocked": "目前候選路徑沒有新進展",
            "budget_low": "剩餘預算不足以安全繼續",
        },
        "interrupt_when": "發現需要沿中間線索追查或需要恢復搜尋",
    },
    {
        "option_id": "O3_RESOLVE_BRIDGE", "name": "Resolve bridge",
        "goal": "沿著已取得的中間線索取得下一個必要資訊",
        "initiation": ["目前已有可追查的 Entity、Sentence 或 Chunk", "問題仍有未解決部分"],
        "primitive_actions": ["SEARCH", "EXPAND", "READ"],
        "policy": {
            "direct_evidence_found": "回報 COMPLETE，交回 Option Selector",
            "new_bridge_found": "沿新線索繼續處理",
            "unread_chunk_available": "評估是否 READ",
            "no_progress": "回報 BLOCKED",
        },
        "termination": {
            "subgoal_complete": "下一個 subgoal 已取得合法證據",
            "blocked": "目前橋接線索沒有新進展",
            "budget_low": "剩餘預算不足以安全繼續",
        },
        "interrupt_when": "另一個 option 比目前 option 更適合",
    },
    {
        "option_id": "O4_RECOVER", "name": "Recover",
        "goal": "處理空結果、重複、錯誤或沒有新資料並找到可行路徑",
        "initiation": ["上一個 action 空結果、失敗、重複或沒有進展"],
        "primitive_actions": ["SEARCH", "EXPAND", "READ"],
        "policy": {
            "new_path_found": "回報 COMPLETE，交回 Option Selector",
            "unread_chunk_available": "若仍有可讀資料，優先評估 READ",
            "no_progress": "回報 BLOCKED，不要繼續重複相似 action",
        },
        "termination": {
            "subgoal_complete": "找到可繼續的新路徑",
            "blocked": "沒有可用的恢復路徑",
            "budget_low": "剩餘預算不足以安全繼續",
        },
        "interrupt_when": "已知缺口需要直接事實或橋接追查",
    },
    {
        "option_id": "O5_ANSWER", "name": "Answer",
        "goal": "選擇合法證據並形成最終答案",
        "initiation": ["已有一個或更多合法可引用證據", "不再需要新的 retrieval subgoal"],
        "primitive_actions": ["FINISH"],
        "policy": {
            "evidence_sufficient": "選擇支持答案的合法 evidence refs 並輸出 FINISH",
            "evidence_insufficient": "回報 BLOCKED，交回 Option Selector",
        },
        "termination": {
            "subgoal_complete": "FINISH 已執行，episode 結束",
            "blocked": "證據不足以安全回答",
            "budget_low": "剩餘預算不足以安全繼續",
        },
        "interrupt_when": "沒有合法可引用證據",
    },
)
