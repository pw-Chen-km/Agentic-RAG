"""Instructed Qwen 27B action-selection probes, separate from natural runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.interface import get_interface_contract
from agentic_rag.agent.models import EpisodeState, Message
from agentic_rag.agent.providers.ollama import OllamaChatPolicy
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.substrate.storage import Substrate


CASES = (
    ("C0", "find_passages", False),
    ("C1", "find_sentences", False),
    ("C2", "follow_entity_to_passages", True),
    ("C3", "follow_entity_to_sentences", True),
    ("C5", "find_sentences", False),
    ("C5", "follow_entity_to_passages", True),
    ("C4", "find_sentences", False),
    ("C4", "follow_entity_to_sentences", True),
    ("A1", "find_sentences", False),
)


def action_name(action) -> str:
    if action.type == "SEARCH":
        return "find_passages" if action.target.value == "CHUNK" else "find_sentences"
    if action.type == "EXPAND":
        return ("follow_entity_to_passages" if action.kind.value.endswith("CHUNK")
                else "follow_entity_to_sentences")
    return "finish" if action.type == "FINISH" else str(action.type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    substrate = Substrate.open(args.substrate)
    scope_id = next(iter(substrate.doc_ids_by_scope))
    entity_id = sorted(substrate.entity_ids_by_scope[scope_id])[0]
    policy = OllamaChatPolicy(
        model="qwen3.8:27b-q4_K_M", host="http://127.0.0.1:11440",
        temperature=0, think=False, num_ctx=32768, max_output_tokens=512,
        timeout_seconds=600, max_retries=0,
    )
    rows = []
    for condition, requested, entity_visible in CASES:
        state = EpisodeState.initial()
        if entity_visible:
            state.visible_entity_ids.add(entity_id)
            state.semantic_memory_node_ids.append(entity_id)
            state.reference_registry.register(entity_id, "ENTITY")
        built = PolicyContextBuilder(
            substrate, interface_contract=get_interface_contract(condition),
        ).build(
            "Who starred in The Newcomers?", SkillDocument.load(args.skill),
            state, [], scope_id=scope_id,
        )
        available = [entry["function"]["name"] for entry in built.tool_definitions]
        if requested not in available:
            raise AssertionError(f"{condition}: {requested} was not available")
        arguments = (" with entity_ref E1 and query null" if entity_visible else
                     " with query 'The Newcomers film cast'")
        instruction = (f"For this action-format test, choose {requested}{arguments}. "
                       "Do not finish or explain.")
        try:
            decision = policy.decide(
                [*built.messages, Message(role="user", content=instruction)],
                decision_format=built.decision_format, tools=built.provider_tools,
            )
            selected = action_name(decision.action)
            row = {
                "condition": condition, "requested": requested, "selected": selected,
                "state": "synthetic E1 visible" if entity_visible else "initial",
                "available": available, "valid": selected == requested,
                "usage": {key: policy.last_usage_metadata.get(key) for key in (
                    "provider_input_tokens", "provider_output_tokens", "provider_total_tokens",
                    "tool_call_count", "native_tool_calling")},
            }
        except Exception as exc:
            row = {"condition": condition, "requested": requested,
                   "state": "synthetic E1 visible" if entity_visible else "initial",
                   "available": available, "valid": False,
                   "error_type": type(exc).__name__, "error_message": str(exc)}
        rows.append(row)
        report = {"probe_type": "instructed action selection, not natural policy uptake",
                  "model": "qwen3.8:27b-q4_K_M", "passed": all(item["valid"] for item in rows),
                  "cases": rows}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
