"""Run deterministic interface calibration before a live policy run.

This gate checks the compiled condition contracts and single-decision schemas
without calling an LLM.  It is deliberately separate from the live calibration
performed on the target server, where each provider tool is executed against a
real substrate.  The report makes that distinction explicit instead of
mistaking a static check for an execution success rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.interface import get_interface_contract
from agentic_rag.agent.models import EpisodeState, Message
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.agent.tool_calling import tool_schema_sha256
from agentic_rag.substrate.storage import Substrate


CONDITIONS = ("C0", "C1", "C2", "C3", "C5", "C4", "A1")


def calibrate(
    substrate_path: Path,
    conditions: tuple[str, ...],
    *,
    live: bool = False,
    model: str = "qwen3.5:4b",
    host: str = "http://localhost:11434",
    timeout_seconds: float = 600.0,
    max_output_tokens: int = 512,
    repetitions: int = 1,
    seed: int | None = 20260805,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    substrate = Substrate.open(substrate_path)
    study_skill = SkillDocument.load(
        Path(__file__).resolve().parents[1] / "skills" / "interface_study.md"
    )
    rows: list[dict[str, Any]] = []
    static_valid = 0
    scope_id = next(iter(substrate.doc_ids_by_scope))
    entity_id = sorted(substrate.entity_ids_by_scope[scope_id])[0]
    for name in conditions:
        policy = None
        if live:
            from agentic_rag.agent.providers.ollama import OllamaChatPolicy

            # Formal runs create one provider instance per condition.  Keep
            # calibration's model lifecycle identical so a previous condition
            # cannot affect the next condition's tool-call probe.
            policy = OllamaChatPolicy(
                model=model,
                host=host,
                temperature=0,
                think=False,
                num_ctx=32768,
                timeout_seconds=timeout_seconds,
                max_retries=0,
                max_output_tokens=max_output_tokens,
                seed=seed,
            )
        contract = get_interface_contract(name)
        builder = PolicyContextBuilder(
            substrate, interface_contract=contract
        )
        initial = builder.build(
            "Calibration question",
            study_skill,
            EpisodeState.initial(),
            [],
            scope_id=scope_id,
        )
        initial_names = [str(item["function"]["name"]) for item in initial.tool_definitions]
        expected_initial = {"find_passages", "finish"}
        if contract.global_sentence_search:
            expected_initial.add("find_sentences")
        if set(initial_names) != expected_initial:
            raise ValueError(
                f"{name}: initial tool registry mismatch: {initial_names}; "
                f"expected {sorted(expected_initial)}"
            )

        # Entity actions are dynamic: they appear only after a source span has
        # exposed an entity.  Exercise that second state explicitly; checking
        # only EpisodeState.initial() would incorrectly report C2/C3/C4/C5 as
        # missing their navigation tools.
        entity_state = EpisodeState.initial()
        entity_state.visible_entity_ids.add(entity_id)
        entity_state.semantic_memory_node_ids.append(entity_id)
        entity_state.reference_registry.register(entity_id, "ENTITY")
        entity_visible = builder.build(
            "Calibration question",
            study_skill,
            entity_state,
            [],
            scope_id=scope_id,
        )
        entity_names = [
            str(item["function"]["name"])
            for item in entity_visible.tool_definitions
        ]
        expected_entity = set(expected_initial)
        if contract.entity_continuation.value == "chunk":
            expected_entity.add("follow_entity_to_passages")
        elif contract.entity_continuation.value == "sentence":
            expected_entity.add("follow_entity_to_sentences")
        if set(entity_names) != expected_entity:
            raise ValueError(
                f"{name}: entity-visible tool registry mismatch: {entity_names}; "
                f"expected {sorted(expected_entity)}"
            )
        static_valid += 1
        row: dict[str, Any] = {
                "condition": name,
                "initial_tool_names": initial_names,
                "initial_tool_schema_sha256": tool_schema_sha256(initial.tool_definitions),
                "initial_decision_schema_sha256": initial.decision_schema_sha256,
                "entity_visible_tool_names": entity_names,
                "entity_visible_tool_schema_sha256": tool_schema_sha256(
                    entity_visible.tool_definitions
                ),
                "entity_visible_decision_schema_sha256": entity_visible.decision_schema_sha256,
                "schema_valid": True,
                "execution_status": "not_run",
        }
        if policy is not None:
            live_checks: list[dict[str, Any]] = []

            def live_check(
                label: str,
                built: Any,
                instruction: str,
                repetition: int,
            ) -> None:
                available = {
                    str(item["function"]["name"])
                    for item in built.tool_definitions
                }
                try:
                    decision = policy.decide(
                        [
                            *built.messages,
                            Message(role="user", content=instruction),
                        ],
                        decision_format=built.decision_format,
                        tools=built.provider_tools,
                    )
                    selected = str(decision.action.type)
                    if selected == "SEARCH":
                        selected = {
                            "DENSE:CHUNK": "find_passages",
                            "DENSE:SENTENCE": "find_sentences",
                        }.get(
                            f"{decision.action.method.value}:{decision.action.target.value}",
                            selected,
                        )
                    elif selected == "EXPAND":
                        selected = {
                            "ENTITY_MENTIONED_IN_CHUNK": "follow_entity_to_passages",
                            "ENTITY_MENTIONED_IN_SENTENCE": "follow_entity_to_sentences",
                        }.get(decision.action.kind, selected)
                    elif selected == "FINISH":
                        selected = "finish"
                    live_checks.append(
                        {
                            "phase": label,
                            "repetition": repetition,
                            "provider_status": "ok",
                            "selected_tool": selected,
                            "selected_tool_allowed": selected in available,
                            "decision_count": policy.last_usage_metadata.get("decision_count"),
                            "selected_action": policy.last_usage_metadata.get("selected_action"),
                            "native_tool_calling": policy.last_usage_metadata.get(
                                "native_tool_calling"
                            ),
                            "tool_call_count": policy.last_usage_metadata.get("tool_call_count"),
                            "usage": {
                                "input_tokens": policy.last_usage_metadata.get(
                                    "provider_input_tokens"
                                ),
                                "output_tokens": policy.last_usage_metadata.get(
                                    "provider_output_tokens"
                                ),
                            },
                        }
                    )
                except Exception as exc:  # preserve provider failure, never score as zero
                    live_checks.append(
                        {
                            "phase": label,
                            "repetition": repetition,
                            "provider_status": "error",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )

            for repetition in range(1, repetitions + 1):
                live_check(
                    "initial",
                    initial,
                    "Use one available retrieval tool to find evidence. Do not explain.",
                    repetition,
                )
                live_check(
                    "entity_visible",
                    entity_visible,
                    "Use the entity-navigation tool for the displayed E1 reference if one is available. Do not explain.",
                    repetition,
                )
            row["live_checks"] = live_checks
        rows.append(row)

    live_checks = [
        check
        for row in rows
        for check in row.get("live_checks", [])
    ]
    protocol_valid_checks = [
        check
        for check in live_checks
        if check.get("provider_status") == "ok"
        and check.get("native_tool_calling") is True
        and check.get("tool_call_count") == 1
        and check.get("selected_tool_allowed") is True
        and check.get("selected_action") == check.get("selected_tool")
    ]
    protocol_valid_rate = (
        len(protocol_valid_checks) / len(live_checks) if live_checks else None
    )
    if live:
        gate = (
            "live_protocol_smoke_passed"
            if protocol_valid_rate == 1.0
            else "live_protocol_smoke_failed"
        )
    else:
        gate = "static_contracts_passed"
    return {
        "calibration_version": "interface-study-native-tool-calling-v1",
        "substrate": substrate.root.as_posix(),
        "conditions": rows,
        "schema_valid_rate": static_valid / len(conditions) if conditions else None,
        "live_protocol_check_count": len(live_checks),
        "live_protocol_valid_count": len(protocol_valid_checks),
        "protocol_valid_rate": protocol_valid_rate,
        "execution_success_rate": None,
        "execution_status": (
            "not_measured_by_protocol_probe"
            if live
            else "deferred_to_live_workflow_smoke"
        ),
        "live_provider": {
            "enabled": live,
            "model": model if live else None,
            "host": host if live else None,
            "think": False if live else None,
            "repetitions_per_state": repetitions if live else None,
            "seed": seed if live else None,
        },
        "gate": gate,
        "gate_scope": (
            "small provider-format smoke; does not certify 99% reliability or retrieval execution"
            if live
            else "static condition and schema compilation only"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--live", action="store_true", help="also call the configured local Ollama provider")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="number of live provider probes per condition and context state",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = calibrate(
        args.substrate,
        tuple(args.conditions),
        live=args.live,
        model=args.model,
        host=args.host,
        timeout_seconds=args.timeout_seconds,
        max_output_tokens=args.max_output_tokens,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
