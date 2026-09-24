"""Run a small natural action-selection probe against a local Ollama model.

This probe does not execute retrieval.  It checks that the model can select one
operation from the v6.1 prompt in an initial observation and in an observation
containing real source text plus one visible entity.  It is therefore useful
when the stored dense index must not yet be used for a full smoke run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_rag.agent.context import PolicyContextBuilder
from agentic_rag.agent.interface import get_interface_contract
from agentic_rag.agent.models import (
    EpisodeState,
    ExpandAction,
    FinishAction,
    SearchAction,
)
from agentic_rag.agent.providers.ollama import OllamaChatPolicy
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.substrate.storage import Substrate


CONDITIONS = ("C0", "C1", "C2", "C3", "C5", "C4", "A1")


def action_name(action: object) -> str:
    if isinstance(action, SearchAction):
        return "find_passages" if action.target.value == "CHUNK" else "find_sentences"
    if isinstance(action, ExpandAction):
        return (
            "follow_entity_to_passages"
            if action.kind.value.endswith("CHUNK")
            else "follow_entity_to_sentences"
        )
    if isinstance(action, FinishAction):
        return "finish"
    return str(getattr(action, "type", type(action).__name__))


def action_arguments(action: object) -> dict[str, object]:
    if isinstance(action, SearchAction):
        return {"query": action.query}
    if isinstance(action, ExpandAction):
        return {"entity_ref": action.source_ref, "query": action.query}
    if isinstance(action, FinishAction):
        return {"evidence_refs": list(action.evidence_refs)}
    return {}


def build_visible_source_state(substrate: Substrate, scope_id: str) -> EpisodeState:
    state = EpisodeState.initial()
    selected: tuple[str, object, object] | None = None
    for candidate_chunk_id in sorted(substrate.chunk_ids_by_scope[scope_id]):
        for candidate_sentence in substrate.sentences_by_chunk[candidate_chunk_id]:
            candidate_mention = next(
                (
                    item
                    for item in substrate.mentions
                    if item.sentence_id == candidate_sentence.sentence_id
                ),
                None,
            )
            if candidate_mention is not None:
                selected = (candidate_chunk_id, candidate_sentence, candidate_mention)
                break
        if selected is not None:
            break
    if selected is None:
        raise ValueError("the substrate has no sentence with an entity mention")
    chunk_id, sentence, mention = selected
    state.visible_chunk_ids.add(chunk_id)
    state.visible_passage_ids.add(chunk_id)
    state.read_chunk_ids.add(chunk_id)
    state.visible_sentence_ids.update(
        item.sentence_id for item in substrate.sentences_by_chunk[chunk_id]
    )
    state.eligible_sentence_ids.update(
        item.sentence_id for item in substrate.sentences_by_chunk[chunk_id]
    )
    state.visible_entity_ids.add(mention.entity_id)
    state.semantic_memory_node_ids.extend([chunk_id, mention.entity_id])
    state.reference_registry.register(chunk_id, "CHUNK")
    state.reference_registry.register(mention.entity_id, "ENTITY")
    return state


def run_probe(
    substrate: Substrate,
    question: str,
    skill: SkillDocument,
    *,
    model: str,
    host: str,
) -> dict[str, object]:
    scope_id = next(iter(substrate.doc_ids_by_scope))
    policy = OllamaChatPolicy(
        model=model,
        host=host,
        temperature=0,
        think=False,
        num_ctx=32768,
        max_output_tokens=512,
        timeout_seconds=600,
        max_retries=0,
    )
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for state_name, state in (
            ("initial", EpisodeState.initial()),
            ("visible_source_entity", build_visible_source_state(substrate, scope_id)),
        ):
            built = PolicyContextBuilder(
                substrate,
                interface_contract=get_interface_contract(condition),
            ).build(question, skill, state, [], scope_id=scope_id)
            available = [
                item["properties"]["name"]["const"] for item in built.decision_format.model_json_schema()
                .get("$defs", {})
                .values()
                if "name" in item.get("properties", {})
            ]
            try:
                decision = policy.decide(
                    built.messages,
                    decision_format=built.decision_format,
                    tools=built.provider_tools,
                )
                rows.append({
                    "condition": condition,
                    "state": state_name,
                    "available_actions": available,
                    "selected_action": action_name(decision.action),
                    "arguments": action_arguments(decision.action),
                    "assessment": decision.assessment.model_dump(mode="json")
                    if decision.assessment is not None
                    else None,
                    "usage": {
                        key: policy.last_usage_metadata.get(key)
                        for key in (
                            "provider_input_tokens",
                            "provider_output_tokens",
                            "provider_total_tokens",
                            "tool_call_count",
                            "native_tool_calling",
                        )
                    },
                    "error": None,
                })
            except Exception as exc:  # preserve every model/protocol failure
                rows.append({
                    "condition": condition,
                    "state": state_name,
                    "available_actions": available,
                    "selected_action": None,
                    "arguments": None,
                    "assessment": None,
                    "usage": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                })
    return {
        "probe": "natural action selection; retrieval not executed",
        "model": model,
        "host": host,
        "substrate": substrate.root.resolve().as_posix(),
        "substrate_embedding_model": substrate.manifest.embedding_model.model_dump(mode="json"),
        "question": question,
        "conditions": list(CONDITIONS),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--skill", type=Path, default=Path("skills/interface_study.md"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.8:27b-q4_K_M")
    parser.add_argument("--host", default="http://127.0.0.1:11440")
    args = parser.parse_args()
    report = run_probe(
        Substrate.open(args.substrate),
        args.question,
        SkillDocument.load(args.skill),
        model=args.model,
        host=args.host,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output.as_posix(), "rows": len(report["rows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
