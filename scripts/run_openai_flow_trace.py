"""Run one real OpenAI episode and export provider/controller raw I/O.

The trace deliberately records HTTP JSON bodies but never request headers.
That makes the model inputs and outputs inspectable without persisting the
OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from agentic_rag.agent.answer import OpenAIResponsesAnswerGenerator
from agentic_rag.agent.config import AgentConfig, AnswerConfig, PolicyConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.policy import OpenAIResponsesPolicy


_OPENAI_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}")


class HttpJsonTraceRecorder:
    """Capture request/response bodies while excluding all HTTP headers."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._pending: dict[int, dict[str, Any]] = {}

    def on_request(self, request: httpx.Request) -> None:
        event = {
            "sequence": len(self.events) + 1,
            "request": {
                "method": request.method,
                "url": str(request.url),
                "headers": "[NOT RECORDED: may contain credentials]",
                "json_body": _decode_json(bytes(request.content)),
            },
            "response": None,
        }
        self.events.append(event)
        self._pending[id(request)] = event

    def on_response(self, response: httpx.Response) -> None:
        response.read()
        event = self._pending.pop(id(response.request), None)
        if event is None:
            event = next(
                (
                    item
                    for item in reversed(self.events)
                    if item["response"] is None
                ),
                None,
            )
        if event is None:
            return
        event["response"] = {
            "status_code": response.status_code,
            "headers": "[NOT RECORDED]",
            "json_body": _decode_json(response.content),
        }


def _decode_json(payload: bytes) -> Any:
    if not payload:
        return None
    text = payload.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _OPENAI_KEY_PATTERN.sub("[REDACTED_OPENAI_API_KEY]", value)
    return value


def _dump(value: Any) -> str:
    return json.dumps(
        _redact(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )


def _logical_trace(
    *,
    episode: dict[str, Any],
    provider_events: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    trajectory = episode["trajectory"]
    steps: list[dict[str, Any]] = []
    for index, record in enumerate(trajectory):
        event = provider_events[index] if index < len(provider_events) else None
        steps.append(
            {
                "step": record["step"],
                "stage": "POLICY_DECISION_AND_ACTION",
                "model": model,
                "raw_provider_http": event,
                "parsed_policy_decision": record["decision"],
                "validator": {
                    "input": {
                        "decision": record["decision"],
                        "scope_id": episode["scope_id"],
                        "state_before": record["state_before"],
                    },
                    "output": {
                        "status": record["validation_status"],
                        "error": record["validation_error"],
                    },
                },
                "router": {
                    "input": {
                        "action": (
                            record["decision"]["action"]
                            if record["decision"] is not None
                            else None
                        ),
                        "question": episode["query"],
                        "scope_id": episode["scope_id"],
                    },
                    "output": record["observation"],
                },
                "state_transition": {
                    "before": record["state_before"],
                    "after": record["state_after"],
                },
                "usage": record["usage"],
            }
        )

    answer_event = (
        provider_events[len(trajectory)]
        if len(provider_events) > len(trajectory)
        else None
    )
    return {
        "trace_format": "agentic-rag-full-flow-v1",
        "security": {
            "api_key_persisted": False,
            "http_request_headers_recorded": False,
            "note": (
                "HTTP JSON bodies are recorded. Authorization and all other "
                "request headers are intentionally excluded."
            ),
        },
        "episode": {
            "episode_id": episode["episode_id"],
            "question": episode["query"],
            "scope_id": episode["scope_id"],
            "model": model,
        },
        "policy_and_action_steps": steps,
        "answer_generation": {
            "stage": "EVIDENCE_ONLY_ANSWER_GENERATION",
            "model": model,
            "raw_provider_http": answer_event,
            "input_contract": {
                "question": episode["query"],
                "resolved_evidence": episode["resolved_evidence"],
            },
            "parsed_answer": episode["answer"],
        },
        "final_result": {
            "termination_reason": episode["termination_reason"],
            "answer": episode["answer"],
            "selected_evidence_refs": episode["selected_evidence_refs"],
            "usage": episode["usage"],
            "final_state": episode["final_state"],
            "error_code": episode["error_code"],
            "error_message": episode["error_message"],
        },
        "unassigned_provider_events": provider_events[len(trajectory) + 1 :],
    }


def _walkthrough(flow: dict[str, Any]) -> str:
    episode = flow["episode"]
    lines = [
        "# Agentic RAG full flow walkthrough",
        "",
        f"- Episode: `{episode['episode_id']}`",
        f"- Model: `{episode['model']}`",
        f"- Scope: `{episode['scope_id']}`",
        f"- Question: `{episode['question']}`",
        "",
        "API keys and HTTP headers are not recorded. Every JSON block below is "
        "the actual application/provider body or typed controller record from "
        "this run.",
    ]
    for step in flow["policy_and_action_steps"]:
        lines.extend(
            [
                "",
                f"## Step {step['step']}: Policy → Validator → Router → State",
                "",
                "### Raw OpenAI HTTP request/response",
                "",
                "```json",
                _dump(step["raw_provider_http"]),
                "```",
                "",
                "### Parsed PolicyDecision",
                "",
                "```json",
                _dump(step["parsed_policy_decision"]),
                "```",
                "",
                "### Validator input/output",
                "",
                "```json",
                _dump(step["validator"]),
                "```",
                "",
                "### Router input/output",
                "",
                "```json",
                _dump(step["router"]),
                "```",
                "",
                "### Controller state transition",
                "",
                "```json",
                _dump(step["state_transition"]),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## Evidence-only answer generation",
            "",
            "### Raw OpenAI HTTP request/response",
            "",
            "```json",
            _dump(flow["answer_generation"]["raw_provider_http"]),
            "```",
            "",
            "### Answer Generator input/output contract",
            "",
            "```json",
            _dump(
                {
                    "input": flow["answer_generation"]["input_contract"],
                    "output": flow["answer_generation"]["parsed_answer"],
                }
            ),
            "```",
            "",
            "## Final EpisodeResult",
            "",
            "```json",
            _dump(flow["final_result"]),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("substrate_path", type=Path)
    parser.add_argument("question")
    parser.add_argument("--scope-id", required=True)
    parser.add_argument("--skill-file", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--episode-id", required=True)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    config = AgentConfig.from_yaml(args.config)
    if not isinstance(config.policy, PolicyConfig):
        raise SystemExit("trace runner requires an OpenAI policy provider")
    if not isinstance(config.answer, AnswerConfig):
        raise SystemExit("trace runner requires an OpenAI answer provider")

    trace_dir = args.output / f"{args.episode_id}-trace"
    if trace_dir.exists():
        raise SystemExit(f"trace output already exists: {trace_dir}")

    recorder = HttpJsonTraceRecorder()
    http_client = httpx.Client(
        timeout=120,
        event_hooks={
            "request": [recorder.on_request],
            "response": [recorder.on_response],
        },
    )
    client = OpenAI(
        api_key=api_key,
        max_retries=0,
        http_client=http_client,
    )
    try:
        policy = OpenAIResponsesPolicy(
            model=config.policy.model,
            client=client,
            max_retries=config.policy.max_retries,
            enabled_expansions=config.enabled_expansions,
        )
        answer = OpenAIResponsesAnswerGenerator(
            model=config.answer.model,
            client=client,
            max_retries=config.answer.max_retries,
        )
        harness = AgentHarness.from_config(
            args.substrate_path,
            config,
            args.skill_file,
            args.output,
            policy=policy,
            answer_generator=answer,
        )
        result = harness.run(
            args.question,
            args.scope_id,
            episode_id=args.episode_id,
        )
    finally:
        client.close()

    episode = result.model_dump(mode="json")
    provider_events = _redact(recorder.events)
    flow = _logical_trace(
        episode=episode,
        provider_events=provider_events,
        model=config.policy.model,
    )
    serialized = _dump(
        {
            "provider_http_events": provider_events,
            "logical_flow": flow,
        }
    )
    if api_key in serialized or _OPENAI_KEY_PATTERN.search(serialized):
        raise RuntimeError("refusing to persist a trace containing an API key")

    trace_dir.mkdir(parents=True)
    (trace_dir / "raw_provider_http.json").write_text(
        _dump(provider_events) + "\n",
        encoding="utf-8",
    )
    (trace_dir / "full_flow.json").write_text(
        _dump(flow) + "\n",
        encoding="utf-8",
    )
    (trace_dir / "WALKTHROUGH.md").write_text(
        _walkthrough(flow),
        encoding="utf-8",
    )
    print(
        _dump(
            {
                "episode_dir": result.artifact_dir,
                "trace_dir": str(trace_dir),
                "termination_reason": result.termination_reason,
                "answer": result.answer,
                "steps": len(result.trajectory),
                "provider_calls": len(provider_events),
                "usage": result.usage,
            }
        )
    )


if __name__ == "__main__":
    main()
