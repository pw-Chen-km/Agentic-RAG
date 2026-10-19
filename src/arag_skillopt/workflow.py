"""Small, independent SkillOpt workflow for A-RAG.

Legacy smoke/Meta adapter. The current success/failure-separated Phase 1 runner
is arag_skillopt.phase1.Phase1 (scripts/run_phase1.py), not this module.

The official A-RAG runtime remains responsible for tool execution.  This
module only batches its completed trajectories, asks the optimizer for bounded
native-tool edits, and applies them through :class:`RuleStore`.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping, Sequence

from arag.core.llm import LLMClient

from .optimizer_tools import parse_tool_call, tool_schemas
from .skill_rules import RuleStore, load_rules, skill_sha256


@dataclass(frozen=True)
class WorkflowConfig:
    rollout_batch_size: int = 40
    reflection_minibatch_size: int = 5
    max_edits_per_reflection: int = 2
    max_retries: int = 3
    use_validation_gate: bool = True


class SkillOptWorkflow:
    """A-RAG-specific retrieval/meta optimizer with auditable decisions."""

    def __init__(
        self,
        rules: Mapping[str, Any],
        optimizer_client: LLMClient,
        retrieval_prompt: str,
        meta_prompt: str,
        config: WorkflowConfig | None = None,
    ) -> None:
        self.store = RuleStore(rules)
        self.client = optimizer_client
        self.retrieval_prompt = retrieval_prompt
        self.meta_prompt = meta_prompt
        self.config = config or WorkflowConfig()

    def _input(self, prompt: str, cases: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
        compact = json.dumps(list(cases), ensure_ascii=False, separators=(",", ":"))
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Current rules:\n" + json.dumps(self.store.payload, ensure_ascii=False) + "\n\nCompleted cases:\n" + compact},
        ]

    def reflect(self, cases: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any]:
        """Ask native tools; three failures become an explicit no_change."""
        if stage not in {"retrieval", "meta"}:
            raise ValueError("stage must be retrieval or meta")
        prompt = self.retrieval_prompt if stage == "retrieval" else self.meta_prompt
        audit: list[dict[str, Any]] = []
        for retry in range(1, self.config.max_retries + 1):
            try:
                response = self.client.chat(
                    messages=self._input(prompt, cases),
                    tools=tool_schemas(list(self.store.rules) and [r["rule_id"] for r in self.store.rules]),
                    temperature=0.0,
                )
                message = response.get("message", {})
                calls = message.get("tool_calls") or []
                if not calls:
                    raise ValueError("optimizer returned no tool call")
                if len(calls) > self.config.max_edits_per_reflection:
                    raise ValueError("too many optimizer tool calls")
                edits = [parse_tool_call(call) for call in calls]
                if any(edit["operation"] == "no_change" for edit in edits) and len(edits) != 1:
                    raise ValueError("no_change cannot be combined with edits")
                audit.append({"retry": retry, "tool_calls": calls, "parsed": edits,
                              "input_tokens": response.get("input_tokens"),
                              "output_tokens": response.get("output_tokens")})
                return {"stage": stage, "edits": edits, "audit": audit, "status": "ok"}
            except Exception as exc:
                audit.append({"retry": retry, "error": str(exc)})
        return {"stage": stage, "edits": [{"operation": "no_change", "reason": "optimizer failed after retries"}],
                "audit": audit, "status": "no_change_after_retry"}

    def apply(self, proposal: Mapping[str, Any], *, source_stage: str, validation_gate: Callable[[Mapping[str, Any]], bool] | None = None) -> dict[str, Any]:
        edits = list(proposal.get("edits") or [])
        parent_hash = skill_sha256(self.store.payload)
        try:
            candidate = RuleStore(self.store.payload, max_edits=self.config.max_edits_per_reflection).apply(edits)
            candidate_hash = skill_sha256(candidate)
            if not self.config.use_validation_gate:
                accepted = True
            elif validation_gate is None:
                # Never silently bypass the gate.  A smoke caller can
                # explicitly disable it; a real training caller must supply
                # the validation result.
                accepted = False
            else:
                accepted = bool(validation_gate(candidate))
            if accepted:
                self.store = RuleStore(candidate)
            return {"accepted": accepted, "parent_skill_hash": parent_hash,
                    "candidate_skill_hash": candidate_hash, "edits": edits,
                    "active_skill_hash": skill_sha256(self.store.payload), "error": None}
        except Exception as exc:
            return {"accepted": False, "parent_skill_hash": parent_hash,
                    "candidate_skill_hash": None, "edits": edits,
                    "active_skill_hash": parent_hash, "error": str(exc)}

    def run_reflection_minibatches(self, cases: Sequence[Mapping[str, Any]], stage: str) -> list[dict[str, Any]]:
        receipts: list[dict[str, Any]] = []
        size = self.config.reflection_minibatch_size
        for start in range(0, len(cases), size):
            proposal = self.reflect(cases[start:start + size], stage)
            receipts.append(self.apply(proposal, source_stage=stage))
        return receipts
