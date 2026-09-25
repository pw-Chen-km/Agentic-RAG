"""Exercise every study retrieval branch on a real substrate with scripted actions.

This is a backend test, not a claim that the language model chose these actions.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (
    ExpandAction, ExpansionKind, FinishAction, ObservationStatus,
    PolicyDecision, SearchAction, SearchMethod, SearchTarget, Usage,
)
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.substrate.storage import Substrate


PATHS = {
    "C0": ("passages",),
    "C1": ("sentences",),
    "C2": ("passages", "entity_passages"),
    "C3": ("passages", "entity_sentences"),
    "C5": ("passages", "sentences", "entity_passages"),
    "C4": ("passages", "sentences", "entity_sentences"),
    "A1": ("sentences",),
}


class ScriptedBranchPolicy:
    last_usage = Usage(policy_calls=1)
    last_usage_metadata = {"native_tool_calling": True, "tool_call_count": 1}

    def __init__(self, path: tuple[str, ...]) -> None:
        self.path = path
        self.index = 0

    def decide(self, messages, *, tools=None, **kwargs) -> PolicyDecision:
        if not tools:
            raise AssertionError("current interface must expose native tools")
        if self.index >= len(self.path):
            return PolicyDecision(action=FinishAction(answer="Diagnostic complete", evidence_refs=[]))
        branch = self.path[self.index]
        self.index += 1
        if branch in {"passages", "sentences"}:
            return PolicyDecision(action=SearchAction(
                query="The Newcomers film cast", method=SearchMethod.DENSE,
                target=SearchTarget.CHUNK if branch == "passages" else SearchTarget.SENTENCE,
            ))
        entity_ref = None
        for message in messages:
            match = re.search(r"\b(E[1-9][0-9]*)\s+—\s+", message.content or "")
            if match:
                entity_ref = match.group(1)
                break
        if entity_ref is None:
            raise AssertionError(f"{branch}: no navigable entity was shown to the policy")
        return PolicyDecision(action=ExpandAction(
            kind=(ExpansionKind.ENTITY_MENTIONED_IN_CHUNK if branch == "entity_passages"
                  else ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE),
            source_ref=entity_ref, query=None,
        ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    substrate = Substrate.open(args.substrate)
    scope_id = next(iter(substrate.doc_ids_by_scope))
    base_config = AgentConfig.from_yaml(args.config)
    rows = []
    for condition, path in PATHS.items():
        config = base_config.model_copy(update={
            "interface": condition,
            "require_evidence_assessment": False,
        })
        harness = AgentHarness(
            substrate=substrate, config=config, skill=SkillDocument.load(args.skill),
            policy=ScriptedBranchPolicy(path), output_root=args.output / "episodes",
        )
        result = harness.run(
            "What is one of the stars of The Newcomers known for?",
            scope_id, episode_id=f"scripted-{condition}",
        )
        steps = result.trajectory
        outcomes = [{
            "action": branch, "status": step.observation.status.value,
            "validation_status": step.validation_status.value,
            "returned": len(step.context_audit.get("output_delivery", {}).get("returned_references", [])),
            "error_code": step.observation.error_code,
        } for branch, step in zip(path, steps)]
        passed = (
            len(steps) == len(path) + 1
            and all(item["status"] == ObservationStatus.OK.value
                    and item["validation_status"] == "valid" for item in outcomes)
            and result.termination_reason.value == "finish"
        )
        rows.append({"condition": condition, "path": path, "outcomes": outcomes,
                     "termination_reason": result.termination_reason.value, "passed": passed})
        report = {"scope": scope_id, "substrate": str(args.substrate.resolve()),
                  "policy": "scripted branch coverage, not LLM-selected actions",
                  "passed": all(row["passed"] for row in rows), "conditions": rows}
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not passed:
            raise AssertionError(f"{condition}: scripted backend branch failed; see report.json")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
