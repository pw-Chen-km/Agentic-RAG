"""Audit completed or in-progress smoke artifacts without calling a model."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

CONDITIONS = {"C0", "C1", "C2", "C3", "C4", "C5", "A1"}
FILES = {"episode.json", "conversation.json", "target_system_prompt.txt",
         "target_user_prompt.txt", "skill.md", "effective_config.json"}


def audit(root):
    rows, violations, tools_used = [], [], Counter()
    for ds in ("hotpotqa", "novel", "medical"):
        for path in sorted((root / ds / "episodes").glob("*/episode.json")):
            ep = json.loads(path.read_text(encoding="utf-8"))
            eid = ep["episode_id"]
            condition = eid.split("--", 1)[0]
            errors = []
            missing = FILES - {p.name for p in path.parent.iterdir()}
            if missing:
                errors.append(f"missing_artifacts:{sorted(missing)}")
            steps = ep.get("trajectory", [])
            normal, finalize, protocol_invalid, state_invalid, executed = 0, 0, 0, 0, 0
            uptake = Counter()
            for index, step in enumerate(steps):
                metadata = (step.get("observation") or {}).get("metadata") or {}
                is_final = metadata.get("budget_finalize") is True
                finalize += is_final
                normal += not is_final
                defs = step.get("tool_definitions", [])
                available = {t["function"]["name"]:t["function"] for t in defs}
                if is_final and set(available) != {"finish"}:
                    errors.append(f"step_{index}:finalize_has_retrieval")
                if condition == "C0" and set(available) - {"find_passages", "finish"}:
                    errors.append(f"step_{index}:C0_capability_leak")
                if condition == "A1" and any(k.startswith("follow_") for k in available):
                    errors.append(f"step_{index}:A1_navigation_leak")
                provider = step.get("provider_metadata") or {}
                structured = provider.get("raw_structured_decision")
                native_calls = provider.get("raw_tool_calls") or []
                if native_calls:
                    errors.append(f"step_{index}:unexpected_native_tool_calls")
                if provider.get("constrained_single_decision") is not True:
                    errors.append(f"step_{index}:not_constrained_single_decision")
                calls = []
                if isinstance(structured, dict) and isinstance(structured.get("action"), dict):
                    calls = [structured["action"]]
                if step.get("validation_status") == "valid" and (
                    provider.get("decision_count") != 1 or len(calls) != 1
                ):
                    errors.append(f"step_{index}:decision_count_not_one")
                for action in calls:
                    name = action.get("name", "")
                    uptake[name] += 1
                    tools_used[name] += 1
                    if name not in available:
                        errors.append(f"step_{index}:unavailable_tool:{name}")
                if step.get("validation_status") == "invalid":
                    if provider.get("failure_category") == "protocol_invalid":
                        protocol_invalid += 1
                    else:
                        state_invalid += 1
                schema = step.get("decision_schema") or {}
                canonical = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if not schema or hashlib.sha256(canonical.encode()).hexdigest() != step.get("decision_schema_sha256"):
                    errors.append(f"step_{index}:decision_schema_digest_mismatch")
                observation = step.get("observation") or {}
                if step.get("validation_status") == "valid" and observation.get("status") == "ok":
                    executed += 1
                if any(provider.get(k) is None for k in ("provider_input_tokens", "provider_output_tokens", "provider_total_tokens")):
                    errors.append(f"step_{index}:provider_usage_unavailable")
                content = "\n".join(m.get("content") or "" for m in step.get("messages", []))
                for span in step.get("visible_source_spans", []):
                    if not (span.get("seen_by_policy") and span.get("complete") and span.get("span_type") == "sentence"):
                        errors.append(f"step_{index}:span_not_verified_policy_input")
                    if not span.get("text") or span["text"] not in content:
                        errors.append(f"step_{index}:span_absent_from_messages")
                if index == 0 and step.get("visible_source_spans"):
                    errors.append("initial_input_exposes_source")
            if normal > 15 or finalize > 1:
                errors.append("decision_budget_exceeded")
            if ep.get("termination_reason") in {"runtime_error", "policy_error"}:
                errors.append(f"terminal:{ep.get('termination_reason')}:{ep.get('error_message')}")
            rows.append(dict(dataset=ds, condition=condition, episode_id=eid,
                terminal=ep.get("termination_reason"), policy_calls=len(steps),
                protocol_invalid=protocol_invalid, state_invalid=state_invalid,
                execution_ok=executed, tool_uptake=dict(uptake), usage=ep.get("usage"), errors=errors))
            violations.extend(f"{ds}/{eid}:{e}" for e in errors)
    completed = len(rows)
    for ds in ("hotpotqa", "novel", "medical"):
        actual = {r["condition"] for r in rows if r["dataset"] == ds}
        if actual != CONDITIONS:
            violations.append(f"{ds}:missing_conditions:{sorted(CONDITIONS - actual)}")
    return dict(status="artifact_checks_passed" if completed == 21 and not violations else "pending_or_failed",
        episodes=completed, expected=21, tool_uptake=dict(tools_used), violations=violations, results=rows,
        limitations=["21 episodes do not establish 99% reliability or all-tool branch coverage",
            "Gold provenance must be checked structurally; answer substring matches alone cannot establish leakage",
            "Semantic judge is separate; this audit makes no claim of live judge success"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = audit(a.run)
    if a.output:
        a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
