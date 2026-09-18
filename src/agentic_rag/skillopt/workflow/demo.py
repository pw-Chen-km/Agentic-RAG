"""Deterministic fake backend for installation and resume checks only."""
from .runner import skill_hash


class DemoBackend:
    def rollout(self, questions, skill, split, output, purpose):
        return [{"id": q["id"], "split": split, "skill_sha256": skill_hash(skill),
                 "purpose": purpose, "correct": 1, "tokens": 100, "calls": 2,
                 "view": {"id": q["id"], "split": split, "question": q["question"],
                          "steps": [{"action": "SEARCH"}, {"action": "FINISH"}]},
                 "answer_view": {"id": q["id"], "split": split, "correct": 1}}
                for q in questions]

    def optimize(self, stage, operation, payload, output):
        if "rule_catalog" in payload:
            rules = payload["rule_catalog"].get("rules", [])
            target = next((r for r in rules if r["rule_id"] == "R05"), None)
            if target is not None and operation in {"reflect", "merge", "summarize_merge"}:
                rule = dict(target)
                rule["stop_or_recovery"] = "Use a different legal path when no new information is obtained."
                return {"edits": [{"operation": "replace", "rule_id": "R05",
                                    "rule": rule, "reason": "FAKE demo edit",
                                    "supporting_case_ids": []}],
                        "no_change": False, "reason": "FAKE demo proposal; not research evidence"}
            return {"edits": [], "no_change": True, "reason": "FAKE demo no change"}
        section = "answer_policy" if stage == "answer" else "recovery_policy"
        return {"sections": {section: "Use a different available retrieval path when no new evidence is obtained."}
                if stage != "answer" else {section: "Finish once legal evidence covers all requested conditions."},
                "reason": "FAKE demo proposal; not research evidence"}
