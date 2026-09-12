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
        section = "answer_policy" if stage == "answer" else "recovery_policy"
        return {"sections": {section: "Use a different available retrieval path when no new evidence is obtained."}
                if stage != "answer" else {section: "Finish once legal evidence covers all requested conditions."},
                "accept": True, "reason": "FAKE demo proposal; not research evidence"}
