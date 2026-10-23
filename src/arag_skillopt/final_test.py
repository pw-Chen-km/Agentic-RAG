"""Post-training paired test; no optimizer, proposals, gate or Skill selection."""

import copy
import fcntl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .checkpoint import load_checkpoint, save_checkpoint
from .phase1_gate import metrics
from .skill_rules import skill_sha256
from .support import digest
from .trajectory import build_view
from .retrieval_diagnostics import build_retrieval_diagnostics
from .paired_summary import summarize_pair


class FinalTestRunner:
    def __init__(
        self, output, rollout, evaluator, contract, workers=2, support_map=None
    ):
        if not 1 <= workers <= 8:
            raise ValueError("test workers must be between 1 and 8")
        self.root = Path(output)
        self.rollout, self.evaluator = rollout, evaluator
        self.runtime = copy.deepcopy(contract)
        self.workers = workers
        self.support_map = support_map or {"questions": {}}

    def cached(self, path, inputs, compute):
        signature = digest(inputs)
        record = load_checkpoint(path)
        if record:
            if record.get("input_hash") != signature:
                raise ValueError("final test checkpoint changed: " + str(path))
            return record["value"]
        value = compute()
        save_checkpoint(path, {"input_hash": signature, "value": value})
        return value

    def episode(self, question, skill):
        qid = str(question["id"])
        folder = self.root / "episodes" / skill_sha256(skill) / digest(qid)[:20]
        runtime_question = {"id": qid, "question": question["question"]}
        raw = self.cached(
            folder / "raw.json",
            runtime_question,
            lambda: self.rollout(copy.deepcopy(runtime_question), copy.deepcopy(skill)),
        )
        save_checkpoint(
            folder / "retrieval_diagnostics.json",
            {
                "input_hash": digest({"raw": digest(raw)}),
                "value": build_retrieval_diagnostics(raw),
            },
        )
        if raw.get("question_id") != qid or raw.get("question") != question["question"]:
            raise ValueError("test runtime question mismatch")

        def evaluate():
            if raw.get("error") or raw.get("execution_errors"):
                return {"correct": None, "status": "runtime_error"}
            result = self.evaluator(question, raw)
            if result.get("status") != "ok" or type(result.get("correct")) is not bool:
                return {**result, "correct": None, "status": "unavailable"}
            return result

        evaluation = self.cached(
            folder / "evaluation.json",
            {"question": question, "raw_hash": digest(raw)},
            evaluate,
        )
        support = self.support_map["questions"].get(qid)
        if support and support["question"] != question["question"]:
            raise ValueError("test supporting-fact question mismatch")
        view, audit = build_view(raw, evaluation, support, "final test")
        save_checkpoint(folder / "view.json", view)
        save_checkpoint(folder / "lineage.json", {"split": "test", **audit})
        return {"question_id": qid, "raw": raw, "evaluation": evaluation, "view": view}

    @staticmethod
    def arm_summary(episodes):
        known = [e for e in episodes if type(e["evaluation"].get("correct")) is bool]
        correct = sum(e["evaluation"]["correct"] for e in known)
        return {
            **metrics(episodes),
            "evaluated": len(known),
            "correct": correct,
            "unknown": len(episodes) - len(known),
            "accuracy_on_evaluated": correct / len(known) if known else None,
        }

    def run(self, questions, initial, optimized, training_ids, validation_ids):
        ids = [str(q["id"]) for q in questions]
        if (
            not ids
            or len(ids) != len(set(ids))
            or set(ids) & (set(training_ids) | set(validation_ids))
        ):
            raise ValueError(
                "test IDs must be unique and disjoint from train/validation"
            )
        contract = {
            "purpose": "post_training_test_only",
            "test_feedback_to_optimizer": False,
            "test_hash": digest(questions),
            "skill_hashes": {
                "initial": skill_sha256(initial),
                "optimized": skill_sha256(optimized),
            },
            "workers": self.workers,
            "support_map_hash": digest(self.support_map),
            "runtime": self.runtime,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".test.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            saved = load_checkpoint(self.root / "contract.json")
            if saved and saved != contract:
                raise ValueError("final test contract changed; use a new output")
            if not saved and any(p.name != ".test.lock" for p in self.root.iterdir()):
                raise ValueError("nonempty final test output without contract")
            save_checkpoint(self.root / "contract.json", contract)
            arms = {}
            for name, skill in (("initial", initial), ("optimized", optimized)):
                save_checkpoint(self.root / (name + "_skill.json"), skill)
                save_checkpoint(
                    self.root / "checkpoint.json", {"status": "running", "arm": name}
                )
                if name == "optimized" and initial == optimized:
                    arms[name] = arms["initial"]
                else:
                    with ThreadPoolExecutor(max_workers=self.workers) as pool:
                        arms[name] = list(
                            pool.map(
                                lambda q, frozen=skill: self.episode(q, frozen),
                                questions,
                            )
                        )
            paired, counts = (
                [],
                {
                    k: 0
                    for k in (
                        "both_correct",
                        "optimized_only",
                        "initial_only",
                        "both_wrong",
                        "unknown",
                    )
                },
            )
            question_text = {str(q["id"]): q["question"] for q in questions}
            for a, b in zip(arms["initial"], arms["optimized"]):
                x, y = a["evaluation"]["correct"], b["evaluation"]["correct"]
                category = (
                    "unknown"
                    if type(x) is not bool or type(y) is not bool
                    else "both_correct"
                    if x and y
                    else "optimized_only"
                    if y
                    else "initial_only"
                    if x
                    else "both_wrong"
                )
                counts[category] += 1
                paired_summary = summarize_pair(
                    a["question_id"],
                    a["raw"],
                    b["raw"],
                    a["evaluation"],
                    b["evaluation"],
                )
                paired.append(
                    {
                        "question_id": a["question_id"],
                        "question": question_text[a["question_id"]],
                        "initial_predicted_answer": a["raw"].get("answer"),
                        "optimized_predicted_answer": b["raw"].get("answer"),
                        "initial_correct": x,
                        "optimized_correct": y,
                        "category": category,
                        "paired_summary": paired_summary,
                    }
                )
            save_checkpoint(self.root / "paired.json", {"rows": paired})
            summary = {
                "status": "complete",
                "question_count": len(ids),
                "identical_skills_reused": initial == optimized,
                "initial": self.arm_summary(arms["initial"]),
                "optimized": self.arm_summary(arms["optimized"]),
                "paired": counts,
                "accuracy_delta": (counts["optimized_only"] - counts["initial_only"])
                / len(ids)
                if not counts["unknown"]
                else None,
                "test_feedback_to_optimizer": False,
            }
            save_checkpoint(self.root / "summary.json", summary)
            save_checkpoint(
                self.root / "checkpoint.json", {"status": "complete", "arm": "complete"}
            )
            return summary
