"""Offline, paired three-way TEST analysis. Never calls an LLM/optimizer."""

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from arag_skillopt.checkpoint import load_checkpoint, save_checkpoint
from arag_skillopt.support import digest


def load_arm(root, identity, questions):
    rows = {}
    for q in questions:
        folder = root / "episodes" / identity / digest(q["id"])[:20]
        raw_record = load_checkpoint(folder / "raw.json")
        evaluation_record = load_checkpoint(folder / "evaluation.json")
        if not raw_record or not evaluation_record:
            raise ValueError("missing original result: " + str(folder))
        raw, evaluation = raw_record["value"], evaluation_record["value"]
        if raw.get("question_id") != q["id"] or raw.get("question") != q["question"]:
            raise ValueError("question mismatch: " + str(folder))
        if raw_record["input_hash"] != digest({"id": q["id"], "question": q["question"]}):
            raise ValueError("runtime question hash mismatch")
        if evaluation_record["input_hash"] != digest({"question": q, "raw_hash": digest(raw)}):
            raise ValueError("evaluation lineage mismatch")
        correct = evaluation.get("correct") if evaluation.get("status") == "ok" else None
        if type(correct) is not bool:
            correct = None
        usage = raw.get("usage", {})
        rows[q["id"]] = {
            "correct": correct, "evaluation_status": evaluation.get("status"),
            "model_calls": raw.get("policy_calls"), "tool_calls": raw.get("tool_calls"),
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens") if usage.get("complete") else None,
            "elapsed_seconds": raw.get("elapsed_seconds"), "termination": raw.get("termination_reason"),
            "execution_errors": raw.get("execution_errors", []), "result_path": str(folder),
        }
    if len(rows) != len(questions):
        raise ValueError("duplicate question IDs")
    return rows


def pair_summary(a, b):
    counts = {k: 0 for k in ("both_correct", "b_only", "a_only", "both_wrong", "unknown")}
    common = []
    for qid in a:
        x, y = a[qid]["correct"], b[qid]["correct"]
        category = "unknown" if type(x) is not bool or type(y) is not bool else "both_correct" if x and y else "b_only" if y else "a_only" if x else "both_wrong"
        counts[category] += 1
        if category != "unknown":
            common.append(qid)
    n = len(common)
    result = {"counts": counts, "complete_pairs": n, "delta_on_complete_pairs": None,
              "paired_bootstrap_95_ci": None, "mcnemar_exact_p": None,
              "missing_pairs_excluded_explicitly": counts["unknown"]}
    if n:
        result["delta_on_complete_pairs"] = (counts["b_only"] - counts["a_only"]) / n
        probabilities = [counts["a_only"] / n, (counts["both_correct"] + counts["both_wrong"]) / n, counts["b_only"] / n]
        sample = np.random.default_rng(42).multinomial(n, probabilities, size=10000)
        result["paired_bootstrap_95_ci"] = np.quantile((sample[:, 2] - sample[:, 0]) / n, [.025, .975]).tolist()
        discordant = counts["a_only"] + counts["b_only"]
        result["mcnemar_exact_p"] = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(counts["a_only"], counts["b_only"]) + 1)) / 2 ** discordant) if discordant else 1.0
    for key in ("total_tokens", "model_calls", "tool_calls"):
        valid = [qid for qid in common if isinstance(a[qid][key], (int, float)) and isinstance(b[qid][key], (int, float))]
        sa, sb = sum(a[qid][key] for qid in valid), sum(b[qid][key] for qid in valid)
        result[key] = {"paired_count": len(valid), "a_sum": sa, "b_sum": sb,
                       "b_relative_change": (sb - sa) / sa if sa else None}
    return result


def summarize_arm(rows):
    known = [v for v in rows.values() if type(v["correct"]) is bool]
    correct = sum(v["correct"] for v in known)
    costs = [v for v in rows.values() if isinstance(v["total_tokens"], (int, float))]
    return {"count": len(rows), "evaluated": len(known), "correct": correct,
            "wrong": len(known) - correct, "unknown": len(rows) - len(known),
            "accuracy_on_evaluated": correct / len(known) if known else None,
            "full600_accuracy": correct / len(rows) if len(known) == len(rows) else None,
            "cost_complete_count": len(costs), "total_target_tokens_on_cost_complete": sum(v["total_tokens"] for v in costs),
            "model_calls": sum(v["model_calls"] for v in rows.values() if isinstance(v["model_calls"], int)),
            "tool_calls": sum(v["tool_calls"] for v in rows.values() if isinstance(v["tool_calls"], int))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-output", required=True)
    parser.add_argument("--original-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paired, original, out = [Path(x).resolve() for x in (args.paired_output, args.original_output, args.output)]
    for base in (paired, original):
        if load_checkpoint(base / "checkpoint.json").get("status") != "complete" or load_checkpoint(base / "summary.json").get("status") != "complete":
            raise ValueError("all three arms must finish before comparison")
    pc, oc = [load_checkpoint(base / "contract.json") for base in (paired, original)]
    if pc["test_hash"] != oc["test_hash"] or pc["runtime"]["data_hashes"] != oc["runtime"]["data_hashes"] or pc["runtime"]["model_digests"] != oc["runtime"]["model_digests"] or pc["runtime"]["settings"] != oc["runtime"]["settings"] or pc["runtime"]["source_hashes"] != oc["runtime"]["source_hashes"]:
        raise ValueError("comparison controls differ")
    path = Path(pc["runtime"]["settings"]["data"]["test"])
    from arag_skillopt.phase1_runtime import load_questions
    questions = load_questions(path)
    if len(questions) != 600 or digest(questions) != pc["test_hash"]:
        raise ValueError("must compare identical test600")
    arms = {name: load_arm(paired, identity, questions) for name, identity in pc["skill_hashes"].items()}
    arms["original"] = load_arm(original, oc["original_prompt_sha256"], questions)
    summaries = {name: summarize_arm(arm) for name, arm in arms.items()}
    pairs = {f"{a}_vs_{b}": pair_summary(arms[a], arms[b]) for a, b in itertools.combinations(["original", "initial", "optimized"], 2)}
    out.mkdir(parents=True, exist_ok=True)
    summary = {"status": "complete", "question_count": 600, "arms": summaries, "pairs": pairs,
               "test_feedback_to_optimizer": False, "source_contract_hashes": {"paired": digest(pc), "original": digest(oc)},
               "analysis_seed": 42, "bootstrap_samples": 10000,
               "limitations": ["Unknown evaluations are not wrong and are excluded explicitly from paired metrics.", "Runs were sequential, not interleaved; temperature zero does not guarantee reproducibility.", "Cost comparisons use the same questions with complete measurements.", "Target costs exclude Optimizer/Judge training overhead; all raw usage is retained."]}
    save_checkpoint(out / "summary.json", summary)
    with (out / "per_question.jsonl").open("w") as f:
        for q in questions:
            f.write(json.dumps({"question_id": q["id"], "question": q["question"], "arms": {name: arm[q["id"]] for name, arm in arms.items()}}, ensure_ascii=False) + "\n")
    text = ["# HotpotQA test600: original / formatted Initial / optimized", "", "All arms use the same 600 held-out questions, model digest, index, tools and budgets. Test feedback never enters training.", "", "| Arm | Correct / evaluated | Unknown | Accuracy on evaluated |", "|---|---:|---:|---:|"]
    for name, s in summaries.items():
        accuracy = f"{s['accuracy_on_evaluated']:.2%}" if s['accuracy_on_evaluated'] is not None else "N/A"
        text.append(f"| {name} | {s['correct']}/{s['evaluated']} | {s['unknown']} | {accuracy} |")
    for name, r in pairs.items():
        text.extend(["", f"## {name}", "", f"Complete paired questions: {r['complete_pairs']}; unknown pairs: {r['counts']['unknown']}.", f"B only correct: {r['counts']['b_only']}; A only correct: {r['counts']['a_only']}.", f"Accuracy delta B−A: {r['delta_on_complete_pairs']}; paired 95% CI: {r['paired_bootstrap_95_ci']}; McNemar exact p: {r['mcnemar_exact_p']}.", f"Target token relative change: {r['total_tokens']['b_relative_change']}; model-call relative change: {r['model_calls']['b_relative_change']}."])
    text.extend(["", "## Limitations", ""] + ["- " + x for x in summary["limitations"]])
    (out / "report.md").write_text("\n".join(text) + "\n")
    save_checkpoint(out / "checkpoint.json", {"status": "complete", "arms": ["original", "initial", "optimized"]})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
