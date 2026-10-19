"""Verified same-question branches and exact-text comparison views."""

import copy
import json

from .checkpoint import load_checkpoint
from .compact_rendering import (
    DISPLAY_GUIDANCE,
    DISPLAY_VERSION,
    PLAIN_DISPLAY_VERSION,
    CompactDisplay,
    additional_answer,
    cost_line,
    evaluation_line,
    progress_lines,
)
from .skill_rules import skill_sha256, validate_rule
from .support import digest
from .trajectory import TextHistory, build_view, decisions_from_raw

COMPARISON_VERSION = "arag-same-question-decisions-v2-compact"
PLAIN_COMPARISON_VERSION = "arag-same-question-context-v6"

PLAIN_DISPLAY_GUIDANCE = (
    "Each Decision is one turn in which the agent chooses what to do next. "
    "All tools listed in that Decision were chosen before any of their results "
    "came back. A result can influence only the later Decision named beside it. "
    "Earlier results remain available unless the record says they were removed. "
    "To avoid printing the same document text repeatedly, r1 and similar names "
    "label text already printed here. [r1] means that same text; "
    "[r1@start:end] means the characters from start up to, but not including, "
    "end in the original text labelled r1. Sharing printed text between Run A "
    "and Run B does not give either agent information from the other run. "
    "Each run's received documents, full reads and costs are counted separately. "
    "A new Chunk ID means a newly returned document section, not necessarily "
    "useful information. Checks against reference evidence were calculated "
    "after the run; the agent could not see those checks. GT is the label for "
    "that reference evidence. N/A means a value "
    "could not be measured, not zero."
)


def _historical_context(comparisons, historical_skill_a, historical_skill_b):
    """Check exact historical rules before showing them in a new context format."""
    if historical_skill_a is None or historical_skill_b is None:
        raise ValueError("plain-language Meta context requires historical Skills A and B")
    skills = (historical_skill_a, historical_skill_b)
    public = []
    for skill in skills:
        if (
            not isinstance(skill, dict)
            or skill.get("schema_version") != "arag-skill-v1"
            or set(skill) - {"schema_version", "retrieval_policy", "source_prompt_sha256"}
        ):
            raise ValueError("invalid historical Skill schema")
        rules = skill.get("retrieval_policy")
        if not isinstance(rules, list):
            raise ValueError("historical Skill rules must be a list")
        seen = set()
        for rule in rules:
            if not isinstance(rule, dict):
                raise ValueError("invalid historical rule")
            validate_rule(rule)
            if rule["rule_id"] in seen:
                raise ValueError("duplicate historical rule ID")
            seen.add(rule["rule_id"])
        public.append({"schema_version": skill["schema_version"], "retrieval_policy": rules})
    hashes = tuple(skill_sha256(skill) for skill in skills)
    diffs = comparisons[0]["rule_differences"]
    if not isinstance(diffs, list) or not diffs:
        raise ValueError("historical Skills must have recorded rule differences")
    reconstructed = {r["rule_id"]: copy.deepcopy(r) for r in public[0]["retrieval_policy"]}
    touched = set()
    for change in diffs:
        if not isinstance(change, dict) or set(change) != {"rule_id", "operation", "before", "after"}:
            raise ValueError("invalid historical rule difference")
        rid, operation = change["rule_id"], change["operation"]
        if not isinstance(rid, str) or rid in touched:
            raise ValueError("duplicate or invalid historical difference rule ID")
        touched.add(rid)
        if not isinstance(operation, str) or operation not in {"add", "replace", "delete"}:
            raise ValueError("unknown historical rule operation")
        exists = rid in reconstructed
        if operation == "add":
            if exists or change["before"] is not None:
                raise ValueError("historical add does not match Skill A")
        elif not exists or change["before"] != reconstructed[rid]:
            raise ValueError("historical rule before does not match Skill A")
        if operation == "delete":
            if change["after"] is not None:
                raise ValueError("historical delete must have no replacement")
            del reconstructed[rid]
        else:
            after = change["after"]
            if not isinstance(after, dict) or after.get("rule_id") != rid:
                raise ValueError("historical replacement rule ID mismatch")
            validate_rule(after)
            reconstructed[rid] = copy.deepcopy(after)
    order = [rule["rule_id"] for rule in public[1]["retrieval_policy"]]
    if set(reconstructed) != set(order) or [reconstructed[rid] for rid in order] != public[1]["retrieval_policy"]:
        raise ValueError("historical differences do not reconstruct Skill B")
    if diffs != rule_differences(historical_skill_a, historical_skill_b):
        raise ValueError("recorded differences do not exactly describe historical Skills A and B")
    # Check all group members, not just their equal diffs: two historical pairs
    # can share a change while having different unchanged rules.
    for comparison in comparisons:
        if (comparison.get("parent_skill_hash"), comparison.get("candidate_skill_hash")) != hashes:
            raise ValueError("Meta reflection group historical Skill identity mismatch")
        if comparison["rule_differences"] != diffs:
            raise ValueError("a Meta reflection group must share one historical Skill pair")
    return public[0], order


def rule_differences(parent, candidate):
    a = {r["rule_id"]: r for r in parent["retrieval_policy"]}
    b = {r["rule_id"]: r for r in candidate["retrieval_policy"]}
    return [
        {
            "rule_id": rid,
            "operation": "add"
            if rid not in a
            else "delete"
            if rid not in b
            else "replace",
            "before": a.get(rid),
            "after": b.get(rid),
        }
        for rid in sorted(a.keys() | b.keys())
        if a.get(rid) != b.get(rid)
    ]


def load_branch(phase1, ref, expected_hash, expected_qid, question):
    """Load immutable raw/evaluation receipts; never accepts validation/test paths."""
    qid = str(expected_qid)
    if (
        ref["split"] != "train"
        or ref["question_id"] != qid
        or ref["skill_hash"] != expected_hash
    ):
        raise ValueError("Meta branch question, split, or Skill mismatch")
    folder = (phase1.root / ref["episode_directory"]).resolve()
    required = (
        phase1.root / "episodes" / "train" / expected_hash / digest({"id": qid})[:20]
    ).resolve()
    if folder != required:
        raise ValueError("Meta branch is outside its exact TRAIN episode directory")
    raw_receipt = load_checkpoint(folder / "raw.json")
    eval_receipt = load_checkpoint(folder / "evaluation.json")
    lineage = load_checkpoint(folder / "lineage.json")
    if not raw_receipt or not eval_receipt or not lineage:
        raise ValueError("missing Meta branch receipt")
    raw, evaluation = raw_receipt["value"], eval_receipt["value"]
    if lineage["split"] != "train" or lineage["skill_hash"] != expected_hash:
        raise ValueError("Meta branch lineage mismatch")
    if raw.get("question_id") != qid or raw.get("question") != question["question"]:
        raise ValueError("Meta branch does not belong to the requested question")
    if digest(raw) != ref["raw_hash"] or digest(evaluation) != ref["evaluation_hash"]:
        raise ValueError("Meta branch content changed")
    if raw_receipt["input_hash"] != digest(
        {"id": qid, "question": question["question"]}
    ):
        raise ValueError("Meta branch runtime input mismatch")
    if eval_receipt["input_hash"] != digest({"question": question, "raw": digest(raw)}):
        raise ValueError("Meta branch evaluator input mismatch")
    return {"raw": raw, "evaluation": evaluation}


def _behavior(decision):
    return {
        "calls": decision["calls"],
        "results": [
            {
                "tool": r["step"].get("tool_name"),
                "arguments": r["step"].get("arguments"),
                "text": r["step"].get("tool_result"),
                "error": r["step"].get("error"),
                "visible_at": r["visible_at"],
            }
            for r in decision["results"]
        ],
        "response": decision.get("response"),
        "error": decision.get("error"),
        "forced_answer": decision.get("force_answer_instruction", False),
    }


def _visible(raw, index):
    audits = raw.get("decision_audit", [])
    if index >= len(audits):
        return None
    # Tool call IDs can differ between independent rollouts; source text and its
    # actual delivery order are the comparable evidence, not generated call IDs.
    return [
        m["content"]
        for m in audits[index]["request"]["messages"]
        if m.get("role") == "tool"
    ]


def make_comparison(question, parent, candidate, branch_a, branch_b, batch):
    qid = str(question["id"])
    if not rule_differences(parent, candidate):
        raise ValueError("Meta requires distinct parent/candidate rules")
    for branch in (branch_a, branch_b):
        raw = branch["raw"]
        if raw.get("question_id") != qid or raw.get("question") != question["question"]:
            raise ValueError("Meta requires same-question branches")
    raws = [b["raw"] for b in (branch_a, branch_b)]
    decisions = [decisions_from_raw(r) for r in raws]
    behavior = [[_behavior(d) for d in ds] for ds in decisions]
    first = next(
        (
            i
            for i in range(max(map(len, behavior)))
            if (behavior[0][i] if i < len(behavior[0]) else None)
            != (behavior[1][i] if i < len(behavior[1]) else None)
        ),
        None,
    )
    usage = [r.get("usage", {}) for r in raws]
    costs = [
        {
            "model_calls": r.get("policy_calls"),
            "tool_calls": len(r.get("trajectory", [])),
            "model_usage": r.get("usage"),
            "retrieved_text_tokens": r.get("total_retrieved_tokens"),
        }
        for r in raws
    ]
    eligible = all(
        b["evaluation"].get("status") == "ok"
        and type(b["evaluation"].get("correct")) is bool
        and not r.get("execution_errors")
        and not r.get("error")
        and r.get("decision_audit")
        and u.get("complete")
        and isinstance(u.get("total_tokens"), (int, float))
        for b, r, u in zip((branch_a, branch_b), raws, usage)
    )
    identical = (
        behavior[0] == behavior[1]
        and costs[0] == costs[1]
        and branch_a["evaluation"].get("correct")
        == branch_b["evaluation"].get("correct")
    )
    return {
        "comparison_version": COMPARISON_VERSION,
        "question_id": qid,
        "question": question["question"],
        "train_batch": batch,
        "match_level": "same_question",
        "causal_claim": False,
        "parent_skill_hash": skill_sha256(parent),
        "candidate_skill_hash": skill_sha256(candidate),
        "rule_differences": rule_differences(parent, candidate),
        "first_difference": None
        if first is None
        else {
            "decision": first + 1,
            "earlier_actions_and_results_equal": behavior[0][:first]
            == behavior[1][:first],
            "confirmed_visible_results_equal": _visible(raws[0], first) is not None
            and _visible(raws[0], first) == _visible(raws[1], first),
            "exact_state_match": False,
        },
        "eligible_for_reflection": bool(eligible and not identical),
        "skip_reason": "incomplete_runtime_or_evaluation"
        if not eligible
        else "identical_behavior_and_cost"
        if identical
        else None,
        "costs": {"A": costs[0], "B": costs[1]},
        "outcomes": {
            "A": branch_a["evaluation"].get("correct"),
            "B": branch_b["evaluation"].get("correct"),
        },
    }


def comparison_views(branch_a, branch_b, support_entry):
    """Deduplicate text across branches without recomputing novelty/recall across them."""
    history = TextHistory()
    result = {}
    for label, branch in (("A", branch_a), ("B", branch_b)):
        view, audit = build_view(
            branch["raw"], branch["evaluation"], support_entry, "Branch " + label
        )
        bodies = {
            x["location"]: x["raw_body"] for x in audit["lineage"] if "raw_body" in x
        }
        for decision in view["decisions"]:
            for ri, observation in enumerate(decision["results"], 1):
                for ui, unit in enumerate(observation["items"], 1):
                    if "content" not in unit:
                        continue
                    loc = f"Decision {decision['decision']}, result {ri}, item {ui}"
                    unit["content"], _ = history.render(
                        unit["chunk_id"], bodies[loc], f"Branch {label} / {loc}"
                    )
        result[label] = view
    return result


def render_comparisons_with_audit(
    comparisons, *, plain_language=False, historical_skill_a=None, historical_skill_b=None
):
    """Hashes are audit-only. Historical rule differences appear once per group."""
    if not comparisons:
        audit = {
            "renderer_version": PLAIN_DISPLAY_VERSION if plain_language else DISPLAY_VERSION,
            "cases": [],
        }
        if plain_language:
            audit["comparison_context_version"] = PLAIN_COMPARISON_VERSION
        return "No eligible same-question comparisons.", audit
    diffs = comparisons[0]["rule_differences"]
    if any(c["rule_differences"] != diffs for c in comparisons):
        raise ValueError("a Meta reflection group must share one historical Skill pair")
    if plain_language:
        skill_a, skill_b_order = _historical_context(
            comparisons, historical_skill_a, historical_skill_b
        )
        lines = [
            PLAIN_DISPLAY_GUIDANCE,
            "Complete rules used by Run A (a saved previous attempt; shared by these questions):",
            json.dumps(skill_a, ensure_ascii=False),
            "Rules used by Run B: start with Run A's complete rules above, then apply every change below. Each before and after contains the complete rule, or null for a rule that did not exist.",
            json.dumps(diffs, ensure_ascii=False),
            "Run B rule order: " + json.dumps(skill_b_order),
            "These saved runs used the rules shown here. The Current Skill supplied separately is the set of rules you can improve now; it may already address a problem in these older runs.",
            "The same question and similar recorded actions do not prove that the agents had the same information or that one edited rule caused a difference.",
        ]
    else:
        lines = [
        DISPLAY_GUIDANCE,
        "Historical Skill A -> B rule differences (shared by this group):",
        json.dumps(diffs, ensure_ascii=False),
        "A and B are historical evidence. The Current Skill above is the editable Skill.",
        "A/B may share displayed text, never visibility or recall state. Common recorded actions/results are not identical hidden states or causal proof.",
        ]
    audits = []
    for comparison in comparisons:
        display = CompactDisplay()
        for label in ("A", "B"):
            display.prepare(comparison["views"][label], f"Branch {label} / ")
        a, b = (comparison["views"][label]["decisions"] for label in ("A", "B"))
        common = 0
        recorded_difference = comparison.get("first_difference")
        # The raw comparison also sees tool headers/scores and other metadata
        # omitted from the derived body view. Never extend a shared prefix past
        # an already-recorded raw difference just because body text is equal.
        common_limit = (
            recorded_difference["decision"] - 1
            if recorded_difference is not None
            else min(len(a), len(b))
        )
        for da, db in zip(a, b):
            if common >= common_limit:
                break
            if display.behavior_key(da, "Branch A / ") != display.behavior_key(
                db, "Branch B / "
            ):
                break
            common += 1
        first_text = json.dumps(comparison["first_difference"])
        if plain_language:
            if recorded_difference is None:
                first_text = "No difference in the recorded actions, returned results or model responses; measured costs may still differ."
            else:
                first_text = f"Decision {recorded_difference['decision']}."
                if recorded_difference.get("confirmed_visible_results_equal"):
                    first_text += " Before this choice, both runs had received the same tool-result text."
                else:
                    first_text += " Do not assume the two runs had received the same information before this choice."
        lines += [
            "\nCase ID: " + comparison["question_id"],
            "Question: " + comparison["question"],
            "First recorded difference: " + first_text,
        ]
        if common:
            lines.append("\nRecorded actions and results shared by Run A and Run B:" if plain_language else "\nCommon recorded actions/results for A and B:")
            for da, db in zip(a[:common], b[:common]):
                n = da["decision"]
                lines.extend(
                    display.decision(
                        da, "Branch A / ",
                        f"Run A and Run B, Decision {n}" if plain_language else f"A{n}/B{n}",
                        cost=False, **({"plain_language": True} if plain_language else {}),
                    )
                )
                for ri, result in enumerate(db["results"], 1):
                    for ui, _ in enumerate(result["items"], 1):
                        original = display.location("Branch A / ", n, ri, ui)
                        if original in display.locations:
                            display.locations[
                                display.location("Branch B / ", n, ri, ui)
                            ] = display.locations[original]
                if da["results"] and (
                    da["retrieval_statistics"] != db["retrieval_statistics"]
                    or da["supporting_fact_progress_evaluator_only"]
                    != db["supporting_fact_progress_evaluator_only"]
                ):
                    lines.append("Run B document and evidence counts (calculated separately):" if plain_language else "Branch B progress (independently calculated):")
                    lines.extend(progress_lines(db, plain_language=plain_language))
                lines.append(cost_line(da.get("model_usage"), f"Run A, Decision {n} cost" if plain_language else f"A{n} cost"))
                lines.append(cost_line(db.get("model_usage"), f"Run B, Decision {n} cost" if plain_language else f"B{n} cost"))
        for label in ("A", "B"):
            view = comparison["views"][label]
            lines.append(
                ("\nRun " if plain_language else "\nBranch ") + label
                + ((f" (Decisions 1 through {common} shown above)" if plain_language else f" (D1-D{common} shown above)") if common else "")
            )
            for decision in view["decisions"][common:]:
                lines.extend(
                    display.decision(
                        decision, f"Branch {label} / ",
                        f"Run {label}, Decision {decision['decision']}" if plain_language else f"{label}{decision['decision']}",
                        **({"plain_language": True} if plain_language else {}),
                    )
                )
            lines.extend(additional_answer(view))
        lines.append("\nAnswer evaluation and measured costs for the two runs:" if plain_language else "\nPaired episode evaluation:")
        for label in ("A", "B"):
            lines.append(evaluation_line(comparison["views"][label], "Run " + label if plain_language else label,
                                         plain_language=plain_language))
        audits.append(
            {
                "question_id": comparison["question_id"],
                "common_prefix_decisions": common,
                "text_references": display.references,
                "location_references": display.locations,
                "identical_results": display.result_references,
            }
        )
    audit = {"renderer_version": PLAIN_DISPLAY_VERSION if plain_language else DISPLAY_VERSION, "cases": audits}
    if plain_language:
        audit["comparison_context_version"] = PLAIN_COMPARISON_VERSION
    return "\n".join(lines), audit


def render_comparisons(comparisons):
    return render_comparisons_with_audit(comparisons)[0]
