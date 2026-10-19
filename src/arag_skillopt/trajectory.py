"""Decision-ordered optimizer view with exact-text backreferences and lineage."""

import copy
import difflib
import json
from collections import defaultdict

from .compact_rendering import DISPLAY_VERSION, render_single
from .observations import result_units
from .support import SupportTracker

RENDERER_VERSION = DISPLAY_VERSION


def decisions_from_raw(raw):
    steps = raw.get("trajectory", [])
    audits = raw.get("decision_audit", [])
    cursor = 0
    decisions = []
    prior_tool_messages = set()
    if not audits:
        groups = defaultdict(list)
        for step in steps:
            if "loop" not in step:
                raise ValueError(
                    "legacy trajectory has no loop; cannot invent decision boundaries"
                )
            groups[step["loop"]].append(step)
        for n, group in sorted(groups.items()):
            decisions.append(
                {
                    "decision": n,
                    "calls": [
                        {"tool": s["tool_name"], "arguments": s["arguments"]}
                        for s in group
                    ],
                    "results": [{"step": s, "visible_at": None} for s in group],
                    "visibility_verified": False,
                    "response": None,
                }
            )
        return decisions
    for index, audit in enumerate(audits):
        message = audit.get("response", {}).get("message", {})
        calls = message.get("tool_calls") or []
        decision = {
            "decision": index + 1,
            "calls": [],
            "results": [],
            "visibility_verified": True,
            "response": message.get("content"),
            "error": audit.get("error"),
            "usage": {
                k: audit.get("response", {}).get(k)
                for k in ("input_tokens", "output_tokens")
            },
        }
        # Never assert full accumulated visibility after a future context crop.
        present = {
            (m.get("tool_call_id"), m.get("content"))
            for m in audit["request"]["messages"]
            if m.get("role") == "tool"
        }
        missing = prior_tool_messages - present
        decision["removed_prior_tool_messages"] = [m[0] for m in sorted(missing)]
        prior_tool_messages |= present
        decision["force_answer_instruction"] = bool(
            audit["request"]["messages"]
            and audit["request"]["messages"][-1]
            .get("content", "")
            .startswith("You have reached the limit.")
        )
        for call in calls:
            func = call["function"]
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"unparsed_arguments": args}
            decision["calls"].append({"tool": func["name"], "arguments": args})
            if cursor >= len(steps):
                decision["results"].append(
                    {
                        "step": {
                            "tool_name": func["name"],
                            "tool_result": "",
                            "error": "no_execution_record",
                        },
                        "visible_at": None,
                    }
                )
                continue
            step = steps[cursor]
            cursor += 1
            if step["tool_name"] != func["name"] or (
                step.get("arguments") != args and "unparsed_arguments" not in args
            ):
                raise ValueError("audit/trajectory tool order mismatch")
            visible_at = next(
                (
                    j + 1
                    for j in range(index + 1, len(audits))
                    if any(
                        m.get("role") == "tool"
                        and m.get("tool_call_id") == call.get("id")
                        and m.get("content") == step["tool_result"]
                        for m in audits[j]["request"]["messages"]
                    )
                ),
                None,
            )
            decision["results"].append(
                {
                    "step": step,
                    "visible_at": visible_at,
                    "raw_step_index": cursor - 1,
                    "tool_call_id": call.get("id"),
                }
            )
        decisions.append(decision)
    if cursor != len(steps):
        raise ValueError("unpaired tool execution records")
    return decisions


class TextHistory:
    def __init__(self):
        self.by_chunk = defaultdict(list)

    def render(self, cid, text, location):
        matches = []
        for old_text, old_location in self.by_chunk[cid]:
            if text == old_text:
                return [
                    {"same_as": old_location, "source_range": [0, len(old_text)]}
                ], False
            # Entire short snippet fragments can also be referenced on READ.
            # Offsets always refer to verbatim older text, not reconstructed claims.
            fragment_start = 0
            for fragment in old_text.split(" ... "):
                cleaned = fragment.removeprefix("... ").removesuffix(" ...").strip()
                if cleaned:
                    old_start = old_text.find(cleaned, fragment_start)
                    new_start = text.find(cleaned)
                    if old_start >= 0 and new_start >= 0:
                        matches.append(
                            (
                                new_start,
                                new_start + len(cleaned),
                                old_location,
                                old_start,
                            )
                        )
                fragment_start += len(fragment) + 5
            for block in difflib.SequenceMatcher(
                None, old_text, text, autojunk=False
            ).get_matching_blocks():
                if block.size >= 20 and text[block.b : block.b + block.size].strip():
                    matches.append(
                        (block.b, block.b + block.size, old_location, block.a)
                    )
        # Prefer the longest exact spans; never summarize or alter remaining text.
        selected = []
        for match in sorted(matches, key=lambda m: (-(m[1] - m[0]), m[0])):
            if not any(match[0] < s[1] and s[0] < match[1] for s in selected):
                selected.append(match)
        segments, position = [], 0
        for start, end, old_location, old_start in sorted(selected):
            if start > position:
                segments.append({"text": text[position:start]})
            segments.append(
                {
                    "same_as": old_location,
                    "source_range": [old_start, old_start + end - start],
                }
            )
            position = end
        if position < len(text):
            segments.append({"text": text[position:]})
        if not segments:
            segments = [{"text": text}]
        self.by_chunk[cid].append((text, location))
        return segments, any(s.get("text", "").strip(" .\n\t") for s in segments)


def build_view(raw, evaluation, support_entry=None, skill_version="current batch"):
    """Return optimizer-safe view and separate audit lineage. No Skill hashes in view."""
    decisions = decisions_from_raw(raw)
    tracker, history = SupportTracker(support_entry), TextHistory()
    seen, read, pending = set(), set(), []
    views, lineage = [], []
    for decision in decisions:
        n = decision["decision"]
        due = [u for at, u in pending if at <= n]
        tracker.observe(due)
        pending = [(at, u) for at, u in pending if at > n]
        for unit in due:
            seen.add(unit["chunk_id"])
            if unit["kind"] == "full":
                read.add(unit["chunk_id"])
        before = tracker.snapshot()
        returned, first_reads, old_new = set(), set(), set()
        result_views, statuses, shown_units = [], [], []
        for result_index, result in enumerate(decision["results"], 1):
            step = result["step"]
            body = step.get("tool_result", "")
            error = step.get("error")
            if error or body.startswith("Error"):
                units, status = [], "tool_error"
            else:
                units, status = result_units(step["tool_name"], body)
                status = status or (
                    "partial_error"
                    if any(u["kind"] == "not_found" for u in units)
                    else "success"
                )
            statuses.append(status)
            rendered_units = []
            for unit_index, unit in enumerate(units, 1):
                cid = unit["chunk_id"]
                loc = f"Decision {n}, result {result_index}, item {unit_index}"
                rendered = {"chunk_id": cid, "kind": unit["kind"]}
                if unit["kind"] in {"snippet", "full"}:
                    rendered["content"], added = history.render(cid, unit["text"], loc)
                    if cid in seen and added:
                        old_new.add(cid)
                else:
                    rendered["message"] = unit["text"]
                rendered_units.append(rendered)
                lineage.append(
                    {
                        "location": loc,
                        "raw_step_index": result.get("raw_step_index"),
                        "tool_result_start": unit.get("start"),
                        "tool_result_end": unit.get("end"),
                        "raw_body": unit["text"],
                        "visible_at": result["visible_at"],
                    }
                )
                if unit["available"]:
                    returned.add(cid)
                    if unit["kind"] == "full" and cid not in read:
                        first_reads.add(cid)
                    if result["visible_at"] is not None:
                        pending.append((result["visible_at"], unit))
                        shown_units.append(unit)
            result_view = {
                "tool": step["tool_name"],
                "outcome": status,
                "items": rendered_units,
                "results_visible_at_decision": result["visible_at"],
            }
            if not units:
                result_view["raw_result"] = body
            result_views.append(result_view)
        # Attribution to this decision's results; later delivery is explicitly recorded.
        projected = copy.deepcopy(tracker)
        hits = projected.observe(shown_units)
        after = projected.snapshot()
        scored = (
            bool(statuses)
            and decision["visibility_verified"]
            and all(s in {"success", "empty"} for s in statuses)
        )
        progress_ok = (
            scored
            and decision["visibility_verified"]
            and all(r["visible_at"] is not None for r in decision["results"])
        )
        stats = {
            "status": "complete"
            if scored
            else "not_applicable"
            if not statuses
            else "unavailable_or_partial",
            "returned_chunks": len(returned) if scored else None,
            "new_chunks": len(returned - seen) if scored else None,
            "new_source_ratio": len(returned - seen) / len(returned)
            if scored and returned
            else None,
            "cumulative_chunks_after_results": len(seen | returned) if scored else None,
            "first_full_reads": len(first_reads) if scored else None,
            "old_chunks_with_new_text": sorted(old_new) if scored else None,
        }
        progress = {
            "evaluable": progress_ok and before["evaluable"],
            "before": before,
            "after_results_visible": after if progress_ok else None,
            "visible_recall_delta": after["visible_recall"] - before["visible_recall"]
            if progress_ok and before["evaluable"]
            else None,
            "read_recall_delta": after["read_recall"] - before["read_recall"]
            if progress_ok and before["evaluable"]
            else None,
            "not_evaluable_reason": None
            if progress_ok and before["evaluable"]
            else "no_retrieval_action"
            if not statuses
            else "mapping_or_visibility_or_execution_unavailable",
        }
        lineage.append({"decision": n, "support_hits": hits})
        views.append(
            {
                "decision": n,
                "visible_before_this_decision": (
                    "Some earlier tool results were removed or changed in this request: "
                    + ", ".join(decision["removed_prior_tool_messages"])
                    + ". Do not assume all earlier Results remain visible; consult the saved raw request."
                    if decision.get("removed_prior_tool_messages")
                    else "Earlier delivered Results; see their decision references."
                    if seen
                    else "No retrieved content confirmed visible."
                ),
                "read_state_before": {
                    "visible_chunk_count": len(seen),
                    "full_read_count": len(read),
                },
                "calls_selected_together": decision["calls"],
                "results": result_views,
                "retrieval_statistics": stats,
                "supporting_fact_progress_evaluator_only": progress,
                "model_response": decision.get("response"),
                "model_usage": decision.get("usage"),
                "model_error": decision.get("error"),
                "forced_answer": decision.get("force_answer_instruction", False),
            }
        )
    end_recall = tracker.snapshot()
    if any(not d["visibility_verified"] for d in decisions):
        end_recall.update(
            evaluable=False,
            visible_hits=None,
            read_hits=None,
            visible_recall=None,
            read_recall=None,
            unavailable_reasons=["missing_actual_model_requests"],
        )
    result = {
        "question_id": str(raw.get("question_id", raw.get("qid", raw.get("id")))),
        "question": raw.get("question"),
        "skill_version": skill_version,
        "decisions": views,
        "episode_evaluation": {
            "correct": evaluation.get("correct"),
            "evaluation_status": evaluation.get("status"),
            "final_answer": raw.get(
                "answer", raw.get("pred_answer", raw.get("final_answer"))
            ),
            "termination": raw.get("termination_reason", "not_recorded"),
            "model_calls": raw.get("policy_calls"),
            "tool_calls": len(raw.get("trajectory", [])),
            "model_usage": raw.get("usage", {"complete": False}),
            "retrieved_text_tokens": raw.get("total_retrieved_tokens"),
            "supporting_fact_recall_at_end": end_recall,
        },
    }
    return result, {"renderer_version": RENDERER_VERSION, "lineage": lineage}


def render_view_with_audit(view, guidance=True, *, plain_language=False):
    return render_single(view, guidance, plain_language=plain_language)


def render_view(view, guidance=True, *, plain_language=False):
    return render_view_with_audit(view, guidance, plain_language=plain_language)[0]
