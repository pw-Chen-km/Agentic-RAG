"""Lossless, compact display of already-derived Decision views.

No evidence selection, summarization, scoring or runtime changes happen here.
Short references address reconstructed original bodies, never display markup.
"""

import json

from .tokens import count

DISPLAY_VERSION = "arag-optimizer-context-compact-v2"
PLAIN_DISPLAY_VERSION = "arag-optimizer-context-plain-v6"
PLAIN_DISPLAY_GUIDANCE = (
    "How to read these records: one Decision chooses all its tools before their "
    "results arrive. Each result states the later Decision when it becomes available. "
    "Earlier results remain available unless the record explicitly says they were removed. "
    "Labels such as r1 name text already displayed. [r1] repeats that exact text; "
    "[r1@start:end] repeats characters starting at start and ending just before end "
    "in the original text, not in the printed reference. References save space; "
    "they do not erase tool calls or their costs. New sources means new chunk IDs, "
    "not necessarily useful new information. GT means reference evidence used only "
    "for evaluation after the run. The agent did not know that evaluation. "
    "N/A means unavailable, not zero."
)

DISPLAY_GUIDANCE = (
    "Reading convention: a Decision selects all its calls before receiving their "
    "results. Results become visible only at the indicated later Decision. "
    "Earlier delivered results remain available unless a removal is explicitly "
    "reported. r1 etc. label original result bodies: [r1] repeats that body; "
    "[r1@start:end] repeats its original half-open character range, not the "
    "printed reference markup. Identical-result references preserve Chunk order "
    "and text; they do not erase tool executions or costs. New sources count "
    "distinct Chunk IDs, not semantic usefulness. GT progress is evaluator-only "
    "hindsight; N/A is unknown, not zero."
)


def number(value):
    return "N/A" if value is None else str(value)


class CompactDisplay:
    """One instance per question; Meta shares it across A/B, not across cases."""

    def __init__(self):
        self.bodies = {}
        self.locations = {}
        self.canonical = {}
        self.references = []
        self.result_references = []
        self.calls = {}
        self.results = {}

    @staticmethod
    def location(scope, decision, result, item):
        return f"{scope}Decision {decision}, result {result}, item {item}"

    @staticmethod
    def source(scope, segment):
        source = segment["same_as"]
        return source if source.startswith("Branch ") else scope + source

    def prepare(self, view, scope=""):
        """Verify/reconstruct exact bodies before display; never mutate the view."""
        for decision in view["decisions"]:
            for ri, result in enumerate(decision["results"], 1):
                for ui, item in enumerate(result["items"], 1):
                    if "content" not in item:
                        continue
                    parts = []
                    for segment in item["content"]:
                        if "text" in segment:
                            parts.append(segment["text"])
                            continue
                        source = self.source(scope, segment)
                        if source not in self.bodies:
                            raise ValueError(
                                "unresolved original text reference: " + source
                            )
                        start, end = segment["source_range"]
                        original = self.bodies[source]
                        if (
                            type(start) is not int
                            or type(end) is not int
                            or not 0 <= start <= end <= len(original)
                        ):
                            raise ValueError("invalid original text reference range")
                        parts.append(original[start:end])
                    location = self.location(scope, decision["decision"], ri, ui)
                    self.bodies[location] = "".join(parts)

    def result_key(self, decision, ri, scope):
        result = decision["results"][ri - 1]
        items = [
            (
                item["chunk_id"],
                item["kind"],
                self.bodies.get(self.location(scope, decision["decision"], ri, ui)),
                item.get("message"),
            )
            for ui, item in enumerate(result["items"], 1)
        ]
        return (
            result["tool"],
            result["outcome"],
            tuple(items),
            result.get("raw_result"),
        )

    def behavior_key(self, decision, scope):
        """Recorded behavior only, not costs/hidden state or causality."""
        return (
            json.dumps(decision["calls_selected_together"], sort_keys=True),
            tuple(
                self.result_key(decision, ri, scope)
                for ri in range(1, len(decision["results"]) + 1)
            ),
            tuple(r["results_visible_at_decision"] for r in decision["results"]),
            decision.get("model_response"),
            decision.get("model_error"),
            decision.get("forced_answer", False),
            decision["visible_before_this_decision"],
            json.dumps(decision["read_state_before"], sort_keys=True),
        )

    def body(self, item, location, scope):
        original = self.bodies[location]
        key = (item["chunk_id"], original)
        if key in self.canonical:
            ref = self.canonical[key]
            self.locations[location] = ref
            marker = "[" + ref + "]"
            return marker if count(marker) < count(original) else original
        parts, segments = [], []
        for segment in item["content"]:
            if "text" in segment:
                parts.append(segment["text"])
                segments.append({"text": segment["text"]})
                continue
            source = self.source(scope, segment)
            start, end = segment["source_range"]
            literal = self.bodies[source][start:end]
            ref = self.locations.get(source)
            marker = (
                "[" + ref + "]"
                if ref and start == 0 and end == len(self.bodies[source])
                else f"[{ref}@{start}:{end}]"
                if ref
                else None
            )
            # Tiny exact overlaps are cheaper and clearer as verbatim text.
            if marker and count(marker) < count(literal):
                parts.append(marker)
                segments.append({"ref": ref, "source_range": [start, end]})
            else:
                parts.append(literal)
                segments.append({"text": literal})
        ref = "r" + str(len(self.references) + 1)
        self.canonical[key] = ref
        self.locations[location] = ref
        self.references.append(
            {
                "ref": ref,
                "source_location": location,
                "chunk_id": item["chunk_id"],
                "kind": item["kind"],
                "original_characters": len(original),
                "display_segments": segments,
            }
        )
        return f"{ref}: " + "".join(parts)

    def decision(self, decision, scope="", label=None, cost=True, *, plain_language=False):
        n = decision["decision"]
        label = label or (f"Decision {n}" if plain_language else f"D{n}")
        lines = [f"\nDecision {n}" + (" (" + label + ")" if scope else "")]
        state = decision["read_state_before"]
        before = decision["visible_before_this_decision"]
        if "removed or changed" in before:
            lines.append("Before: " + before)
        else:
            lines.append(
                f"Before: delivered sources={state['visible_chunk_count']}; full reads={state['full_read_count']}."
            )
        calls = decision["calls_selected_together"]
        key = json.dumps(calls, sort_keys=True)
        if calls and key in self.calls:
            lines.append("Calls selected together: identical to " + self.calls[key])
        elif calls:
            lines.append("Calls selected together:")
            for call in calls:
                lines.append(
                    "- "
                    + call["tool"]
                    + "("
                    + json.dumps(
                        call["arguments"], ensure_ascii=False, separators=(",", ":")
                    )
                    + ")"
                )
            self.calls[key] = label
        else:
            lines.append("Calls: none.")
        for ri, result in enumerate(decision["results"], 1):
            at = result["results_visible_at_decision"]
            delivery = (f"Decision {at}" if plain_language else f"D{at}") if at is not None else "NOT confirmed delivered"
            result_label = f"{label}, result {ri}" if plain_language else f"{label}.R{ri}"
            header = (
                f"Results {ri}: {result['tool']}; {result['outcome']} -> {delivery}"
            )
            result_key = self.result_key(decision, ri, scope)
            previous = self.results.get(result_key)
            if previous:
                lines.append(header + "; identical to " + previous[0])
                self.result_references.append(
                    {
                        "result": result_label,
                        "identical_to": previous[0],
                        "visible_at_decision": at,
                    }
                )
                for ui, old_location in enumerate(previous[1], 1):
                    location = self.location(scope, n, ri, ui)
                    if old_location in self.locations:
                        self.locations[location] = self.locations[old_location]
                continue
            lines.append(header)
            locations = []
            for ui, item in enumerate(result["items"], 1):
                location = self.location(scope, n, ri, ui)
                locations.append(location)
                content = (
                    self.body(item, location, scope)
                    if "content" in item
                    else item.get("message", "")
                )
                lines.append(f"Chunk {item['chunk_id']} / {item['kind']} / " + content)
            if "raw_result" in result:
                lines.append(result["raw_result"])
            self.results[result_key] = (result_label, locations)
        if decision["results"]:
            lines.extend(progress_lines(decision, plain_language=plain_language))
        if cost:
            lines.append(cost_line(decision.get("model_usage")))
        response = decision.get("model_response")
        if response or not calls:
            lines.append(
                "Model response:\n" + (str(response) if response is not None else "N/A")
            )
        if decision.get("model_error"):
            lines.append("Model error: " + decision["model_error"])
        if decision.get("forced_answer"):
            lines.append("Runtime forced an answer at this decision.")
        return lines


def cost_line(usage, label="Cost"):
    usage = usage or {}
    return (
        label
        + ": model input="
        + number(usage.get("input_tokens"))
        + "; output="
        + number(usage.get("output_tokens"))
    )


def readable_status(value):
    explanations = {
        "no_gold_mapping": "no reliable reference-evidence positions were supplied",
        "missing_recall_record": "the reference-evidence measurement is missing",
        "missing_actual_model_requests": "the model's actual inputs were not recorded",
        "mapping_or_visibility_or_execution_unavailable": "reference matching, confirmed result delivery, or tool execution is incomplete",
        "no_retrieval_action": "no document-finding or reading tool was used",
        "unavailable_or_partial": "some results could not be counted completely",
        "invalid_coordinate": "a reference sentence position is invalid",
        "ambiguous_source_text": "the reference text matches more than one source",
        "unmapped": "reference text could not be reliably matched to a passage",
        "empty_sentence": "the reference sentence is empty",
    }
    return explanations.get(value, str(value).replace("_", " "))


def progress_lines(decision, *, plain_language=False):
    stats = decision["retrieval_statistics"]
    if stats["status"] == "complete":
        ratio = (
            f"{stats['new_source_ratio']:.1%}"
            if stats["new_source_ratio"] is not None
            else "N/A"
        )
        lines = [
            f"Sources: new {stats['new_chunks']}/{stats['returned_chunks']} ({ratio}); total={stats['cumulative_chunks_after_results']}; first full reads={stats['first_full_reads']}"
        ]
        if stats["old_chunks_with_new_text"]:
            lines[-1] += "; old chunks with added text=" + ",".join(
                stats["old_chunks_with_new_text"]
            )
    else:
        lines = ["Sources: N/A (" + (readable_status(stats["status"]) if plain_language else stats["status"]) + ")."]
    progress = decision["supporting_fact_progress_evaluator_only"]
    if progress["evaluable"]:
        before, after = progress["before"], progress["after_results_visible"]
        total = before["gold_fact_count"]
        lines.append(
            (f"Reference evidence after these results become visible: found {before['visible_hits']}->{after['visible_hits']}/{total}; fully read {before['read_hits']}->{after['read_hits']}/{total}."
             if plain_language else
             f"GT progress after delivery: seen {before['visible_hits']}->{after['visible_hits']}/{total}; READ {before['read_hits']}->{after['read_hits']}/{total}.")
        )
    else:
        if plain_language:
            reasons = [progress["not_evaluable_reason"], *progress["before"]["unavailable_reasons"]]
            lines.append("Reference evidence found/read: N/A (" + "; ".join(
                readable_status(reason) for reason in dict.fromkeys(reasons) if reason
            ) + ").")
            return lines
        lines.append(
            "GT progress: N/A ("
            + str(progress["not_evaluable_reason"])
            + "; facts="
            + number(progress["before"]["gold_fact_count"])
            + "; mapping issues="
            + json.dumps(progress["before"]["unavailable_reasons"])
            + ")."
        )
    return lines


def additional_answer(view):
    """Normally the terminal response IS the answer. Preserve mismatches explicitly."""
    answer = view["episode_evaluation"].get("final_answer")
    if answer is None or any(
        d.get("model_response") == answer for d in view["decisions"]
    ):
        return []
    return [
        "Recorded final answer (different from displayed responses):\n" + str(answer)
    ]


def evaluation_line(view, label="Episode evaluation", *, plain_language=False):
    e = view["episode_evaluation"]
    usage = e.get("model_usage") or {}
    recall = e.get("supporting_fact_recall_at_end") or {
        "evaluable": False,
        "unavailable_reasons": ["missing_recall_record"],
    }
    facts = (
        f"seen {recall['visible_hits']}/{recall['gold_fact_count']}; READ {recall['read_hits']}/{recall['gold_fact_count']}"
        if recall["evaluable"]
        else "N/A " + json.dumps(recall["unavailable_reasons"])
    )
    if plain_language:
        facts = (
            f"found {recall['visible_hits']}/{recall['gold_fact_count']}; fully read {recall['read_hits']}/{recall['gold_fact_count']}"
            if recall["evaluable"] else "N/A (" + "; ".join(
                readable_status(reason) for reason in recall["unavailable_reasons"]
            ) + ")"
        )
    return (
        label
        + ": correct="
        + number(e.get("correct"))
        + "; evaluation="
        + (readable_status(e.get("evaluation_status")) if plain_language else str(e.get("evaluation_status")))
        + "; termination="
        + (readable_status(e.get("termination")) if plain_language else str(e.get("termination")))
        + "; model calls="
        + number(e.get("model_calls"))
        + "; tool calls="
        + number(e.get("tool_calls"))
        + "; model input/output/total="
        + "/".join(
            number(usage.get(k))
            for k in ("input_tokens", "output_tokens", "total_tokens")
        )
        + "; usage complete="
        + str(usage.get("complete", False))
        + "; retrieved text tokens="
        + number(e.get("retrieved_text_tokens"))
        + ("; final reference evidence found/read=" if plain_language else "; final GT recall=")
        + facts
    )


def render_single(view, guidance=True, *, plain_language=False):
    display = CompactDisplay()
    display.prepare(view)
    lines = ([(PLAIN_DISPLAY_GUIDANCE if plain_language else DISPLAY_GUIDANCE)] if guidance else []) + [
        "Question: " + str(view["question"]),
        "Case ID: " + view["question_id"],
        "Skill version: " + view["skill_version"],
    ]
    for decision in view["decisions"]:
        lines.extend(display.decision(decision, plain_language=plain_language))
    lines.extend(additional_answer(view))
    lines.append("\n" + evaluation_line(view, plain_language=plain_language))
    return "\n".join(lines), {
        "renderer_version": PLAIN_DISPLAY_VERSION if plain_language else DISPLAY_VERSION,
        "text_references": display.references,
        "location_references": display.locations,
        "identical_results": display.result_references,
    }
