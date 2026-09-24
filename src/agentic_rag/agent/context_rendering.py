"""Shared, source-preserving context sections and non-semantic progress audit."""
from __future__ import annotations

import json


def span_key(span):
    return (span.get("sentence_id"), span.get("start"), span.get("end"))


def action_summary(record):
    decision = record.decision or record.resolved_decision
    action = decision.action if decision else None
    names = {"CHUNK": "find_passages", "SENTENCE": "find_sentences"}
    if action is None:
        name, arguments = "unparsed tool call", {}
    elif action.type == "SEARCH":
        name, arguments = names.get(action.target.value, "search"), {"query": action.query}
    elif action.type == "EXPAND":
        name = {"ENTITY_MENTIONED_IN_CHUNK": "follow_entity_to_passages",
                "ENTITY_MENTIONED_IN_SENTENCE": "follow_entity_to_sentences"}.get(action.kind.value, "follow")
        arguments = {"entity_ref": getattr(action, "source_ref", None)}
        if action.kind.value not in {"ENTITY_MENTIONED_IN_CHUNK", "ENTITY_MENTIONED_IN_SENTENCE"}:
            arguments["query"] = action.query
    else:
        name, arguments = ("finish" if action.type == "FINISH" else "read_passage"), {}
    observation = record.observation
    status = observation.status.value if observation else "error"
    code = observation.error_code if observation else "unavailable"
    if code == "duplicate_action" or status == "duplicate_action":
        executed, outcome = False, "rejected: repeated action"
        explanation = "The same tool and arguments were already executed. This request was not executed. No new source text was added."
    elif status == "invalid_action" or record.validation_status.value == "invalid":
        executed, outcome = False, "rejected"
        if (record.validation_error or "").startswith("arguments.assessment"):
            explanation = ("The assessment must contain exactly supported_facts and missing_information. "
                           "Both values must be lists of strings, not a string or an object. Empty lists are allowed.")
        elif code in {"source_not_visible", "reference_not_available", "reference_not_evidence"}:
            explanation = "The selected reference is not available for this operation."
        elif code in {"search_pair_not_enabled", "expansion_not_enabled"}:
            explanation = "The selected tool is not available in this interface."
        else:
            explanation = "The request did not satisfy the available tool's input requirements."
        explanation += " This request was not executed. No new source text was added."
    elif status == "error":
        executed, outcome = None, "not completed"
        explanation = ("The result could not fit within the remaining retrieval budget."
                       if code == "retrieved_token_budget_exceeded"
                       else "The tool did not complete successfully.")
        explanation += " No new source text was added."
    else:
        executed, outcome = True, "completed"
        explanation = "The tool completed. Completion does not establish that the question is answered."
    delivery = record.context_audit.get("output_delivery", {})
    returned = delivery.get("returned_references", [])
    new_text = delivery.get("new_source_references", [])
    return {"turn": record.policy_attempt, "tool": name, **arguments,
            "status": outcome, "executed": executed, "explanation": explanation,
            "returned_references": returned, "new_source_references": new_text,
            "returned_count": len(returned), "new_text_source_count": len(new_text)}


def output_delivery(observation, state, input_spans, input_references=()):
    """Record output projection separately from next-turn actual exposure."""
    seen = {span_key(s) for s in input_spans}
    spans = [s for s in observation.metadata.get("projected_source_spans", [])
             if s.get("span_type") == "sentence" and s.get("complete") and s.get("visible")]
    unique = {span_key(s): s for s in spans}
    returned = []
    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            sid, cid = value.get("sentence_id"), value.get("chunk_id")
            source_id = sid if sid in state.eligible_sentence_ids else cid if cid in state.visible_passage_ids else None
            if source_id and isinstance(value.get("text"), str):
                ref = state.reference_registry.ref_for(source_id)
                if ref and ref not in returned:
                    returned.append(ref)
                # A complete passage is one returned unit. Its nested sentence
                # metadata must not expose unpresented sentence labels.
                if source_id == cid and not sid:
                    return
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
    visit(observation.results)
    new_spans = [s for key, s in unique.items() if key not in seen]
    refs = []
    for span in new_spans:
        # Prefer the returned passage label when sentence text arrived within it.
        candidates = [state.reference_registry.ref_for(span.get("chunk_id")),
                      state.reference_registry.ref_for(span.get("sentence_id"))]
        ref = next((r for r in candidates if r in returned), None)
        if ref and ref not in refs:
            refs.append(ref)
    first_visible = [ref for ref in returned if ref not in input_references]
    return {"returned_references": returned, "new_source_references": refs,
            "first_visible_references": first_visible,
            "newly_projected_source_spans": new_spans}


def format_action_summary(attempt):
    """One compact, source-free line for the latest action or earlier history."""
    arguments = []
    if "entity_ref" in attempt:
        arguments.append("entity=" + str(attempt["entity_ref"]))
    if "query" in attempt:
        query = attempt["query"]
        arguments.append("query=" + (json.dumps(query, ensure_ascii=False) if query is not None
                                      else "original question"))
    call = attempt["tool"] + ("(" + ", ".join(arguments) + ")" if arguments else "")
    return (f'{attempt["turn"]}. {call} | {attempt["status"]} | '
            f'returned {attempt["returned_count"]}; new text {attempt["new_text_source_count"]}')


def render_context(
    blocks,
    spans,
    entity_cards,
    trajectory,
    state,
    *,
    require_assessment,
    entity_filter_audit=None,
):
    previously_seen = {span_key(s) for record in trajectory for s in record.visible_source_spans}
    new_spans = [s for s in spans if span_key(s) not in previously_seen]
    new_ids = {s["sentence_id"] for s in new_spans}
    new_blocks = [b for b in blocks if new_ids.intersection(b["sentence_ids"])]
    old_blocks = [b for b in blocks if not new_ids.intersection(b["sentence_ids"])]
    attempts = [action_summary(record) for record in trajectory]
    sections = []
    previous = None
    if require_assessment:
        for record in reversed(trajectory):
            decision = record.decision or record.resolved_decision
            if decision and decision.assessment is not None:
                previous = {"turn": record.policy_attempt, **decision.assessment.model_dump(mode="json")}
                break
        sections.append("PREVIOUS ASSESSMENT\n" + (
            "Your assessment before action " + str(previous["turn"]) +
            ". This is your earlier judgment, not source text; it does not include subsequent results.\n" +
            json.dumps(previous, ensure_ascii=False) if previous else "No previous assessment is available."))
    if attempts:
        latest = attempts[-1]
        sections.append("LAST ACTION AND RESULT\n" + format_action_summary(latest) +
                        "\n" + latest["explanation"] +
                        ("\nNew source text is shown below." if new_spans else "\nNo new source text was added."))
    else:
        sections.append("LAST ACTION AND RESULT\nNo action has been taken.")
    sections.append("NEW SOURCE TEXT\n" + (
        "Complete units containing text not previously shown. A complete passage may include previously shown sentences.\n" +
        "\n\n".join(b["text"] for b in new_blocks) if new_blocks else "None."))
    sections.append("PREVIOUSLY SHOWN SOURCE TEXT\n" + (
        "\n\n".join(b["text"] for b in old_blocks) if old_blocks else "None."))
    if entity_cards:
        sections.append("Visible entity references:\n" + "\n".join(entity_cards))
    sections.append("EARLIER ACTION HISTORY\n" + (
        "\n".join(format_action_summary(item) for item in attempts[:-1])
        if len(attempts) > 1 else "None."))
    sections.append(f"BUDGET\nRemaining decisions: {state.remaining_step_budget}\n"
                    f"Remaining retrieval token estimate: {state.remaining_retrieved_token_budget}")
    if require_assessment:
        sections.append('Return exactly one decision. At the top level, include "supported_facts" '
                        'and "missing_information", each as a list of strings, plus one "action". '
                        "Use these exact key names. Update the brief facts and information gaps from the source text now shown, "
                        "then choose exactly one operation from OPERATIONS AVAILABLE NOW, or finish. "
                        "You may revise your previous assessment. "
                        "If finishing with unresolved gaps, keep those gaps in missing_information.")
    else:
        sections.append("Return exactly one decision. Choose exactly one operation from OPERATIONS AVAILABLE NOW, or finish.")
    audit = {"version": "sectioned-context-v6.2-entity-navigation-filter", "assessment_requested": require_assessment,
             "previous_assessment": previous,
             "newly_visible_source_spans": new_spans,
             "new_source_references": [b["ref"] for b in new_blocks],
             "new_section_references": [b["ref"] for b in new_blocks],
             "old_section_references": [b["ref"] for b in old_blocks],
             "latest_action": attempts[-1] if attempts else None,
             "entity_filter_audit": list(entity_filter_audit or [])}
    return "\n\n".join(sections), audit, attempts
