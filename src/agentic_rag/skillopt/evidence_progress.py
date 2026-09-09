"""Model-free reference progress over the exact saved Policy inputs.

References and source mappings are evaluator-only inputs.  Returned diagnostics
contain IDs and numbers, never source text.  Lexical scores are a word-overlap
proxy, not entailment or correctness; source-span scores measure exposure, not
whether the exposed passage is sufficient to answer the question.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agentic_rag.agent.models import ObservationStatus, StepRecord, ValidationStatus


EVIDENCE_PROGRESS_VERSION = "policy-visible-reference-progress-v2"
LEXICAL_TOKENIZER_VERSION = "nfkc-casefold-words-signed-decimals-v1"
SENTENCE_SPLITTER_VERSION = "punctuation-whitespace-newline-v1"
_WORDS = re.compile(r"[+\-−]?\d+(?:[.,]\d+)*|[^\W_]+", re.UNICODE)
_SENTENCES = re.compile(r"(?<=[.!?。！？])\s+|[\r\n]+")
_METHODS = {
    "sentence": "exact_source",
    "paragraph": "paragraph_span",
    "reference_text": "lexical_f1",
}


class _Unavailable(ValueError):
    """An ID-only reason code; never interpolate reference or source text."""


@dataclass(frozen=True)
class _Exposure:
    unit_id: str
    text: str
    eligible: bool
    preview: bool = False


@dataclass(frozen=True)
class _Span:
    fact_id: str
    unit_id: str
    unit_text: str
    positions: tuple[tuple[frozenset[int], int], ...]


def _tokens(text: str) -> tuple[str, ...]:
    # Preserve negation words and numerical tokens, including their signs.
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("−", "-")
    return tuple(_WORDS.findall(normalized))


def _fragments(text: str) -> tuple[tuple[str, ...], ...]:
    """Pinned, dependency-free splitting; no length cap or GT-based tuning.

    Abbreviations and long sentences can affect this deliberately simple
    segmentation.  In particular, words spread across separate sentences are
    never joined to manufacture a match to a reference statement.
    """

    return tuple(tokens for part in _SENTENCES.split(text) if (tokens := _tokens(part)))


def _f1(first: tuple[str, ...], second: tuple[str, ...]) -> float:
    if not first or not second:
        return 0.0
    overlap = sum((Counter(first) & Counter(second)).values())
    return 2.0 * overlap / (len(first) + len(second))


def _nonwhite(text: str, start: int = 0, end: int | None = None) -> list[int]:
    return [index for index in range(start, len(text) if end is None else end)
            if not text[index].isspace()]


def _nfkc_groups(text: str, start: int, end: int) -> list[tuple[str, frozenset[int]]]:
    """Map normalized characters to all contributing original positions.

    This rare, explicit sidecar path handles compatibility expansion and
    combining characters.  A changed normalized suffix conservatively inherits
    all contributing original positions, so a partial glyph is never credited
    as a complete original character.
    """
    segment = text[start:end]
    if unicodedata.normalize("NFKC", segment) == segment:
        return [(char, frozenset({start + index})) for index, char in enumerate(segment)
                if not char.isspace()]
    rendered = ""
    groups: list[frozenset[int]] = []
    for index in range(len(segment)):
        updated = unicodedata.normalize("NFKC", segment[:index + 1])
        prefix = 0
        while prefix < min(len(rendered), len(updated)) and rendered[prefix] == updated[prefix]:
            prefix += 1
        positions = frozenset({start + index}).union(*groups[prefix:])
        groups = groups[:prefix] + [positions] * (len(updated) - prefix)
        rendered = updated
    return [(char, positions) for char, positions in zip(rendered, groups, strict=True)
            if not char.isspace()]


def _aligned_positions(mapped: Mapping[str, Any], fact_text: str) -> tuple[tuple[frozenset[int], int], ...]:
    unit_text = mapped["unit_text"]
    unit_start, unit_end = mapped["unit_start"], mapped["unit_end"]
    fact_start, fact_end = mapped["fact_start"], mapped["fact_end"]
    normalization = mapped.get("normalization")
    if normalization not in {None, "literal", "NFKC"}:
        raise _Unavailable("unsupported_mapping_normalization")
    if normalization == "NFKC":
        if mapped.get("verification") != "saved_source_sentence_provenance":
            raise _Unavailable("unverified_mapping_normalization")
        unit_groups = _nfkc_groups(unit_text, unit_start, unit_end)
        fact_groups = _nfkc_groups(fact_text, fact_start, fact_end)
        if not unit_groups or [char for char, _ in unit_groups] != [char for char, _ in fact_groups]:
            raise _Unavailable("mapping_text_mismatch")
        required: dict[int, set[int]] = defaultdict(set)
        for (_, unit_positions), (_, fact_positions) in zip(unit_groups, fact_groups, strict=True):
            for fact_position in fact_positions:
                if not fact_text[fact_position].isspace():
                    required[fact_position].update(unit_positions)
        return tuple((frozenset(positions), fact_position) for fact_position, positions in sorted(required.items()))
    unit_positions = _nonwhite(unit_text, unit_start, unit_end)
    fact_positions = _nonwhite(fact_text, fact_start, fact_end)
    if (not unit_positions or "".join(unit_text[i] for i in unit_positions)
            != "".join(fact_text[i] for i in fact_positions)):
        raise _Unavailable("mapping_text_mismatch")
    return tuple((frozenset({unit_position}), fact_position)
                 for unit_position, fact_position in zip(unit_positions, fact_positions, strict=True))


def _prepare_reference(reference: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[_Span]]]:
    kind = reference.get("kind")
    if not isinstance(kind, str) or kind not in _METHODS or reference.get("method") != _METHODS[kind]:
        raise _Unavailable("unsupported_reference_method")
    if reference.get("status") != "ready":
        raise _Unavailable("reference_marked_unavailable")
    raw_facts = reference.get("facts")
    if not isinstance(raw_facts, list) or not raw_facts:
        raise _Unavailable("missing_reference_facts")
    facts: list[dict[str, Any]] = []
    by_unit: dict[str, list[_Span]] = defaultdict(list)
    unit_texts: dict[str, str] = {}
    seen_ids: set[str] = set()
    for raw in raw_facts:
        if not isinstance(raw, Mapping):
            raise _Unavailable("invalid_reference_fact")
        fact_id, text = raw.get("fact_id"), raw.get("text")
        if not isinstance(fact_id, str) or not fact_id or fact_id in seen_ids:
            raise _Unavailable("invalid_reference_fact_id")
        if not isinstance(text, str) or not text.strip():
            raise _Unavailable("empty_reference_fact")
        seen_ids.add(fact_id)
        required = set(_nonwhite(text))
        facts.append({"fact_id": fact_id, "tokens": _tokens(text), "required": required})
        if kind == "reference_text":
            if not facts[-1]["tokens"]:
                raise _Unavailable("reference_has_no_word_tokens")
            continue
        raw_maps = raw.get("mappings")
        if not isinstance(raw_maps, list) or not raw_maps:
            raise _Unavailable("missing_reference_mapping")
        covered: set[int] = set()
        for mapped in raw_maps:
            if not isinstance(mapped, Mapping):
                raise _Unavailable("invalid_reference_mapping")
            unit_id, unit_text = mapped.get("unit_id"), mapped.get("unit_text")
            if not isinstance(unit_id, str) or not unit_id or not isinstance(unit_text, str):
                raise _Unavailable("invalid_reference_mapping")
            coords = [mapped.get(name) for name in ("unit_start", "unit_end", "fact_start", "fact_end")]
            if any(type(value) is not int for value in coords):
                raise _Unavailable("invalid_mapping_coordinates")
            unit_start, unit_end, fact_start, fact_end = coords
            if not (0 <= unit_start < unit_end <= len(unit_text)
                    and 0 <= fact_start < fact_end <= len(text)):
                raise _Unavailable("invalid_mapping_coordinates")
            positions = _aligned_positions(mapped, text)
            if unit_id in unit_texts and unit_texts[unit_id] != unit_text:
                raise _Unavailable("conflicting_mapping_unit_text")
            unit_texts[unit_id] = unit_text
            by_unit[unit_id].append(_Span(fact_id, unit_id, unit_text, positions))
            covered.update(fact_position for _, fact_position in positions)
        # Missing parts cannot silently disappear from the denominator.
        if not required <= covered:
            raise _Unavailable("incomplete_reference_mapping")
    return facts, dict(by_unit)


def _saved_exposures(step: StepRecord) -> list[_Exposure]:
    view = step.policy_view
    if view is None:
        raise _Unavailable("missing_policy_view")
    state = step.state_before
    exposures: list[_Exposure] = []
    for item in view.policy_state.semantic_memory:
        kind = item.node_type
        if kind == "ENTITY":
            continue
        if step.context_reference_map is not None:
            mapped = step.context_reference_map.typed_refs.get(item.ref)
            if mapped is None or mapped.node_type != kind:
                raise _Unavailable("unresolved_policy_reference")
            stable_id = mapped.stable_id
        else:
            stable_id = state.reference_registry.stable_id_for(item.ref, kind)
        visible_ids = state.visible_sentence_ids if kind == "SENTENCE" else state.visible_chunk_ids
        if not stable_id or stable_id not in visible_ids:
            raise _Unavailable("unresolved_policy_reference")
        if kind == "SENTENCE":
            exposures.append(_Exposure(stable_id, item.text, stable_id in state.eligible_sentence_ids))
        else:
            read = stable_id in state.read_chunk_ids
            if read != item.has_been_read:
                raise _Unavailable("inconsistent_read_status")
            if read:
                if item.text is None:
                    raise _Unavailable("missing_read_chunk_text")
                exposures.append(_Exposure(stable_id, item.text, True))
            else:
                for preview in item.previews:
                    exposures.append(_Exposure(stable_id, preview, False, preview=True))
    return exposures


def _literal_span(exposure: _Exposure, unit_text: str) -> tuple[int, int] | None:
    text = exposure.text.strip()
    if not text:
        return None
    if text == unit_text:
        return 0, len(unit_text)
    start = unit_text.find(text)
    if start >= 0:
        if unit_text.find(text, start + 1) >= 0:
            raise _Unavailable("ambiguous_exposed_text_alignment")
        return start, start + len(text)
    # An explicitly truncated preview may have an appended ellipsis; accept
    # only a unique, literally verified prefix.  Never guess an interior span.
    if exposure.preview:
        for suffix in ("…", "..."):
            if text.endswith(suffix):
                prefix = text[:-len(suffix)].rstrip()
                if prefix and unit_text.startswith(prefix) and unit_text.find(prefix, 1) < 0:
                    return 0, len(prefix)
    raise _Unavailable("unmatched_exposed_text_alignment")


class _Pool:
    def __init__(self, kind: str, facts: list[dict[str, Any]], mappings: dict[str, list[_Span]]) -> None:
        self.kind, self.facts, self.mappings = kind, facts, mappings
        self.best = {fact["fact_id"]: 0.0 for fact in facts}
        self.covered: dict[str, set[int]] = {fact["fact_id"]: set() for fact in facts}
        self.unit_positions: dict[str, set[int]] = defaultdict(set)
        self.seen: set[tuple[str, str]] = set()

    def add(self, exposures: Sequence[_Exposure], *, eligible: bool = False) -> None:
        for exposure in exposures:
            if eligible and not exposure.eligible:
                continue
            key = (exposure.unit_id, exposure.text)
            if key in self.seen:
                continue
            self.seen.add(key)
            if self.kind == "reference_text":
                fragments = _fragments(exposure.text)
                for fact in self.facts:
                    fact_id = fact["fact_id"]
                    self.best[fact_id] = max(self.best[fact_id],
                                             max((_f1(fact["tokens"], fragment) for fragment in fragments), default=0.0))
            else:
                mappings = self.mappings.get(exposure.unit_id, [])
                if not mappings:
                    continue  # The complete gold mapping identifies this as a non-gold unit.
                shown = _literal_span(exposure, mappings[0].unit_text)
                if shown is None:
                    continue
                start, end = shown
                self.unit_positions[exposure.unit_id].update(range(start, end))
                for mapped in mappings:
                    self.covered[mapped.fact_id].update(
                        fact_pos for unit_positions, fact_pos in mapped.positions
                        if unit_positions <= self.unit_positions[exposure.unit_id]
                    )

    def snapshot(self) -> dict[str, Any]:
        values: list[dict[str, Any]] = []
        touched = complete = 0
        for fact in self.facts:
            fact_id = fact["fact_id"]
            if self.kind == "reference_text":
                score = self.best[fact_id]
            else:
                obtained = self.covered[fact_id] & fact["required"]
                fraction = len(obtained) / len(fact["required"])
                touched += bool(obtained)
                complete += obtained == fact["required"]
                score = float(obtained == fact["required"]) if self.kind == "sentence" else fraction
            values.append({"fact_id": fact_id, "score": score})
        result = {"score": sum(item["score"] for item in values) / len(values), "per_fact": values}
        if self.kind != "reference_text":
            result.update(touched_fact_count=touched, complete_fact_count=complete)
        return result


def _difference(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {"before": dict(before), "after": dict(after), "delta": after["score"] - before["score"],
            "per_fact_delta": [{"fact_id": right["fact_id"], "delta": right["score"] - left["score"]}
                               for left, right in zip(before["per_fact"], after["per_fact"], strict=True)]}


def _action_type(step: StepRecord) -> str | None:
    decision = step.resolved_decision or step.decision
    return decision.action.type if decision is not None else None


def _execution_reason(step: StepRecord) -> str | None:
    if step.validation_status is not ValidationStatus.VALID:
        return "action_not_executed"
    if _action_type(step) == "FINISH":
        return None
    if step.observation is None:
        return "action_not_executed"
    if step.observation.status in {ObservationStatus.INVALID_ACTION, ObservationStatus.DUPLICATE_ACTION}:
        return "action_not_executed"
    if step.observation.status is ObservationStatus.ERROR:
        return "tool_execution_failed"
    if _action_type(step) not in {"SEARCH", "READ", "EXPAND"}:
        return "action_not_executed"
    return None


def score_reference_progress(
    trajectory: Sequence[StepRecord], reference: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one text-free diagnostic per action and evaluator metadata.

    The next saved Policy input is the first possible exposure of this action's
    result.  Missing required views or unverifiable alignment permanently mark
    cumulative history incomplete; later READs never repair earlier scores.
    A real terminal action with no next input instead has zero policy gain.
    """

    kind, method = reference.get("kind"), reference.get("method")
    metadata: dict[str, Any] = {
        "version": EVIDENCE_PROGRESS_VERSION, "kind": kind, "method": method,
        "tokenizer_version": LEXICAL_TOKENIZER_VERSION if kind == "reference_text" else None,
        "sentence_splitter_version": SENTENCE_SPLITTER_VERSION if kind == "reference_text" else None,
        "available": False, "reason": None, "fact_count": 0, "fact_ids": [],
        "exposure_source": "saved_policy_view_only", "retrieval_text_included": False,
        "meaning": "lexical_overlap_not_correctness" if kind == "reference_text" else "source_text_exposure_not_answer_sufficiency",
        "history_complete": True,
    }
    failure: str | None = None
    try:
        facts, mappings = _prepare_reference(reference)
    except _Unavailable as exc:
        facts, mappings = [], {}
        failure = str(exc)
        metadata["reason"] = failure
    else:
        metadata.update(available=True, fact_count=len(facts), fact_ids=[fact["fact_id"] for fact in facts])
    pools = [_Pool(str(kind), facts, mappings), _Pool(str(kind), facts, mappings)] if facts else []
    rows: list[dict[str, Any]] = []
    history_failure: str | None = None
    for index, step in enumerate(trajectory):
        has_next = index + 1 < len(trajectory)
        row: dict[str, Any] = {
            "step": step.step, "policy_attempt": step.policy_attempt, "kind": kind, "method": method,
            "status": "unavailable", "reason": None,
            "next_policy_saw_result": False if not has_next else None,
            "visible": None, "eligible": None,
        }
        rows.append(row)
        if failure or history_failure:
            row["reason"] = failure or "incomplete_exposure_history"
            continue
        try:
            current = _saved_exposures(step)
            for eligible, pool in enumerate(pools):
                pool.add(current, eligible=bool(eligible))
            before = [pool.snapshot() for pool in pools]
        except _Unavailable as exc:
            history_failure = str(exc)
            row["reason"] = history_failure
            metadata["history_complete"] = False
            continue
        execution_reason = _execution_reason(step)
        if execution_reason:
            row["reason"] = execution_reason
            continue
        action = _action_type(step)
        empty = action != "FINISH" and step.observation is not None and not step.observation.results
        if action == "FINISH" or empty or not has_next:
            row.update(status="ready", reason=("finish_no_retrieval" if action == "FINISH" else
                       "executed_empty" if empty else "terminal_result_not_presented"))
            row["visible"] = _difference(before[0], before[0])
            row["eligible"] = _difference(before[1], before[1])
            # An empty action has no new result to consume; do not imply that
            # missing next-view evidence was reconstructed from state_after.
            if has_next and trajectory[index + 1].policy_view is not None:
                row["next_policy_saw_result"] = True
            continue
        try:
            upcoming = _saved_exposures(trajectory[index + 1])
            for eligible, pool in enumerate(pools):
                pool.add(upcoming, eligible=bool(eligible))
        except _Unavailable as exc:
            history_failure = str(exc)
            row["reason"] = "missing_next_policy_view" if history_failure == "missing_policy_view" else history_failure
            metadata["history_complete"] = False
            continue
        row.update(status="ready", next_policy_saw_result=True)
        row["visible"] = _difference(before[0], pools[0].snapshot())
        row["eligible"] = _difference(before[1], pools[1].snapshot())
    metadata["unavailable_step_count"] = sum(row["status"] != "ready" for row in rows)
    if history_failure:
        metadata["history_incomplete_reason"] = history_failure
    return rows, metadata
