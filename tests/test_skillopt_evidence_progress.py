from __future__ import annotations

import copy
import json

import pytest

from agentic_rag.agent.models import (
    Assessment,
    ChunkMemoryItem,
    ContextNodeReference,
    ContextReferenceMap,
    EntityMemoryItem,
    EpisodeState,
    ExpandAction,
    ExpansionKind,
    FinishAction,
    Observation,
    ObservationStatus,
    PolicyDecision,
    PolicyStateView,
    PolicyView,
    ReadAction,
    SearchAction,
    SearchMethod,
    SearchTarget,
    SentenceMemoryItem,
    StepRecord,
    ValidationStatus,
)
from agentic_rag.skillopt.evidence_progress import score_reference_progress


def _state(*, sentences=(), chunks=(), read=(), eligible=(), entities=()):
    state = EpisodeState.initial()
    for stable_id in chunks:
        state.reference_registry.register(stable_id, "CHUNK")
    for stable_id in sentences:
        state.reference_registry.register(stable_id, "SENTENCE")
    for stable_id in entities:
        state.reference_registry.register(stable_id, "ENTITY")
    state.visible_chunk_ids = set(chunks)
    state.visible_sentence_ids = set(sentences)
    state.visible_entity_ids = set(entities)
    state.read_chunk_ids = set(read)
    state.eligible_sentence_ids = set(eligible)
    return state


def _step(number, state, memory=(), *, action="SEARCH", status=ObservationStatus.OK,
          results=None, view=True, validation=ValidationStatus.VALID, frozen=True):
    if action == "FINISH":
        selected = FinishAction(answer="answer text never scored", evidence_refs=["C1"])
    elif action == "READ":
        selected = ReadAction(chunk_ref="C1")
    elif action == "EXPAND":
        selected = ExpandAction(kind=ExpansionKind.ENTITY_MENTIONED_IN_SENTENCE, source_ref="E1")
    else:
        selected = SearchAction(query="query text never scored", method=SearchMethod.BM25,
                                target=SearchTarget.SENTENCE)
    refs = ContextReferenceMap(typed_refs={
        ref: ContextNodeReference(node_type=state.reference_registry.node_type_for_ref(ref),
                                  stable_id=stable_id)
        for ref, stable_id in state.reference_registry.ref_to_stable_id.items()
    })
    return StepRecord(
        step=number, policy_attempt=number,
        decision=PolicyDecision(assessment=Assessment(), action=selected),
        validation_status=validation,
        observation=Observation(status=status, results=[{"result": "not scored"}] if results is None else results),
        state_before=state.model_copy(deep=True), state_after=state.model_copy(deep=True),
        context_reference_map=refs if frozen else None,
        policy_view=PolicyView(policy_state=PolicyStateView(
            step=number - 1, policy_attempts=number - 1, semantic_memory=list(memory), budget="saved budget"
        )) if view else None,
    )


def _chunk(text, *, read=False, ref="C1"):
    return ChunkMemoryItem(ref=ref, has_been_read=read, chunk_position=0,
                           text=text if read else None, previews=[] if read else [text])


def _sentence(text, *, ref="S1"):
    return SentenceMemoryItem(ref=ref, text=text, parent_chunk_ref="C1")


def _lexical(*texts):
    return {"kind": "reference_text", "method": "lexical_f1", "status": "ready",
            "facts": [{"fact_id": f"G{index}", "text": text} for index, text in enumerate(texts, 1)]}


def _mapped(text, *, kind="paragraph", unit_id="chunk-a", unit_text=None):
    unit_text = text if unit_text is None else unit_text
    return {"kind": kind, "method": "paragraph_span" if kind == "paragraph" else "exact_source",
            "status": "ready", "facts": [{"fact_id": "G1", "text": text,
                "mappings": [{"unit_id": unit_id, "unit_text": unit_text, "unit_start": 0,
                              "unit_end": len(unit_text), "fact_start": 0, "fact_end": len(text)}]}]}


def test_preview_then_read_has_visible_one_before_eligible_one():
    text = "Marie Curie won Nobel Prize"
    initial = _state()
    preview = _state(chunks=["chunk-a"])
    read = _state(chunks=["chunk-a"], read=["chunk-a"])
    trajectory = [_step(1, initial), _step(2, preview, [_chunk(text)], action="READ"),
                  _step(3, read, [_chunk(text, read=True)], action="FINISH")]
    original = [step.model_dump(mode="json") for step in trajectory]
    reference = _lexical(text)
    reference_before = copy.deepcopy(reference)
    rows, metadata = score_reference_progress(trajectory, reference)
    assert rows[0]["visible"]["after"]["score"] == 1
    assert rows[0]["eligible"]["after"]["score"] == 0
    assert rows[1]["visible"]["delta"] == 0
    assert rows[1]["eligible"]["delta"] == 1
    assert rows[2]["visible"]["delta"] == 0
    assert metadata["history_complete"]
    assert "touched_fact_count" not in rows[0]["visible"]["after"]
    assert reference == reference_before
    assert [step.model_dump(mode="json") for step in trajectory] == original
    assert text not in json.dumps([rows, metadata])


def test_future_read_never_backfills_expand_gain():
    empty = _state(entities=["entity-a"])
    preview = _state(chunks=["chunk-a"], entities=["entity-a"])
    read = _state(chunks=["chunk-a"], entities=["entity-a"], read=["chunk-a"])
    steps = [_step(1, empty, action="EXPAND"),
             _step(2, preview, [_chunk("unrelated preview")], action="READ"),
             _step(3, read, [_chunk("biopsy confirms diagnosis", read=True)], action="FINISH")]
    rows, _ = score_reference_progress(steps, _lexical("biopsy confirms diagnosis"))
    assert rows[0]["visible"]["delta"] == 0
    assert rows[1]["visible"]["delta"] == 1
    assert rows[1]["eligible"]["delta"] == 1


def test_repeated_text_new_ids_and_repeated_ids_do_not_increase_score():
    first = _state(chunks=["c"], sentences=["s1"], eligible=["s1"])
    second = _state(chunks=["c"], sentences=["s1", "s2"], eligible=["s1", "s2"])
    steps = [_step(1, _state()), _step(2, first, [_sentence("alpha beta")]),
             _step(3, second, [_sentence("alpha beta"), _sentence("alpha beta", ref="S2")]),
             _step(4, second, [_sentence("alpha beta", ref="S2")], action="FINISH")]
    rows, _ = score_reference_progress(steps, _lexical("alpha beta"))
    assert [row["visible"]["delta"] for row in rows] == [1, 0, 0, 0]


def test_pool_score_is_mean_of_per_fact_maximum():
    state = _state(chunks=["c"], sentences=["s"], eligible=["s"])
    rows, _ = score_reference_progress(
        [_step(1, _state()), _step(2, state, [_sentence("biopsy confirms cancer diagnosis")], action="FINISH")],
        _lexical("biopsy confirms diagnosis", "surgery removes lesion"),
    )
    assert rows[0]["visible"]["after"]["score"] == pytest.approx(3 / 7)
    assert rows[0]["visible"]["after"]["per_fact"] == [
        {"fact_id": "G1", "score": pytest.approx(6 / 7)}, {"fact_id": "G2", "score": 0}]


@pytest.mark.parametrize("status,validation", [
    (ObservationStatus.INVALID_ACTION, ValidationStatus.INVALID),
    (ObservationStatus.DUPLICATE_ACTION, ValidationStatus.INVALID),
    (ObservationStatus.ERROR, ValidationStatus.VALID),
])
def test_invalid_and_tool_error_are_na_not_zero(status, validation):
    rows, _ = score_reference_progress([_step(1, _state(), status=status, validation=validation)], _lexical("a"))
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["visible"] is None


def test_empty_execution_is_zero():
    rows, _ = score_reference_progress([_step(1, _state(), results=[])], _lexical("alpha"))
    assert rows[0]["status"] == "ready"
    assert rows[0]["reason"] == "executed_empty"
    assert rows[0]["visible"]["delta"] == 0


def test_terminal_result_not_presented_is_zero_not_missing_view():
    step = _step(1, _state(), results=[{"text": "alpha beta"}])
    step.state_after = _state(chunks=["c"], sentences=["s"], eligible=["s"])
    rows, _ = score_reference_progress([step], _lexical("alpha beta"))
    assert rows[0]["status"] == "ready"
    assert rows[0]["next_policy_saw_result"] is False
    assert rows[0]["reason"] == "terminal_result_not_presented"
    assert rows[0]["visible"]["after"]["score"] == 0


def test_missing_next_view_is_na_and_future_read_cannot_repair_history():
    read = _state(chunks=["chunk-a"], read=["chunk-a"])
    steps = [_step(1, _state()), _step(2, _state(), view=False),
             _step(3, read, [_chunk("alpha beta", read=True)], action="FINISH")]
    rows, metadata = score_reference_progress(steps, _lexical("alpha beta"))
    assert rows[0]["reason"] == "missing_next_policy_view"
    assert rows[0]["next_policy_saw_result"] is None
    assert all(row["visible"] is None for row in rows)
    assert rows[2]["reason"] == "incomplete_exposure_history"
    assert not metadata["history_complete"]


def test_current_missing_view_is_na_even_if_next_view_has_evidence():
    read = _state(chunks=["chunk-a"], read=["chunk-a"])
    rows, _ = score_reference_progress([_step(1, _state(), view=False),
                                       _step(2, read, [_chunk("alpha", read=True)], action="FINISH")],
                                      _lexical("alpha"))
    assert rows[0]["reason"] == "missing_policy_view"
    assert all(row["status"] == "unavailable" for row in rows)


def test_query_entity_assessment_title_answer_and_tool_raw_never_score():
    state = _state(entities=["entity-a"])
    secret = "biopsy confirms diagnosis"
    first = _step(1, state, [EntityMemoryItem(ref="E1", canonical_name=secret)],
                  results=[{"text": secret}])
    first.decision.action.query = secret
    first.decision.assessment.supported_facts = [secret]
    first.decision.assessment.missing_information = [secret]
    second = _step(2, state, [EntityMemoryItem(ref="E1", canonical_name=secret)], action="FINISH")
    second.decision.action.answer = secret
    rows, metadata = score_reference_progress([first, second], _lexical(secret))
    assert rows[0]["visible"]["after"]["score"] == 0
    assert secret not in json.dumps([rows, metadata])


@pytest.mark.parametrize("reference,exposed,expected", [
    ("The duke killed the king", "The king killed the duke", 1),
    ("not -5.0", "not 5.0", 0.5),
    ("not -5.0", "-5.0", 2 / 3),
    ("ＡＬＰＨＡ beta", "alpha BETA", 1),
    ("alpha alpha beta", "alpha beta", 0.8),
])
def test_tokenization_f1_and_known_role_reversal_limitation(reference, exposed, expected):
    state = _state(chunks=["c"], sentences=["s"], eligible=["s"])
    rows, _ = score_reference_progress([_step(1, _state()),
                                       _step(2, state, [_sentence(exposed)], action="FINISH")],
                                      _lexical(reference))
    assert rows[0]["visible"]["after"]["score"] == pytest.approx(expected)


def test_does_not_join_words_from_separate_sentences():
    state = _state(chunks=["chunk-a"], read=["chunk-a"])
    rows, _ = score_reference_progress([_step(1, _state()),
                                       _step(2, state, [_chunk("alpha. beta.", read=True)], action="FINISH")],
                                      _lexical("alpha beta"))
    assert rows[0]["visible"]["after"]["score"] == pytest.approx(2 / 3)


def test_long_sentence_is_not_truncated_to_a_token_window():
    text = " ".join(f"word{index}" for index in range(100))
    state = _state(chunks=["c"], sentences=["s"], eligible=["s"])
    rows, _ = score_reference_progress([_step(1, _state()),
                                       _step(2, state, [_sentence(text)], action="FINISH")], _lexical(text))
    assert rows[0]["visible"]["after"]["score"] == 1


def test_paragraph_partial_then_full_uses_nonwhite_positions():
    text = "Alpha beta. Gamma delta."
    reference = _mapped(text)
    preview = _state(chunks=["chunk-a"])
    read = _state(chunks=["chunk-a"], read=["chunk-a"])
    rows, metadata = score_reference_progress([
        _step(1, _state()), _step(2, preview, [_chunk("Alpha beta.")], action="READ"),
        _step(3, read, [_chunk(text, read=True)], action="FINISH")], reference)
    after = rows[0]["visible"]["after"]
    assert after["score"] == pytest.approx(10 / 21)
    assert after["touched_fact_count"] == 1
    assert after["complete_fact_count"] == 0
    assert rows[1]["visible"]["after"]["score"] == 1
    assert rows[1]["eligible"]["after"]["score"] == 1
    serialized = json.dumps([rows, metadata])
    assert text not in serialized
    assert "unit_text" not in serialized
    assert "mappings" not in serialized


def test_sentence_preview_touched_but_main_score_only_when_full():
    text = "Alpha beta gamma."
    preview = _state(chunks=["chunk-a"])
    read = _state(chunks=["chunk-a"], read=["chunk-a"])
    rows, _ = score_reference_progress([
        _step(1, _state()), _step(2, preview, [_chunk("Alpha beta")], action="READ"),
        _step(3, read, [_chunk(text, read=True)], action="FINISH")], _mapped(text, kind="sentence"))
    assert rows[0]["visible"]["after"]["score"] == 0
    assert rows[0]["visible"]["after"]["touched_fact_count"] == 1
    assert rows[1]["visible"]["after"]["score"] == 1


def test_overlapping_mapped_ranges_are_unioned_and_whitespace_can_differ():
    text = "A B C D"
    reference = _mapped(text, unit_text="A  B C D")
    reference["facts"][0]["mappings"].append(copy.deepcopy(reference["facts"][0]["mappings"][0]))
    state = _state(chunks=["chunk-a"])
    view = ChunkMemoryItem(ref="C1", has_been_read=False, chunk_position=0, previews=["A  B C", "B C D"])
    rows, _ = score_reference_progress([_step(1, _state()), _step(2, state, [view], action="FINISH")], reference)
    assert rows[0]["visible"]["after"]["score"] == 1


def test_partial_source_offsets_across_two_units_cover_one_original_paragraph():
    reference = {"kind": "paragraph", "method": "paragraph_span", "status": "ready", "facts": [
        {"fact_id": "G1", "text": "Alpha beta", "mappings": [
            {"unit_id": "c1", "unit_text": "Header Alpha", "unit_start": 7, "unit_end": 12,
             "fact_start": 0, "fact_end": 5},
            {"unit_id": "c2", "unit_text": "beta Tail", "unit_start": 0, "unit_end": 4,
             "fact_start": 6, "fact_end": 10},
        ]}
    ]}
    first = _state(chunks=["c1"], read=["c1"])
    second = _state(chunks=["c1", "c2"], read=["c1", "c2"])
    rows, _ = score_reference_progress([
        _step(1, _state()), _step(2, first, [_chunk("Header Alpha", read=True)]),
        _step(3, second, [_chunk("Header Alpha", read=True), _chunk("beta Tail", read=True, ref="C2")], action="FINISH"),
    ], reference)
    assert rows[0]["visible"]["after"]["score"] == pytest.approx(5 / 9)
    assert rows[1]["visible"]["delta"] == pytest.approx(4 / 9)
    assert rows[1]["eligible"]["after"]["complete_fact_count"] == 1


def test_ambiguous_preview_alignment_is_na_not_first_occurrence():
    text = "Alpha. Alpha."
    state = _state(chunks=["chunk-a"])
    rows, metadata = score_reference_progress([_step(1, _state()),
                                              _step(2, state, [_chunk("Alpha.")], action="FINISH")],
                                             _mapped(text))
    assert rows[0]["reason"] == "ambiguous_exposed_text_alignment"
    assert rows[0]["visible"] is None
    assert not metadata["history_complete"]


def test_verified_unique_prefix_can_drop_rendered_ellipsis():
    text = "Alpha beta gamma"
    state = _state(chunks=["chunk-a"])
    rows, _ = score_reference_progress([_step(1, _state()),
                                       _step(2, state, [_chunk("Alpha beta…")], action="FINISH")], _mapped(text))
    assert rows[0]["visible"]["after"]["score"] == pytest.approx(9 / 14)


@pytest.mark.parametrize("change,reason", [
    (lambda r: r["facts"][0].update(mappings=[]), "missing_reference_mapping"),
    (lambda r: r["facts"][0]["mappings"][0].update(unit_end=10000), "invalid_mapping_coordinates"),
    (lambda r: r["facts"][0]["mappings"][0].update(unit_start=True), "invalid_mapping_coordinates"),
    (lambda r: r["facts"][0]["mappings"][0].update(unit_text="omega beta"), "mapping_text_mismatch"),
    (lambda r: r["facts"][0]["mappings"][0].update(unit_end=5, fact_end=5), "incomplete_reference_mapping"),
    (lambda r: r["facts"].append({"fact_id": "G2", "text": "unmapped"}), "missing_reference_mapping"),
])
def test_any_required_missing_or_invalid_mapping_makes_whole_reference_unavailable(change, reason):
    reference = _mapped("alpha beta")
    change(reference)
    rows, metadata = score_reference_progress([_step(1, _state())], reference)
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["reason"] == reason
    assert rows[0]["visible"] is None
    assert not metadata["available"]


def test_missing_or_empty_reference_is_not_zero():
    for reference in [_lexical(), {**_lexical("alpha"), "status": "unavailable"}]:
        rows, metadata = score_reference_progress([_step(1, _state())], reference)
        assert rows[0]["visible"] is None
        assert not metadata["available"]


def test_frozen_map_is_authoritative_and_registry_fallback_is_supported():
    state = _state(chunks=["c"], sentences=["s"], eligible=["s"])
    correct = _step(2, state, [_sentence("alpha")], action="FINISH", frozen=False)
    rows, _ = score_reference_progress([_step(1, _state()), correct], _lexical("alpha"))
    assert rows[0]["visible"]["after"]["score"] == 1
    correct.context_reference_map = ContextReferenceMap()
    rows, _ = score_reference_progress([_step(1, _state()), correct], _lexical("alpha"))
    assert rows[0]["reason"] == "unresolved_policy_reference"


def test_read_flag_must_match_saved_state():
    state = _state(chunks=["chunk-a"])
    rows, _ = score_reference_progress([_step(1, _state()),
                                       _step(2, state, [_chunk("alpha", read=True)], action="FINISH")], _lexical("alpha"))
    assert rows[0]["reason"] == "inconsistent_read_status"


def _nfkc_reference(original, runtime):
    reference = _mapped(original, unit_text=runtime)
    reference["facts"][0]["mappings"][0].update(
        normalization="NFKC", verification="saved_source_sentence_provenance"
    )
    return reference


@pytest.mark.parametrize("original,runtime,preview,expected_partial", [
    ("km²", "km2", "km", 2 / 3),
    ("Wait… now", "Wait... now", "Wait.", 0.5),
    ("é", "e\u0301", "e", 0),
])
def test_explicit_nfkc_source_groups_do_not_credit_partial_compatibility_characters(original, runtime, preview, expected_partial):
    state = _state(chunks=["chunk-a"])
    read = _state(chunks=["chunk-a"], read=["chunk-a"])
    rows, metadata = score_reference_progress([
        _step(1, _state()), _step(2, state, [_chunk(preview)], action="READ"),
        _step(3, read, [_chunk(runtime, read=True)], action="FINISH"),
    ], _nfkc_reference(original, runtime))
    assert metadata["available"]
    assert rows[0]["visible"]["after"]["score"] == pytest.approx(expected_partial)
    assert rows[1]["visible"]["after"]["score"] == 1
    assert rows[1]["eligible"]["after"]["score"] == 1


def test_nfkc_group_exposure_can_accumulate_across_separate_verified_presentations():
    state = _state(chunks=["chunk-a"])
    rows, _ = score_reference_progress([
        _step(1, _state()), _step(2, state, [_chunk("f")]),
        _step(3, state, [_chunk("l")], action="FINISH"),
    ], _nfkc_reference("ﬂ", "fl"))
    assert rows[0]["visible"]["after"]["score"] == 0
    assert rows[1]["visible"]["after"]["score"] == 1


def test_nfkc_normalization_requires_explicit_saved_provenance_verification():
    reference = _nfkc_reference("km²", "km2")
    reference["facts"][0]["mappings"][0].pop("verification")
    rows, metadata = score_reference_progress([_step(1, _state())], reference)
    assert rows[0]["reason"] == "unverified_mapping_normalization"
    assert not metadata["available"]
    reference["facts"][0]["mappings"][0].pop("normalization")
    rows, _ = score_reference_progress([_step(1, _state())], reference)
    assert rows[0]["reason"] == "mapping_text_mismatch"
