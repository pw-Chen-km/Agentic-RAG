"""Deterministic evidence-mapping-only train/test exchange; no model scores."""

import hashlib
import unicodedata
from collections import Counter


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def validate_splits(splits, mapping):
    ids, questions, groups = set(), set(), set()
    for rows in splits.values():
        for row in rows:
            qid = row["id"]
            text = " ".join(
                unicodedata.normalize("NFKC", row["question"]).casefold().split()
            )
            group = row.get("question_group_id") or qid
            # Stop rather than split duplicate question families or guess a group swap.
            if qid in ids or text in questions or group in groups:
                raise ValueError("duplicate ID, question, or question family: " + qid)
            ids.add(qid)
            questions.add(text)
            groups.add(group)
            entry = mapping["questions"][qid]
            if entry["question"] != row["question"]:
                raise ValueError("mapping question mismatch: " + qid)
            if not isinstance(entry["evaluable"], bool):
                raise TypeError("invalid evaluable status: " + qid)
            if entry["evaluable"] and (
                not entry["facts"]
                or any(
                    f.get("status") != "mapped" or not f.get("matches")
                    for f in entry["facts"]
                )
            ):
                raise ValueError("inconsistent complete mapping: " + qid)


def statistics(splits, mapping):
    return {
        name: {
            "count": len(rows),
            "question_types": dict(
                sorted(Counter(r["question_type"] for r in rows).items())
            ),
            "fully_mapped": sum(
                mapping["questions"][r["id"]]["evaluable"] for r in rows
            ),
            "not_fully_mapped": sum(
                not mapping["questions"][r["id"]]["evaluable"] for r in rows
            ),
        }
        for name, rows in splits.items()
    }


def plan_swap(splits, mapping, seed=42, expected_swaps=35):
    validate_splits(splits, mapping)
    mapped = mapping["questions"]
    outgoing = [
        (i, row)
        for i, row in enumerate(splits["train"])
        if not mapped[row["id"]]["evaluable"]
    ]
    if len(outgoing) != expected_swaps:
        raise ValueError(
            f"expected {expected_swaps} outgoing questions, got {len(outgoing)}"
        )
    order = lambda pair: (sha256(f"{seed}\0{pair[1]['id']}".encode()), pair[1]["id"])
    replacements = []
    for kind in sorted({row["question_type"] for _, row in outgoing}):
        bad = sorted((p for p in outgoing if p[1]["question_type"] == kind), key=order)
        good = sorted(
            (
                (i, row)
                for i, row in enumerate(splits["test"])
                if row["question_type"] == kind and mapped[row["id"]]["evaluable"]
            ),
            key=order,
        )
        if len(good) < len(bad):
            raise ValueError(
                "not enough same-type fully mapped test questions: " + kind
            )
        for (ti, tr), (si, te) in zip(bad, good):
            replacements.append(
                {
                    "question_type": kind,
                    "train_position": ti,
                    "test_position": si,
                    "outgoing_train_id": tr["id"],
                    "incoming_train_id": te["id"],
                    "unmapped_facts": [
                        {
                            "title": f["title"],
                            "sentence_id": f["sentence_id"],
                            "status": f["status"],
                        }
                        for f in mapped[tr["id"]]["facts"]
                        if f["status"] != "mapped"
                    ],
                    "incoming_historical_exposure_status": te.get(
                        "historical_exposure_status"
                    ),
                    "incoming_historical_analysis_status": te.get(
                        "historical_analysis_status"
                    ),
                }
            )
    result = {name: list(rows) for name, rows in splits.items()}
    for pair in replacements:
        ti, si = pair["train_position"], pair["test_position"]
        result["train"][ti], result["test"][si] = (
            splits["test"][si],
            splits["train"][ti],
        )
    validate_splits(result, mapping)
    before, after = statistics(splits, mapping), statistics(result, mapping)
    assert all(before[s]["count"] == after[s]["count"] for s in before)
    assert all(
        before[s]["question_types"] == after[s]["question_types"] for s in before
    )
    assert result["validation"] == splits["validation"]
    assert after["train"]["not_fully_mapped"] == 0
    return result, {
        "version": "phase1-evidence-mapping-swap-v1",
        "seed": seed,
        "algorithm": 'Within each question_type, sort both pools by sha256("seed\\0canonical_id"), then pair; replace at original positions.',
        "selection_uses": [
            "question_type",
            "canonical_id",
            "complete_evidence_mapping",
        ],
        "selection_uses_answer_scores": False,
        "validation_changed": False,
        "row_contents_and_history_preserved": True,
        "positions_are_zero_based": True,
        "before": before,
        "after": after,
        "swap_count": len(replacements),
        "swaps": replacements,
        "limitations": [
            "Training is now conditioned on complete corpus-to-gold mapping; it is not an unfiltered random sample.",
            "Test mapping availability decreases; answer evaluation still includes all test questions.",
            "History annotations are retained; the revised test must not be described as entirely unseen.",
            "Old results use a different split and must not be directly compared as paired results.",
        ],
    }
