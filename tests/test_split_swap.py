import copy

import pytest

from arag_skillopt.split_swap import plan_swap


def fixture():
    splits = {"train": [], "validation": [], "test": []}
    mapping = {"questions": {}}
    for split, spec in (
        ("train", [("bridge", False), ("comparison", False), ("bridge", True)]),
        ("validation", [("bridge", False)]),
        (
            "test",
            [
                ("bridge", True),
                ("comparison", True),
                ("bridge", True),
                ("bridge", False),
            ],
        ),
    ):
        for i, (kind, mapped) in enumerate(spec):
            qid = f"{split}-{i}"
            row = {
                "id": qid,
                "question": "Question " + qid,
                "question_type": kind,
                "answer": "secret",
                "historical_usage": ["keep-original"],
                "extra": {"preserve": 1},
            }
            splits[split].append(row)
            mapping["questions"][qid] = {
                "question": row["question"],
                "evaluable": mapped,
                "facts": [
                    {
                        "title": "Document",
                        "sentence_id": 1,
                        "status": "mapped" if mapped else "unmapped",
                        "matches": [{"chunk_id": "1"}] if mapped else [],
                    }
                ],
            }
    return splits, mapping


def test_exchange_deterministic_same_type_no_mutation():
    splits, mapping = fixture()
    original = copy.deepcopy(splits)
    result, audit = plan_swap(splits, mapping, expected_swaps=2)
    assert (result, audit) == plan_swap(splits, mapping, expected_swaps=2)
    assert splits == original
    assert result["validation"] == splits["validation"]
    assert audit["after"]["train"]["fully_mapped"] == 3
    assert audit["swap_count"] == 2
    assert {r["id"] for rows in result.values() for r in rows} == {
        r["id"] for rows in splits.values() for r in rows
    }
    for pair in audit["swaps"]:
        ti, si = pair["train_position"], pair["test_position"]
        assert result["train"][ti] == splits["test"][si]
        assert result["test"][si] == splits["train"][ti]


def test_duplicate_family_stops_exchange():
    splits, mapping = fixture()
    splits["train"][0]["question_group_id"] = "same"
    splits["test"][0]["question_group_id"] = "same"
    with pytest.raises(ValueError, match="question family"):
        plan_swap(splits, mapping, expected_swaps=2)


def test_not_enough_same_type_candidates_stops_exchange():
    splits, mapping = fixture()
    splits["test"][1]["question_type"] = "bridge"
    with pytest.raises(ValueError, match="not enough same-type"):
        plan_swap(splits, mapping, expected_swaps=2)


def test_mapping_mismatch_and_swap_count_stop_exchange():
    splits, mapping = fixture()
    with pytest.raises(ValueError, match="expected 35"):
        plan_swap(splits, mapping)
    mapping["questions"]["train-0"]["question"] = "wrong"
    with pytest.raises(ValueError, match="mapping question mismatch"):
        plan_swap(splits, mapping, expected_swaps=2)
