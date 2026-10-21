from arag_skillopt.paired_summary import is_informative_pair


def test_identical_pair_is_filtered_but_cost_or_answer_change_is_kept():
    identical = {
        "first_divergence": None,
        "parent_answer_correct": True,
        "candidate_answer_correct": True,
        "token_delta": 0,
        "policy_call_delta": 0,
    }
    changed = {**identical, "token_delta": 12}
    flipped = {**identical, "candidate_answer_correct": False}
    assert not is_informative_pair(identical)
    assert is_informative_pair(changed)
    assert is_informative_pair(flipped)
