from pathlib import Path

from arag_skillopt import SkillOptWorkflow, WorkflowConfig, load_rules


class FakeClient:
    def __init__(self, calls):
        self.calls = list(calls)

    def chat(self, **kwargs):
        return {"message": {"tool_calls": [self.calls.pop(0)]}, "input_tokens": 1, "output_tokens": 1}


def call(name, arguments):
    return {"function": {"name": name, "arguments": arguments}}


def test_native_retrieval_and_meta_no_change_are_bounded():
    root = Path(__file__).parents[1]
    rules = load_rules(root / "skills/arag_initial_rules.json")
    client = FakeClient([
        call("no_change", {"reason": "No repeated pattern"}),
        call("no_change", {"reason": "No repeated paired pattern"}),
    ])
    engine = SkillOptWorkflow(
        rules, client,
        (root / "prompts/retrieval.md").read_text(),
        (root / "prompts/meta.md").read_text(),
        WorkflowConfig(reflection_minibatch_size=5),
    )
    assert engine.reflect([{"question_id": "q1"}], "retrieval")["edits"][0]["operation"] == "no_change"
    assert engine.reflect([{"question_id": "q1"}], "meta")["edits"][0]["operation"] == "no_change"
