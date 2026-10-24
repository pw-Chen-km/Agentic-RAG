"""Model-facing hop guidance, including the actual composed Agent request."""

import copy
from pathlib import Path

import pytest

from arag.tools.entity_hop import EntityHopTool
from arag.tools.linear_hop import LinearHopTool
from arag.tools.registry import ToolRegistry
from arag_skillopt import phase1_runtime
from arag_skillopt.action_space import ACTION_DESCRIPTIONS
from arag_skillopt.skill_rules import load_rules, render_formatted_prompt


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "prompts/arag_linear_hop_no_keyword_question_entities.txt"
SKILL = ROOT / "skills/arag_entityhop_question_entities_initial_rules.json"
GUIDE = "## Retrieval tool guide"


def schema_tool(cls, source="question_or_visible_evidence", backend="direct_bm25"):
    # Public schemas do not need a corpus, embeddings, tokenizer, or model.
    tool = object.__new__(cls)
    tool.entity_source = source
    tool.backend = backend
    tool.default_top_k = 5
    tool.max_top_k = 10
    tool.default_max_hops = 2
    tool.max_hops = 2
    return tool


@pytest.mark.parametrize("cls", [EntityHopTool, LinearHopTool])
def test_both_hops_have_the_same_entry_conditions_and_result_contract(cls):
    schema = schema_tool(cls).get_schema()["function"]
    description = schema["description"]
    assert description.startswith("Find passages")
    assert "question or visible evidence" in description
    assert "first retrieval or a follow-up on a newly visible entity" in description
    assert "no prior search or read is required" in description
    assert "abbreviated snippets and Chunk IDs; always use read_chunk" in description
    assert schema["parameters"]["required"] == ["entity"]
    assert schema["parameters"]["properties"]["top_k"]["default"] == 5


@pytest.mark.parametrize("cls", [EntityHopTool, LinearHopTool])
def test_visible_evidence_only_source_is_not_silently_relaxed(cls):
    schema = schema_tool(cls, source="visible_evidence").get_schema()["function"]
    assert "only after a search or read has exposed the entity" in schema["description"]
    assert "no prior search or read is required" not in schema["description"]
    assert "question or visible evidence" not in schema["parameters"]["properties"]["entity"]["description"]


def test_descriptions_explain_the_real_backends_without_requiring_graph_knowledge():
    direct = schema_tool(EntityHopTool).get_schema()["function"]["description"]
    linked = schema_tool(EntityHopTool, backend="entity_linked").get_schema()["function"]["description"]
    linear = schema_tool(LinearHopTool).get_schema()["function"]["description"]
    assert "BM25" in direct and "directly mention" in direct
    assert "aliases" in linked and "BM25" not in linked
    assert "original question" in linear
    assert "passages that do not mention the starting entity" in linear
    assert "do not need to know the connecting entity in advance" in linear
    assert "LinearRAG-style" not in linear
    for name in ("entity_hop", "linear_hop"):
        assert "first retrieval" in ACTION_DESCRIPTIONS[name]


def test_initial_skill_has_no_hop_preference_or_required_coverage():
    text = render_formatted_prompt(PROMPT.read_text(), load_rules(SKILL))
    assert "neither hop is the default" in text
    assert "do not call both automatically" in text
    assert "the connecting entity need not be known in advance" in text
    assert "prefer entity_hop" not in text
    assert "prefer linear_hop" not in text
    assert "use a tool just to cover the menu" in text


def test_fixed_guide_and_latest_prompt_survive_skill_replacement():
    original = PROMPT.read_text()
    fixed_prefix, rest = original.split("## Strategy", 1)
    answer = "## When Answering" + rest.split("## When Answering", 1)[1]
    skill = copy.deepcopy(load_rules(SKILL))
    skill["retrieval_policy"] = [skill["retrieval_policy"][-1]]
    rendered = render_formatted_prompt(original, skill)
    assert rendered.startswith(fixed_prefix)
    assert rendered.endswith(answer)
    assert rendered.count(GUIDE) == 1
    assert "### R01:" not in rendered
    assert "you do not need to know the connecting entity in advance" in rendered
    latest = original.replace(GUIDE, GUIDE + "\nLatest tool-guide revision.")
    assert "Latest tool-guide revision." in render_formatted_prompt(latest, skill)


@pytest.mark.parametrize("placement", ["strategy", "answer", "duplicate"])
def test_fixed_guide_cannot_be_put_in_the_replaceable_strategy_or_duplicated(placement):
    original = PROMPT.read_text()
    prefix, remainder = original.split(GUIDE, 1)
    guide_body, remainder = remainder.split("## Strategy", 1)
    without_guide = prefix + "## Strategy" + remainder
    if placement == "strategy":
        invalid = without_guide.replace("## When Answering", GUIDE + guide_body + "## When Answering")
    elif placement == "answer":
        invalid = without_guide + "\n" + GUIDE + guide_body
    else:
        invalid = GUIDE + "\n" + original
    with pytest.raises(ValueError, match="fixed Retrieval tool guide"):
        render_formatted_prompt(invalid, load_rules(SKILL))


def test_real_rollout_sends_latest_guide_and_both_schemas_to_the_client(monkeypatch):
    class RecordingClient:
        def __init__(self):
            self.requests = []

        def chat(self, **request):
            self.requests.append(copy.deepcopy(request))
            return {"message": {"role": "assistant", "content": "fixture answer"},
                    "input_tokens": 10, "output_tokens": 2, "cost": 0}

    client = RecordingClient()
    monkeypatch.setattr(phase1_runtime, "client", lambda config: client)
    registry = ToolRegistry()
    registry.register(schema_tool(EntityHopTool))
    registry.register(schema_tool(LinearHopTool))
    latest = PROMPT.read_text().replace(GUIDE, GUIDE + "\nActual-request guide revision.")
    skill = load_rules(SKILL)
    rollout = phase1_runtime.ARAGRollout(
        {"llm": {}, "prompt_mode": "formatted", "agent": {"max_loops": 2, "token_budget": 28672}},
        registry, latest,
    )
    raw = rollout({"id": "q-fixture", "question": "Which entity is connected?"}, skill)
    request = raw["decision_audit"][0]["request"]
    assert request == client.requests[0]
    assert request["messages"][0]["content"] == render_formatted_prompt(latest, skill)
    assert "Actual-request guide revision." in request["messages"][0]["content"]
    assert request["tools"] == registry.get_all_schemas()
    assert {s["function"]["name"] for s in request["tools"]} == {"entity_hop", "linear_hop"}
