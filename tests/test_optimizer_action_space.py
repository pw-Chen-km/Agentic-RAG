"""Offline regressions for the actual optimizer system messages/tool registry."""

import itertools
import re
import sys
import types
from pathlib import Path

import pytest

from arag_skillopt.action_space import (
    ACTION_DESCRIPTIONS, TERMINAL_ACTION, action_space_contract, configured_actions,
)
from arag_skillopt.phase1 import Phase1
from arag_skillopt.phase1_runtime import make_tools
from arag_skillopt.training_profile import prompt_bundle, validate_runtime_guidance

ROOT = Path(__file__).resolve().parents[1]


def menu_config(keyword=False, entity=True, linear=True):
    return {
        "optimizer_prompt_version": "legacy",
        "keyword_search": {"enabled": keyword},
        "entity_hop": {"enabled": entity},
        "linear_hop": {"enabled": linear, "graph_index_file": "offline-test.json"},
    }


@pytest.mark.parametrize("flags", list(itertools.product((False, True), repeat=3)))
def test_every_model_facing_stage_uses_only_the_enabled_menu(flags):
    cfg = menu_config(*flags)
    engine = object.__new__(Phase1)
    engine.prompts = prompt_bundle(ROOT, cfg, workflow=True)
    engine.tool_guidance = "Exact tool schemas are supplied at runtime."
    engine.action_objective = action_space_contract(configured_actions(cfg))
    skill = {"schema_version": "1", "retrieval_policy": []}
    for stage in ("success", "failure", "merge", "meta", "meta_merge"):
        system = engine.messages(stage, skill, "Observed evidence")[0]["content"]
        declared = re.findall(r"^- ([a-z_]+):", system, re.M)
        assert declared == list(configured_actions(cfg))
        for name in set(ACTION_DESCRIPTIONS) - set(declared):
            assert name not in system
        assert "not a callable tool" in system
        assert "only three tools" not in system


@pytest.mark.parametrize("flags", [None, *itertools.product((False, True), repeat=3)])
def test_real_tool_registry_wiring_matches_menu_without_loading_models(monkeypatch, flags):
    # Only replace heavy constructors; make_tools' real registration/selection
    # code and real ToolRegistry schemas remain under test.
    def stub(name):
        class Tool:
            def __init__(self, *args, **kwargs):
                self.name = name

            def get_schema(self):
                return {"type": "function", "function": {"name": self.name}}
        return Tool

    import arag_skillopt.phase1_runtime as runtime
    monkeypatch.setattr(runtime, "KeywordSearchTool", stub("keyword_search"))
    monkeypatch.setattr(runtime, "ReadChunkTool", stub("read_chunk"))
    for name, cls in (("semantic_search", "SemanticSearchTool"),
                      ("entity_hop", "EntityHopTool"), ("linear_hop", "LinearHopTool")):
        module = types.ModuleType(f"arag.tools.{name}")
        setattr(module, cls, stub(name))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    cfg = {} if flags is None else menu_config(*flags)
    cfg.update(data={"chunks_file": "offline", "index_dir": "offline"},
               embedding={"model": "offline", "device": "cpu"})
    schemas = make_tools(cfg).get_all_schemas()
    assert {s["function"]["name"] for s in schemas} == set(configured_actions(cfg)) - {TERMINAL_ACTION}
    validate_runtime_guidance(cfg, schemas, "", {})


@pytest.mark.parametrize("profile", ["legacy"])
@pytest.mark.parametrize("mismatch", ["missing_hop", "disabled_keyword", "duplicate", "terminal_as_tool"])
def test_preflight_rejects_menu_drift_for_all_current_profiles(profile, mismatch):
    cfg = menu_config()
    cfg["optimizer_prompt_version"] = profile
    names = list(configured_actions(cfg)[:-1])
    if mismatch == "missing_hop":
        names.remove("linear_hop")
    elif mismatch == "disabled_keyword":
        names.append("keyword_search")
    elif mismatch == "duplicate":
        names.append("read_chunk")
    else:
        names.append(TERMINAL_ACTION)
    schemas = [{"function": {"name": name}} for name in names]
    with pytest.raises(ValueError, match="runtime tools differ"):
        validate_runtime_guidance(cfg, schemas, "", {})


def test_standalone_optimizer_templates_no_longer_claim_three_tools():
    from arag_skillopt.prompts import RETRIEVAL_PROMPT, META_PROMPT
    for text in (RETRIEVAL_PROMPT, META_PROMPT, (ROOT / "prompts/retrieval.md").read_text()):
        assert "RUNTIME ACTION SPACE" in text
        assert "three tools" not in text
        assert "only keyword_search" not in text


def test_action_descriptions_are_intuitive_and_relationship_analysis_is_optional():
    assert "meaning" in ACTION_DESCRIPTIONS["semantic_search"]
    assert "concrete" in ACTION_DESCRIPTIONS["entity_hop"]
    assert "bridge" in ACTION_DESCRIPTIONS["linear_hop"]
    assert "full text" in ACTION_DESCRIPTIONS["read_chunk"]
    contract = action_space_contract(("semantic_search", "entity_hop", "linear_hop", "read_chunk", TERMINAL_ACTION))
    assert "unobserved" in contract["objective"]["unobserved_action"]
    assert "more important" in contract["objective"]["action_relationships"]
    from arag_skillopt.prompts import RETRIEVAL_PROMPT, META_PROMPT
    combined = RETRIEVAL_PROMPT + META_PROMPT + (ROOT / "prompts/phase1_common.md").read_text()
    for term in ("substitute", "complement", "repeat", "cost"):
        assert term in combined
    assert "not as a requirement" in combined
