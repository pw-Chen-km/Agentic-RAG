"""Lossless, human-readable serialization for all four reflection views.

The saved reflection audit stays expanded. Only the optimizer-facing copy uses
ID lookups, tables, and explicit links to identical fields in the same step.
Retrieval text, action order, repeated results, and diagnostic values are never
summarized. ``expand_reflection_input`` makes that claim mechanically checkable.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping


VERSION = "agentic-rag-skillopt-compact-v1"
_STEP_LISTS = ("raw_steps", "action_ledger", "abstract_steps")
_TABLE_FIELDS = {
    "visible_references", "retrieved_units", "acquisition_paths",
    "semantic_memory", "new_visible_units", "results", "previews", "sentences", "paths",
}
_ROOTS = ("episode_outcome", "trajectory")


def _steps(value):
    trajectory = value.get("trajectory", {})
    for name in _STEP_LISTS:
        yield from trajectory.get(name, [])


def _id_field(key: str) -> bool:
    return key == "id" or key.endswith("_id") or key.endswith("_ids")


def _map_ids(value, mapping, key=""):
    if isinstance(value, dict):
        return {k: _map_ids(v, mapping, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_ids(v, mapping, key) for v in value]
    if isinstance(value, str) and _id_field(key):
        return mapping.get(value, value)
    return value


def _collect_ids(value, found, key=""):
    if isinstance(value, dict):
        for k, v in value.items():
            _collect_ids(v, found, k)
    elif isinstance(value, list):
        for v in value:
            _collect_ids(v, found, key)
    elif isinstance(value, str) and value and _id_field(key):
        found.add(value)


def _tables(value, key="", *, expand=False):
    # Each distinct field set has its own column list. This preserves the
    # difference between an absent field and a field explicitly set to null.
    if expand and key in _TABLE_FIELDS and isinstance(value, dict) and set(value) == {"columns", "rows"}:
        columns = value["columns"]
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate reflection table columns")
        return [_tables(dict(zip(columns, row, strict=True)), expand=True) for row in value["rows"]]
    if expand and key in _TABLE_FIELDS and isinstance(value, dict) and set(value) == {"column_sets", "rows"}:
        sets = value["column_sets"]
        if any(len(set(columns)) != len(columns) for columns in sets):
            raise ValueError("Duplicate reflection table columns")
        return [_tables(dict(zip(sets[index], cells, strict=True)), expand=True) for index, cells in value["rows"]]
    if isinstance(value, dict):
        return {k: _tables(v, k, expand=expand) for k, v in value.items()}
    if isinstance(value, list):
        rows = [_tables(v, expand=expand) for v in value]
        if not expand and key in _TABLE_FIELDS and rows and all(isinstance(row, dict) for row in rows):
            columns = sorted(rows[0])
            if all(sorted(row) == columns for row in rows):
                return {"columns": columns, "rows": [[row[k] for k in columns] for row in rows]}
            sets = sorted({tuple(sorted(row)) for row in rows})
            # Keep isolated irregular records as objects; a table is useful
            # when the same field layout actually repeats.
            if len(sets) * 2 <= len(rows):
                indices = {keys: i for i, keys in enumerate(sets)}
                return {"column_sets": [list(keys) for keys in sets],
                        "rows": [[indices[tuple(sorted(row))], [row[k] for k in sorted(row)]] for row in rows]}
        return rows
    return value


def compact_reflection_input(rendered: Mapping[str, Any]) -> dict[str, Any]:
    if "input_format" in rendered:
        raise ValueError("Reflection input is already encoded")
    result = copy.deepcopy(dict(rendered))
    identifiers: set[str] = set()
    for root in _ROOTS:
        _collect_ids(result.get(root), identifiers)

    # E/S/C aliases already used by the Agent are preferred. Ambiguous historic
    # mappings receive separate U aliases instead of silently conflating units.
    ref_ids: dict[str, set[str]] = {}
    id_refs: dict[str, set[str]] = {}
    for step in _steps(result):
        for node in step.get("decision_context", {}).get("visible_references", []):
            ref, stable = node.get("ref"), node.get("stable_id")
            if isinstance(ref, str) and isinstance(stable, str):
                ref_ids.setdefault(ref, set()).add(stable)
                id_refs.setdefault(stable, set()).add(ref)
    aliases = {}
    for stable in sorted(identifiers):
        refs = id_refs.get(stable, set())
        if len(refs) == 1:
            ref = next(iter(refs))
            if len(ref_ids[ref]) == 1 and ref not in identifiers:
                aliases[stable] = ref
    reserved = set(aliases.values()) | identifiers | set(ref_ids)
    number = 0
    for stable in sorted(identifiers - aliases.keys()):
        while True:
            number += 1
            alias = f"U{number}"
            if alias not in reserved:
                break
        aliases[stable] = alias
        reserved.add(alias)

    for step in _steps(result):
        context = step.get("decision_context", {})
        interface = step.get("interface_context", {})
        if ("available_action_space" in interface and "available_action_space" in context
                and interface["available_action_space"] == context["available_action_space"]):
            del interface["available_action_space"]
            interface["available_action_space_from"] = "decision_context.available_action_space"
        if "remaining_budget" in step and "remaining_budget" in context:
            if step["remaining_budget"] == context["remaining_budget"]:
                del step["remaining_budget"]
                step["remaining_budget_from"] = "decision_context.remaining_budget"
            elif step["remaining_budget"] == context["remaining_budget"].get("after"):
                del step["remaining_budget"]
                step["remaining_budget_from"] = "decision_context.remaining_budget.after"

    for root in _ROOTS:
        if root in result:
            result[root] = _tables(_map_ids(result[root], aliases))
    result["input_format"] = {
        "version": VERSION,
        "meaning": (
            "ID lookup rows are [short ID, original ID]. They rename identifiers only; "
            "an ID in this lookup is not necessarily visible or legal at every step. "
            "Only that step's visible_references and available_action_space say what was allowed. "
            "U IDs label recorded objects; they are not new Agent action references. "
            "Table rows follow their columns in order, with the original row order, values and nulls. "
            "When records have different fields, column_sets lists their field layouts; "
            "each row is [zero-based column-set number, values in that column order]. "
            "Fields ending in _from point to an identical field in the SAME step. "
            "Queries, missing_information, answers, retrieval text and standard references are unchanged."
        ),
        "id_lookup": [[alias, stable] for stable, alias in sorted(aliases.items(), key=lambda pair: pair[1])],
    }
    return result


def expand_reflection_input(compact: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the exact expanded audit, rejecting an unknown encoding."""
    result = copy.deepcopy(dict(compact))
    fmt = result.pop("input_format")
    if fmt["version"] != VERSION:
        raise ValueError("Unsupported compact reflection input version")
    reverse = dict(fmt["id_lookup"])
    if len(reverse) != len(fmt["id_lookup"]) or len(set(reverse.values())) != len(reverse):
        raise ValueError("Ambiguous reflection ID lookup")
    for root in _ROOTS:
        if root in result:
            result[root] = _map_ids(_tables(result[root], expand=True), reverse)
    for step in _steps(result):
        context = step.get("decision_context", {})
        interface = step.get("interface_context", {})
        if "available_action_space_from" in interface:
            if interface.pop("available_action_space_from") != "decision_context.available_action_space":
                raise ValueError("Unknown action-space field link")
            interface["available_action_space"] = copy.deepcopy(context["available_action_space"])
        if "remaining_budget_from" in step:
            source = step.pop("remaining_budget_from")
            if source not in {"decision_context.remaining_budget", "decision_context.remaining_budget.after"}:
                raise ValueError("Unknown budget field link")
            budget = context["remaining_budget"]
            step["remaining_budget"] = copy.deepcopy(budget["after"] if source.endswith(".after") else budget)
    return result
