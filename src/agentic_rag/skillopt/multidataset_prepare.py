"""Prepare audited, label-separated full-pool SkillOpt dataset splits.

This module never builds an index, calls a model, or silently repairs source
annotations.  Historical roles are locked by default. An explicitly requested
fresh split ignores those locks but preserves all history/exposure audit data;
identical-question groups remain inseparable under either policy.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agentic_rag.evaluation.profiles import get_dataset_profile
from agentic_rag.skillopt.lineage import sha256_file


PREPARATION_VERSION = "skillopt-multidataset-preparation-v2"
PREPARATION_SPLIT_SCHEMA_VERSION = "3.1"
SUPPORTED_PREPARATION_VERSIONS = {
    "3.0": "skillopt-multidataset-preparation-v1",
    PREPARATION_SPLIT_SCHEMA_VERSION: PREPARATION_VERSION,
}
SPLITS = ("train", "validation", "test")
HISTORY_USAGES = ("prepared", "executed", "analyzed", "unknown")
HISTORY_POLICIES = ("locked", "fresh")
DATASET_ROLES = ("training_source", "transfer_only")
ReferenceBuilder = Callable[[str, Path, Path], Any]


class PreparationError(ValueError):
    """An input cannot safely be used, with an auditable reason code."""

    def __init__(self, code: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rank(seed: int, dataset: str, key: str) -> tuple[str, str]:
    return (_digest(f"{seed}\0{dataset}\0{key}"), key)


def _load(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    value = _load(path)
    if isinstance(value, dict):
        for key in ("data", "examples", "records", "questions"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise PreparationError("invalid_source_shape", f"{path} must contain object rows")
    return value


def _source_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or row.get("_id") or row.get("qid") or "")


def _equivalent_id(dataset: str, value: str) -> str:
    return value.removeprefix("musique_") if dataset == "musique" else value


def _join_source(
    canonical: Mapping[str, Any], rows: list[dict[str, Any]], dataset: str,
    *, prefer_row_index: bool,
) -> tuple[int, dict[str, Any]]:
    """Join by verified source row or uniquely by source ID and exact Q/A."""
    source_id = str(canonical.get("source_question_id") or canonical.get("question_id")
                    or canonical.get("id") or "")

    def matches(row: Mapping[str, Any]) -> bool:
        return (
            _equivalent_id(dataset, _source_id(row)) == _equivalent_id(dataset, source_id)
            and row.get("question") == canonical.get("question")
            and row.get("answer") == canonical.get("answer")
        )

    row_index = canonical.get("source_row_index")
    if prefer_row_index and row_index is not None:
        if type(row_index) is not int or not 0 <= row_index < len(rows):
            raise PreparationError("invalid_source_row", "Invalid canonical source_row_index")
        if not matches(rows[row_index]):
            raise PreparationError(
                "source_row_mismatch", "Source row ID/question/answer does not match canonical row",
                {"canonical_id": canonical.get("question_id", canonical.get("id")),
                 "source_row_index": row_index},
            )
        return row_index, rows[row_index]
    candidates = [(index, row) for index, row in enumerate(rows) if matches(row)]
    if len(candidates) != 1:
        raise PreparationError(
            "ambiguous_source_join", "Source ID plus exact question/answer must identify one row",
            {"canonical_id": canonical.get("question_id", canonical.get("id")),
             "matching_rows": [index for index, _ in candidates]},
        )
    return candidates[0]


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": path.resolve().as_posix(), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size}


def _snapshot(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(item.resolve() for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            files.add(path.resolve())
        else:
            raise PreparationError("missing_input", f"Input does not exist: {path}")
    return {path.as_posix(): _file_record(path) for path in sorted(files)}


def _write(path: Path, value: Any, *, jsonl: bool = False) -> None:
    text = ("".join(_canonical_json(row) + "\n" for row in value) if jsonl
            else json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n")
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise PreparationError("output_exists", f"Refusing to overwrite a different output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents two preparation processes from overwriting.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def build_reference_text(
    row: Mapping[str, Any], *, question_id: str, source_record: Mapping[str, Any],
    source_row_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Strictly preserve source evidence strings; deduplicate normalized exact text."""
    evidence = row.get("evidence")
    issues: list[dict[str, Any]] = []
    if not isinstance(evidence, list) or not evidence:
        issues.append({"code": "missing_reference_text", "question_id": question_id})
    elif any(not isinstance(text, str) or not normalized_text(text) for text in evidence):
        issues.append({"code": "malformed_reference_text", "question_id": question_id})
    if issues:
        return {"kind": "reference_text", "method": "lexical_f1",
                "status": "unavailable", "reason": issues[0]["code"], "facts": []}, issues
    facts: list[dict[str, Any]] = []
    by_normalized: dict[str, dict[str, Any]] = {}
    for index, text in enumerate(evidence):
        normalized = normalized_text(text)
        if normalized in by_normalized:
            by_normalized[normalized]["source"]["evidence_indices"].append(index)
            by_normalized[normalized]["source_indices"].append(index)
            continue
        fact = {
            "fact_id": f"reference:{index:04d}", "text": text, "source_indices": [index],
            "source": {"source_sha256": source_record["sha256"],
                       "source_row_index": source_row_index,
                       "evidence_indices": [index]},
            "mappings": [],
        }
        by_normalized[normalized] = fact
        facts.append(fact)
    return {"kind": "reference_text", "method": "lexical_f1",
            "status": "ready", "reason": None, "facts": facts}, []


def _invalid_2wiki_coordinates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Only invalid source coordinates cause exclusion; missing mapping does not."""
    context = row.get("context")
    supporting = row.get("supporting_facts")
    if not isinstance(context, list) or not isinstance(supporting, list):
        return []  # The mapper reports unavailable labels; retain the question.
    titles: dict[str, list[list[Any]]] = defaultdict(list)
    for item in context:
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], list):
            titles[str(item[0])].append(item[1])
    invalid = []
    for fact_index, item in enumerate(supporting):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        title, position = item
        candidates = titles.get(str(title), [])
        if (len(candidates) == 1 and
                (type(position) is not int or not 0 <= position < len(candidates[0]))):
            invalid.append({"fact_index": fact_index, "title": str(title),
                            "sentence_index": position, "sentence_count": len(candidates[0])})
    return invalid


def _history_entries(value: Any, *, role: str, key: str | None = None) -> list[Any]:
    if key:
        for part in key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise PreparationError("history_key_missing", f"History key not found: {key}")
            value = value[part]
    if isinstance(value, Mapping):
        if isinstance(value.get("splits"), Mapping) and role in value["splits"]:
            value = value["splits"][role]
        if isinstance(value, Mapping):
            for candidate in ("items", "selected_ids", "ordered_question_ids", "ids", "questions", "records", "data"):
                if isinstance(value.get(candidate), list):
                    value = value[candidate]
                    break
    if not isinstance(value, list):
        raise PreparationError("invalid_history_shape", "History must resolve to a list of IDs or rows")
    return value


def _history(
    specs: Sequence[Mapping[str, Any]], all_rows: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_id = {row["id"]: row for row in all_rows}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_source[row["source_question_id"]].append(row)
        if row.get("source") == "musique" and row["source_question_id"].startswith("musique_"):
            by_source[row["source_question_id"].removeprefix("musique_")].append(row)
    assignments: dict[str, set[str]] = defaultdict(set)
    events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    artifacts, problems = [], []
    for spec in specs:
        role, usage = spec.get("role"), spec.get("usage", "unknown")
        if role not in SPLITS or usage not in HISTORY_USAGES:
            raise PreparationError("invalid_history_contract", f"Invalid role/usage: {role}/{usage}")
        path = Path(str(spec["path"])).resolve()
        artifact = {**_file_record(path), "role": role, "usage": usage,
                    "format": spec.get("format", "auto"), "key": spec.get("key")}
        artifacts.append(artifact)
        for entry in _history_entries(_load(path), role=role, key=spec.get("key")):
            raw = entry if isinstance(entry, Mapping) else {"id": str(entry)}
            question_id = str(raw.get("id") or raw.get("question_id") or raw.get("qid") or raw.get("_id") or "")
            candidates = [by_id[question_id]] if question_id in by_id else list(by_source.get(question_id, []))
            if not candidates and raw.get("source_question_id") is not None:
                candidates = list(by_source.get(str(raw["source_question_id"]), []))
            # Historical canonical IDs can change when a corpus is rebuilt.
            # A row number alone is not identity: also require its exact Q/A.
            if not candidates and type(raw.get("source_row_index")) is int and "question" in raw and "answer" in raw:
                candidates = [row for row in all_rows
                              if row.get("source_row_index") == raw["source_row_index"]
                              and row["question"] == raw["question"] and row["answer"] == raw["answer"]]
            if not candidates and isinstance(raw.get("question"), str):
                candidates = [row for row in all_rows
                              if normalized_text(row["question"]) == normalized_text(raw["question"])]
            if "question" in raw:
                candidates = [row for row in candidates
                              if normalized_text(row["question"]) == normalized_text(str(raw["question"]))]
            if "answer" in raw:
                candidates = [row for row in candidates if row["answer"] == raw["answer"]]
            groups = {row["question_group_id"] for row in candidates}
            if not candidates or len(groups) != 1:
                problems.append({"code": "unresolved_history_item", "history_path": path.as_posix(),
                                 "id": question_id, "matching_ids": [row["id"] for row in candidates]})
                continue
            row = sorted(candidates, key=lambda row: row["id"])[0]
            group = row["question_group_id"]
            assignments[group].add(str(role))
            event = {"path": path.as_posix(), "role": role, "usage": usage,
                     "matched_id": row["id"], "source_sha256": artifact["sha256"]}
            if event not in events[group]:
                events[group].append(event)
    conflicts = [{"question_group_id": group, "roles": sorted(roles),
                  "ids": sorted(row["id"] for row in all_rows if row["question_group_id"] == group),
                  "questions": sorted({row["question"] for row in all_rows if row["question_group_id"] == group}),
                  "events": events[group]}
                 for group, roles in sorted(assignments.items()) if len(roles) > 1]
    locks = {group: next(iter(roles)) for group, roles in assignments.items() if len(roles) == 1}
    report = {"artifacts": artifacts, "conflicts": conflicts, "unresolved_items": problems,
              "analysis_status_meaning": {
                  "analyzed": "At least one supplied historical record explicitly declares analysis.",
                  "no_analysis_record": "No analysis record was found in the supplied inventory; this does not prove never seen.",
                  "unknown": "At least one record has unknown usage and no explicit analysis record resolves it.",
              },
              "usage_counts": dict(sorted(Counter(event["usage"] for group in events.values()
                                                   for event in group).items())),
              "groups": [{"question_group_id": group, "events": events[group]}
                         for group in sorted(events)]}
    return locks, dict(events), report


def split_rows(
    rows: list[dict[str, Any]], *, dataset: str, seed: int = 42,
    locks: Mapping[str, str] | None = None,
    history_policy: str = "locked",
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Deterministic grouped stratification, with explicit quota deviations.

    The greedy objective minimizes squared row and per-type quota errors at
    each assignment, followed by improving single-group moves.  It is not
    advertised as a global combinatorial optimum. No group is ever split.
    """
    if history_policy not in HISTORY_POLICIES:
        raise PreparationError("invalid_history_policy", f"Unsupported history policy: {history_policy}")
    supplied_locks = dict(locks or {})
    locks = supplied_locks if history_policy == "locked" else {}
    n = len(rows)
    targets = {"train": n // 5, "validation": n // 5, "test": n - 2 * (n // 5)}
    types = Counter(row["question_type"] for row in rows)
    type_targets = {
        kind: {"train": count // 5, "validation": count // 5,
               "test": count - 2 * (count // 5)} for kind, count in types.items()
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["question_group_id"]].append(row)
    assigned: dict[str, str] = {group: role for group, role in locks.items() if group in groups}
    counts = Counter()
    type_counts: dict[str, Counter[str]] = {role: Counter() for role in SPLITS}

    def update(group: str, role: str, sign: int) -> None:
        counts[role] += sign * len(groups[group])
        for row in groups[group]:
            type_counts[role][row["question_type"]] += sign

    for group, role in assigned.items():
        update(group, role, 1)
    exceeded = {role: {"frozen_count": counts[role], "target": targets[role]}
                for role in SPLITS if counts[role] > targets[role]}

    def objective() -> tuple[int, int]:
        # Overall counts have priority; within equal totals, improve strata.
        total_error = sum((counts[role] - targets[role]) ** 2 for role in SPLITS)
        type_error = sum((type_counts[role][kind] - type_targets[kind][role]) ** 2
                         for kind in types for role in SPLITS)
        return total_error, type_error

    order = sorted((group for group in groups if group not in assigned),
                   key=lambda group: (-len(groups[group]), *_rank(seed, dataset, group)))
    for group in order:
        choices = []
        for role in SPLITS:
            update(group, role, 1)
            choices.append((objective(), SPLITS.index(role), role))
            update(group, role, -1)
        role = min(choices)[2]
        assigned[group] = role
        update(group, role, 1)
    # Deterministically improve feasible grouped allocation without moving locks.
    for _ in range(3):
        changed = False
        for group in order:
            old = assigned[group]
            before = objective()
            best = (before, SPLITS.index(old), old)
            update(group, old, -1)
            for role in SPLITS:
                update(group, role, 1)
                candidate = (objective(), SPLITS.index(role), role)
                if candidate[0] < best[0]:
                    best = candidate
                update(group, role, -1)
            chosen = best[2]
            update(group, chosen, 1)
            assigned[group] = chosen
            changed |= chosen != old
        if not changed:
            break
    result = {role: sorted([row for group, items in groups.items() if assigned[group] == role
                            for row in items], key=lambda row: _rank(seed, dataset, row["id"]))
              for role in SPLITS}
    return result, {
        "algorithm": "grouped-sha256-greedy-squared-quota-error-v1", "seed": seed,
        "ratios": {"train": 0.2, "validation": 0.2, "test": 0.6},
        "rounding": "floor train and validation; remainder to test",
        "targets": targets, "actual": {role: len(result[role]) for role in SPLITS},
        "quota_deviation": {role: len(result[role]) - targets[role] for role in SPLITS},
        "per_type_targets": type_targets,
        "history_policy": history_policy,
        "preserves_historical_roles": history_policy == "locked",
        "historical_lock_count": len(supplied_locks),
        "applied_historical_lock_count": len(locks),
        "frozen_roles_over_nominal_quota": exceeded,
        "group_count": len(groups), "largest_group": max(map(len, groups.values()), default=0),
        "normalization": "Unicode NFKC, casefold, whitespace collapse; punctuation retained",
        "duplicate_groups": [{"question_group_id": group,
                              "ids": sorted(row["id"] for row in items),
                              "answer_variant_count": len({row["answer"] for row in items}),
                              "assigned_split": assigned[group]}
                             for group, items in sorted(groups.items()) if len(items) > 1],
    }


def _path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _default_reference_builder(dataset: str, substrate: Path, raw_source: Path) -> Any:
    from agentic_rag.skillopt.reference_mapping import build_reference_mapper
    return build_reference_mapper(dataset, substrate, raw_source)


def _prepare_dataset(
    spec: Mapping[str, Any], output: Path, *, base: Path, seed: int,
    reference_builder: ReferenceBuilder | None,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    dataset = str(spec["dataset"])
    profile = get_dataset_profile(dataset)
    history_policy = str(spec.get("history_policy", "locked"))
    dataset_role = str(spec.get("dataset_role", "training_source"))
    if history_policy not in HISTORY_POLICIES:
        raise PreparationError("invalid_history_policy", f"Unsupported history policy: {history_policy}")
    if dataset_role not in DATASET_ROLES:
        raise PreparationError("invalid_dataset_role", f"Unsupported dataset role: {dataset_role}")
    questions = _path(spec["questions"], base)
    substrate = _path(spec["substrate"], base)
    raw_source = _path(spec.get("raw_source", spec["questions"]), base)
    history_specs = [{**item, "path": _path(item["path"], base).as_posix()}
                     for item in spec.get("history", [])]
    inputs = [questions, raw_source, substrate] + [Path(item["path"]) for item in history_specs]
    before = _snapshot(inputs)
    source_record = _file_record(questions)
    raw_record = _file_record(raw_source)
    source_rows = _rows(questions)
    raw_rows = source_rows if raw_source == questions else _rows(raw_source)
    canonical_path = substrate / "evaluation" / "benchmark_questions.parquet"
    canonical_rows = pq.read_table(canonical_path).to_pylist()
    if not canonical_rows:
        raise PreparationError("empty_canonical_pool", "Canonical question pool is empty")
    manifest = _load(substrate / "manifest.json")
    if manifest.get("dataset") != dataset:
        raise PreparationError("dataset_mismatch", "Substrate dataset does not match specification")
    mapper = None
    if dataset not in ("medical", "novel"):
        mapper = (reference_builder or _default_reference_builder)(dataset, substrate, raw_source)
    rows, archive, exclusions, gaps = [], [], [], []
    seen_ids: set[str] = set()
    for canonical in canonical_rows:
        question_id = str(canonical.get("question_id") or "")
        if not question_id or question_id in seen_ids:
            raise PreparationError("duplicate_canonical_id", "Canonical question IDs must be nonempty and unique")
        seen_ids.add(question_id)
        source_index, source_row = _join_source(canonical, source_rows, dataset, prefer_row_index=True)
        raw_index, raw_row = (_join_source(canonical, raw_rows, dataset, prefer_row_index=False)
                              if raw_source != questions else (source_index, source_row))
        question, answer = canonical.get("question"), canonical.get("answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            raise PreparationError("invalid_question_answer", f"Invalid question/answer: {question_id}")
        question_type = str(canonical.get("question_type") or "")
        if question_type not in profile.allowed_task_types:
            raise PreparationError("invalid_question_type", f"Invalid question type: {question_type}")
        row = {"id": question_id, "question": question, "answer": answer,
               "question_type": question_type, "scope_id": str(canonical["scope_id"]),
               "source": dataset, "source_question_id": _source_id(source_row),
               "source_row_index": source_index, "raw_source_row_index": raw_index,
               "question_group_id": _digest(f"{dataset}\0{normalized_text(question)}")}
        archived = {"id": question_id, "source_question_id": _source_id(source_row),
                    "source_row_index": source_index, "raw_source_row_index": raw_index,
                    "questions_sha256": source_record["sha256"],
                    "raw_source_sha256": raw_record["sha256"],
                    "source_record": source_row, "raw_source_record": raw_row}
        invalid = _invalid_2wiki_coordinates(raw_row) if dataset == "2wikimultihop" else []
        if invalid:
            exclusion = {"id": question_id, "reason": "invalid_gold_coordinate", "coordinates": invalid}
            exclusions.append(exclusion)
            archived["exclusion"] = exclusion
            archive.append(archived)
            continue
        if dataset in ("medical", "novel"):
            reference, issues = build_reference_text(source_row, question_id=question_id,
                                                     source_record=source_record,
                                                     source_row_index=source_index)
        else:
            reference, issues = mapper.for_question(row, raw_index)
        row["reference_evidence"] = reference
        archived["reference_evidence"] = reference
        archive.append(archived)
        gaps.extend({**issue, "question_id": question_id} for issue in issues)
        if reference.get("status") != "ready" and not issues:
            gaps.append({"question_id": question_id, "code": "reference_unavailable",
                         "reason": reference.get("reason")})
        rows.append(row)

    # Resolve history against all canonical rows including excluded IDs. This
    # preserves their audit events without misclassifying known exclusions as
    # unresolvable historical data.
    history_rows = list(rows)
    included_ids = {row["id"] for row in rows}
    for canonical in canonical_rows:
        if canonical["question_id"] not in included_ids:
            source_index, source_row = _join_source(canonical, source_rows, dataset, prefer_row_index=True)
            history_rows.append({"id": canonical["question_id"], "question": canonical["question"],
                                 "answer": canonical["answer"], "source_question_id": _source_id(source_row),
                                 "source": dataset,
                                 "source_row_index": source_index,
                                 "question_group_id": _digest(f"{dataset}\0{normalized_text(canonical['question'])}")})
    locks, events, history_report = _history(history_specs, history_rows)
    history_report.update({
        "history_policy": history_policy,
        "historical_roles_enforced": history_policy == "locked",
        "conflicts_block_splitting": history_policy == "locked",
        "unresolved_items_block_splitting": history_policy == "locked",
        "fresh_split_is_not_unseen_data": history_policy == "fresh",
    })
    ambiguous_history_ids = {
        str(identifier) for issue in history_report["unresolved_items"]
        for identifier in issue.get("matching_ids", [])
    }
    for row in rows:
        group_events = events.get(row["question_group_id"], [])
        row["history_usage"] = sorted({event["usage"] for event in group_events})
        row["historical_roles"] = sorted({event["role"] for event in group_events})
        row["history_ambiguous_match"] = row["id"] in ambiguous_history_ids
        row["historical_analysis_status"] = (
            "analyzed" if "analyzed" in row["history_usage"] else
            "unknown" if "unknown" in row["history_usage"] or row["history_ambiguous_match"] else "no_analysis_record"
        )
        row["history_records_found"] = bool(group_events)
        row["historical_exposure_status"] = (
            "exposed" if set(row["history_usage"]) & {"executed", "analyzed"} else
            "unknown" if "unknown" in row["history_usage"] or row["history_ambiguous_match"] else
            "prepared_only" if "prepared" in row["history_usage"] else "no_record"
        )
    duplicate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        duplicate_groups[row["question_group_id"]].append(row)
    duplicate_report = {
        "normalization": "Unicode NFKC, casefold, whitespace collapse; punctuation retained",
        "group_count": len(duplicate_groups),
        "groups": [{"question_group_id": group, "ids": sorted(row["id"] for row in items),
                    "questions": sorted({row["question"] for row in items}),
                    "answer_variant_count": len({row["answer"] for row in items})}
                   for group, items in sorted(duplicate_groups.items()) if len(items) > 1],
    }
    split_error = None
    selected, selection = None, None
    if history_policy == "locked" and (history_report["conflicts"] or history_report["unresolved_items"]):
        split_error = {"code": "history_conflict_or_unresolved", "details": {
            "conflicts": history_report["conflicts"], "unresolved_items": history_report["unresolved_items"]}}
    else:
        try:
            selected, selection = split_rows(rows, dataset=dataset, seed=seed, locks=locks,
                                             history_policy=history_policy)
        except PreparationError as exc:
            split_error = {"code": exc.code, "message": str(exc), "details": exc.details}
    after = _snapshot(inputs)
    if before != after:
        raise PreparationError("inputs_changed", "A source or index changed during preparation")
    inference_ready = selected is not None
    reference_progress_ready = bool(rows) and not gaps
    training_ready = (dataset_role == "training_source" and inference_ready
                      and reference_progress_ready)
    report = {
        "version": PREPARATION_VERSION, "dataset": dataset,
        "history_policy": history_policy, "dataset_role": dataset_role,
        "canonical_count": len(canonical_rows), "eligible_count": len(rows),
        "excluded_count": len(exclusions), "reference_ready_count": sum(
            row["reference_evidence"].get("status") == "ready" for row in rows),
        "mapping_issue_count": len(gaps), "split_status": "ready" if selected is not None else "blocked",
        "split_error": split_error, "all_four_arms_ready": training_ready,
        "inference_ready": inference_ready,
        "reference_progress_ready": reference_progress_ready,
        "training_ready": training_ready,
        "inputs_unchanged": True, "input_files": before,
        "model_calls": 0, "indexes_rebuilt": False,
        "mapper": getattr(mapper, "metadata", None),
    }
    _write(output / "hidden_reference_archive.jsonl", archive, jsonl=True)
    _write(output / "mapping_gaps.jsonl", gaps, jsonl=True)
    _write(output / "exclusions.jsonl", exclusions, jsonl=True)
    _write(output / "history_report.json", history_report)
    _write(output / "duplicate_groups.json", duplicate_report)
    _write(output / "readiness.json", report)
    if selected is not None:
        split_metadata = {}
        for role in SPLITS:
            path = output / f"{role}.jsonl"
            _write(path, selected[role], jsonl=True)
            split_metadata[role] = {
                "count": len(selected[role]),
                "usage": ("sealed_unused" if role != "test" else "transfer_test")
                         if dataset_role == "transfer_only" else role,
                "training_allowed": dataset_role == "training_source" and role == "train",
                "sealed_unused": dataset_role == "transfer_only" and role != "test",
                "question_type_counts": dict(sorted(Counter(row["question_type"] for row in selected[role]).items())),
                "historical_analysis_status_counts": dict(sorted(Counter(
                    row["historical_analysis_status"] for row in selected[role]).items())),
                "historical_exposure_status_counts": dict(sorted(Counter(
                    row["historical_exposure_status"] for row in selected[role]).items())),
                "group_count": len({row["question_group_id"] for row in selected[role]}),
                "items": [{"id": row["id"], "question_type": row["question_type"]} for row in selected[role]],
                "file": {**_file_record(path), "path": path.name},
            }
        artifacts = {str(item["role"]): {"path": item["path"], "sha256": item["sha256"],
                                          "size_bytes": item["size_bytes"]}
                     for item in manifest.get("source_artifacts", [])}
        split_manifest = {
            "schema_version": PREPARATION_SPLIT_SCHEMA_VERSION, "preparation_version": PREPARATION_VERSION,
            "purpose": "full_pool_trajectory_representation_ablation", "paper_parity": False,
            "history_policy": history_policy, "dataset_role": dataset_role,
            "dataset": {"subset": dataset, "repo_id": profile.repo_id, "revision": profile.revision,
                        "scope_id": rows[0]["scope_id"] if rows else profile.scope_id("dev"),
                        "question_count": len(rows), "source_question_count": len(canonical_rows),
                        "chunk_count": manifest.get("record_counts", {}).get("chunks"),
                        "answer_mode": profile.answer_mode.value,
                        "source_files": artifacts,
                        "question_type_counts": dict(sorted(Counter(row["question_type"] for row in rows).items())),
                        "metric_contract": {"reported": [metric.value for metric in profile.reported_metrics],
                                            "hard": profile.skillopt_hard_metric.value,
                                            "soft": profile.skillopt_soft_metric.value},
                        "substrate": {"path": substrate.as_posix(),
                                      "manifest_sha256": sha256_file(substrate / "manifest.json")}},
            "selection": selection, "splits": split_metadata,
            "readiness": {"all_four_arms_ready": report["all_four_arms_ready"],
                          "inference_ready": inference_ready,
                          "reference_progress_ready": reference_progress_ready,
                          "training_ready": training_ready,
                          "mapping_issue_count": len(gaps)},
            "history_report": {**_file_record(output / "history_report.json"),
                               "path": "history_report.json"},
            "archive": {**_file_record(output / "hidden_reference_archive.jsonl"),
                        "path": "hidden_reference_archive.jsonl"},
        }
        _write(output / "split_manifest.json", split_manifest)
    return report


def prepare_multidataset(
    spec: Mapping[str, Any] | str | Path, output: str | Path,
    *, reference_builder: ReferenceBuilder | None = None,
) -> dict[str, Any]:
    """Prepare each dataset independently; blocked datasets do not stop others."""
    base = Path.cwd()
    if not isinstance(spec, Mapping):
        path = Path(spec).resolve()
        base = path.parent
        spec = _load(path)
    if not isinstance(spec, Mapping) or not isinstance(spec.get("datasets"), list):
        raise PreparationError("invalid_spec", "Specification requires a datasets list")
    seed = spec.get("seed", 42)
    if type(seed) is not int:
        raise PreparationError("invalid_seed", "Seed must be an integer")
    history_policy = str(spec.get("history_policy", "locked"))
    if history_policy not in HISTORY_POLICIES:
        raise PreparationError("invalid_history_policy", f"Unsupported history policy: {history_policy}")
    names = [str(item.get("dataset", "")) for item in spec["datasets"]]
    if len(names) != len(set(names)) or not all(names):
        raise PreparationError("duplicate_spec_dataset", "Datasets must be unique and named")
    destination = Path(output).resolve()
    reports = {}
    for item in spec["datasets"]:
        item = {"history_policy": history_policy, **item}
        dataset = str(item["dataset"])
        get_dataset_profile(dataset)  # Reject arbitrary output path components.
        try:
            reports[dataset] = _prepare_dataset(item, destination / dataset, base=base,
                                                seed=seed, reference_builder=reference_builder)
        except (PreparationError, ValueError, KeyError, OSError) as exc:
            if isinstance(exc, PreparationError) and exc.code == "output_exists":
                raise
            report = {"version": PREPARATION_VERSION, "dataset": dataset,
                      "split_status": "blocked", "all_four_arms_ready": False,
                      "history_policy": item.get("history_policy", "locked"),
                      "dataset_role": item.get("dataset_role", "training_source"),
                      "inference_ready": False, "reference_progress_ready": False,
                      "training_ready": False,
                      "error": {"code": getattr(exc, "code", type(exc).__name__),
                                "message": str(exc), "details": getattr(exc, "details", None)},
                      "model_calls": 0, "indexes_rebuilt": False}
            _write(destination / dataset / "readiness.json", report)
            reports[dataset] = report
    training_reports = [row for row in reports.values() if row.get("dataset_role") == "training_source"]
    result = {"version": PREPARATION_VERSION, "seed": seed, "history_policy": history_policy,
              "datasets": reports,
              "model_calls": 0, "indexes_rebuilt": False,
              "all_inference_ready": bool(reports) and all(row["inference_ready"] for row in reports.values()),
              "all_training_sources_ready": bool(training_reports) and all(row["training_ready"] for row in training_reports),
              "all_four_arms_ready": bool(reports) and all(row["all_four_arms_ready"] for row in reports.values())}
    _write(destination / "preparation_report.json", result)
    return result


def validate_prepared_split(
    split_dir: str | Path, substrate: str | Path,
    require_reference_progress: bool = False,
    *, for_training: bool = False,
) -> dict[str, Any]:
    """Reject stale/blocked outputs before a training run can consume labels."""
    root, substrate_path = Path(split_dir).resolve(), Path(substrate).resolve()
    manifest = _load(root / "split_manifest.json")
    readiness = _load(root / "readiness.json")
    schema_version = str(manifest.get("schema_version", ""))
    expected_version = SUPPORTED_PREPARATION_VERSIONS.get(schema_version)
    if expected_version is None or manifest.get("preparation_version") != expected_version:
        raise PreparationError("invalid_preparation_version", "Unexpected prepared split version")
    history_policy = str(manifest.get("history_policy", "locked"))
    dataset_role = str(manifest.get("dataset_role", "training_source"))
    if history_policy not in HISTORY_POLICIES or dataset_role not in DATASET_ROLES:
        raise PreparationError("invalid_preparation_policy", "Unknown history policy or dataset role")
    if schema_version == "3.0" and (history_policy != "locked" or dataset_role != "training_source"):
        raise PreparationError("invalid_preparation_policy", "Legacy schema only supports historical locks and training sources")
    if readiness.get("version") != expected_version:
        raise PreparationError("invalid_preparation_version", "Readiness report version differs from split manifest")
    if readiness.get("split_status") != "ready" or readiness.get("split_error"):
        raise PreparationError("blocked_prepared_split", "Historical split constraints are unresolved")
    history = _load(root / "history_report.json")
    if schema_version == PREPARATION_SPLIT_SCHEMA_VERSION:
        if (readiness.get("history_policy") != history_policy
                or history.get("history_policy") != history_policy
                or manifest.get("selection", {}).get("history_policy") != history_policy
                or readiness.get("dataset_role") != dataset_role):
            raise PreparationError("preparation_policy_mismatch", "History policy or dataset role differs across preparation artifacts")
        history_record = manifest.get("history_report", {})
        history_path = root / "history_report.json"
        if (history_record.get("path") != history_path.name
                or history_record.get("sha256") != sha256_file(history_path)
                or history_record.get("size_bytes") != history_path.stat().st_size):
            raise PreparationError("history_report_hash_mismatch", "Historical audit report changed")
        if manifest.get("selection", {}).get("preserves_historical_roles") != (history_policy == "locked"):
            raise PreparationError("preparation_policy_mismatch", "History preservation label does not match the selected policy")
    if history_policy == "locked" and (history.get("conflicts") or history.get("unresolved_items")):
        raise PreparationError("invalid_history", "History has unresolved items or conflicting assignments")
    reference_progress_ready = readiness.get("reference_progress_ready", readiness.get("all_four_arms_ready", False))
    if require_reference_progress and not reference_progress_ready:
        raise PreparationError("reference_progress_unavailable", "Not all retained questions have usable reference progress")
    if for_training and dataset_role != "training_source":
        raise PreparationError("dataset_sealed_for_transfer", "Transfer-only train/validation partitions are sealed and cannot be used for training")
    metadata = manifest.get("dataset", {})
    dataset = str(metadata.get("subset") or "")
    get_dataset_profile(dataset)
    pinned = metadata.get("substrate", {})
    if Path(str(pinned.get("path", ""))).resolve() != substrate_path:
        raise PreparationError("substrate_path_mismatch", "Prepared split is bound to another substrate")
    if sha256_file(substrate_path / "manifest.json") != pinned.get("manifest_sha256"):
        raise PreparationError("substrate_hash_mismatch", "Substrate manifest changed")
    expected_inputs = readiness.get("input_files")
    if not isinstance(expected_inputs, dict) or not expected_inputs:
        raise PreparationError("missing_input_hashes", "Prepared split has no input/index hashes")
    for filename, record in expected_inputs.items():
        if _file_record(Path(filename)) != record:
            raise PreparationError("input_hash_mismatch", f"Pinned input/index changed: {filename}")
    expected_substrate_files = {name for name in expected_inputs
                                if Path(name).is_relative_to(substrate_path)}
    actual_substrate_files = {path.resolve().as_posix() for path in substrate_path.rglob("*") if path.is_file()}
    if expected_substrate_files != actual_substrate_files:
        raise PreparationError("substrate_file_set_changed", "Substrate files were added or removed")
    all_ids, all_groups = set(), set()
    split_report = {}
    for role in SPLITS:
        record = manifest.get("splits", {}).get(role)
        if not isinstance(record, dict):
            raise PreparationError("missing_split", f"Missing split metadata: {role}")
        if schema_version == PREPARATION_SPLIT_SCHEMA_VERSION:
            expected_usage = ("transfer_test" if role == "test" else "sealed_unused") if dataset_role == "transfer_only" else role
            if (record.get("usage") != expected_usage
                    or record.get("sealed_unused") != (dataset_role == "transfer_only" and role != "test")
                    or record.get("training_allowed") != (dataset_role == "training_source" and role == "train")):
                raise PreparationError("invalid_split_usage", "Split usage does not match its sealed transfer/training role")
        path = root / f"{role}.jsonl"
        file_record = record.get("file", {})
        if (file_record.get("path") != path.name or
                file_record.get("sha256") != sha256_file(path) or
                file_record.get("size_bytes") != path.stat().st_size):
            raise PreparationError("split_hash_mismatch", f"Split file changed: {role}")
        rows = _rows(path)
        ids = [row["id"] for row in rows]
        groups = {row["question_group_id"] for row in rows}
        if len(rows) != record.get("count") or ids != [row["id"] for row in record.get("items", [])]:
            raise PreparationError("split_order_mismatch", f"Split count or ordered IDs changed: {role}")
        if len(ids) != len(set(ids)) or all_ids.intersection(ids) or all_groups.intersection(groups):
            raise PreparationError("split_overlap", "Question IDs or identical-question groups cross splits")
        for row in rows:
            if row.get("question_group_id") != _digest(f"{dataset}\0{normalized_text(row['question'])}"):
                raise PreparationError("invalid_question_group", "Question group hash does not match question text")
            if row.get("scope_id") != metadata.get("scope_id") or row.get("source") != dataset:
                raise PreparationError("split_scope_mismatch", "Split contains a different dataset/scope")
            if require_reference_progress and row.get("reference_evidence", {}).get("status") != "ready":
                raise PreparationError("reference_progress_unavailable", f"Reference unavailable: {row['id']}")
        all_ids.update(ids)
        all_groups.update(groups)
        split_report[role] = {"count": len(rows), "group_count": len(groups),
                              "sha256": file_record["sha256"]}
    if len(all_ids) != metadata.get("question_count"):
        raise PreparationError("pool_count_mismatch", "Splits do not contain the entire eligible pool")
    return {"valid": True, "dataset": dataset, "scope_id": metadata["scope_id"],
            "schema_version": schema_version, "history_policy": history_policy,
            "dataset_role": dataset_role, "inference_ready": True,
            "reference_progress_ready": bool(reference_progress_ready),
            "training_ready": dataset_role == "training_source" and bool(reference_progress_ready),
            "all_four_arms_ready": readiness["all_four_arms_ready"],
            "total_question_count": len(all_ids), "splits": split_report,
            "split_manifest_sha256": sha256_file(root / "split_manifest.json")}
