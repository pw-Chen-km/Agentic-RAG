"""Phase 1: frozen rollout batch -> separate analysts -> ONE merge -> gate.

The standalone runner performs no Meta or test evaluation. The full workflow can
enable candidate train replay to supply the separate Meta stage.
All completed operations are content-checked checkpoints, including no_change.
"""

import copy
import fcntl
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .candidate_replay import collect_train_branches
from .checkpoint import load_checkpoint, save_checkpoint
from .compact_rendering import DISPLAY_VERSION, PLAIN_DISPLAY_VERSION
from .context_rendering import (
    CONTEXT_RENDERER_VERSION,
    render_phase1_merge_context,
    render_phase1_reflection_context,
)
from .diagnostics import (
    FORMAT_REPAIR_VERSION,
    STATE_ACTION_PROTOCOL,
    VERSION,
    citation_advisories,
    diagnostic_tools,
    parse_diagnoses,
    normalize_redundant_case_id,
    support_ids,
)
from .optimizer_tools import parse_tool_call, tool_schemas
from .phase1_gate import GATE_VERSION, decide
from .skill_rules import RuleStore, compile_initial_skill, skill_sha256
from .support import digest
from .tokens import count
from .trajectory import build_view, render_view, render_view_with_audit
from .action_space import DEFAULT_ACTION_SPACE, action_space_contract, action_usage
from .retrieval_diagnostics import build_retrieval_diagnostics


@dataclass(frozen=True)
class Phase1Config:
    rollout_batch_size: int = 40
    reflection_minibatch_size: int = 5
    max_reflection_edits: int = 2
    max_batch_edits: int = 3
    max_rule_tokens: int = 300
    max_skill_tokens: int = 1500
    max_input_tokens: int = 16000
    max_merge_input_tokens: int | None = None
    max_retries: int = 3
    episode_workers: int = 1
    reflection_workers: int = 1
    min_support_cases: int = 2
    analysis_protocol: str = VERSION
    diagnostic_format_repair: str | None = None
    max_token_cost_increase: float = 0.05
    max_call_cost_increase: float = 0.05
    min_validation_pair_coverage: float = 1.0

    def __post_init__(self):
        if (
            self.rollout_batch_size != 40
            or not 1 <= self.reflection_minibatch_size <= 5
        ):
            raise ValueError("Phase 1 requires rollout 40 and reflection <=5")
        if self.max_reflection_edits != 2 or self.max_batch_edits != 3:
            raise ValueError("Phase 1 edit limits must be 2/reflection, 3/batch")
        if not 1 <= self.episode_workers <= 8 or not 1 <= self.reflection_workers <= 8:
            raise ValueError("workers must be between 1 and 8")
        if type(self.min_support_cases) is not int or self.min_support_cases < 1:
            raise ValueError("min_support_cases must be a positive integer")
        if self.max_merge_input_tokens is not None and (
            type(self.max_merge_input_tokens) is not int or self.max_merge_input_tokens < self.max_input_tokens
        ):
            raise ValueError("merge input budget must be an integer at least the reflection budget")
        if not isinstance(self.analysis_protocol, str) or not self.analysis_protocol:
            raise ValueError("analysis_protocol must be a nonempty string")
        if self.diagnostic_format_repair is not None and (
            self.diagnostic_format_repair != FORMAT_REPAIR_VERSION
            or self.analysis_protocol != STATE_ACTION_PROTOCOL
        ):
            raise ValueError("diagnostic format repair requires the v6 state-action protocol and a known repair version")
        for limit in (self.max_token_cost_increase, self.max_call_cost_increase):
            if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not 0 <= limit <= 1:
                raise ValueError("cost increase limits must be finite fractions in [0,1]")
        coverage = self.min_validation_pair_coverage
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not 0 < coverage <= 1:
            raise ValueError("validation pair coverage must be a finite fraction in (0,1]")


def public_skill(skill):
    return {
        "schema_version": skill["schema_version"],
        "retrieval_policy": skill["retrieval_policy"],
    }


class Phase1:
    def __init__(
        self,
        output,
        skill,
        client,
        rollout,
        evaluator,
        prompts,
        tool_guidance,
        config=None,
        support_map=None,
        contract=None,
        replay_for_meta=False,
        action_space=None,
    ):
        self.root = Path(output)
        self.initial = copy.deepcopy(skill)
        self.client, self.rollout, self.evaluator = client, rollout, evaluator
        self.prompts, self.tool_guidance = prompts, tool_guidance
        self.cfg = config or Phase1Config()
        self.replay_for_meta = replay_for_meta
        self.support_map = support_map or {"questions": {}}
        self.action_space = tuple(action_space or DEFAULT_ACTION_SPACE)
        self.action_objective = action_space_contract(self.action_space)
        self.contract = {
            "phase": "phase1-diagnosis-merge-v3",
            "analysis_protocol": self.cfg.analysis_protocol,
            "gate_version": GATE_VERSION,
            "optimizer_renderer_version": (PLAIN_DISPLAY_VERSION if self.cfg.analysis_protocol == STATE_ACTION_PROTOCOL else DISPLAY_VERSION),
            "optimizer_context_renderer_version": CONTEXT_RENDERER_VERSION,
            "diagnostic_schema_hash": digest(diagnostic_tools(self.cfg.analysis_protocol)),
            "initial_hash": skill_sha256(skill),
            "config": asdict(self.cfg),
            "prompts_hash": digest(prompts),
            "tool_guidance_hash": digest(tool_guidance),
            "support_map_hash": digest(self.support_map),
            "candidate_train_replay": replay_for_meta,
            **(contract or {}),
            "action_space": self.action_objective,
        }

    def store(self, skill, limit):
        return RuleStore(
            skill,
            max_edits=limit,
            max_tokens=self.cfg.max_rule_tokens,
            max_skill_tokens=self.cfg.max_skill_tokens,
        )

    def validation_gate(self, parent, candidate):
        """One configured gate shared by Phase 1 and Meta."""
        return decide(
            parent, candidate,
            max_token_cost_increase=self.cfg.max_token_cost_increase,
            max_call_cost_increase=self.cfg.max_call_cost_increase,
            min_validation_pair_coverage=self.cfg.min_validation_pair_coverage,
        )

    def replay_contract_hash(self, batch):
        """Explicit contract owner of a train replay; default is this run."""
        return digest(self.contract)

    def map_ordered(self, operation, items, workers):
        """Only independent work runs concurrently; persisted/result order stays fixed."""
        if workers == 1:
            return [operation(item) for item in items]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(operation, items))

    def episodes(self, questions, skill, split, version):
        return self.map_ordered(
            lambda q: self.episode(q, skill, split, version),
            questions,
            self.cfg.episode_workers,
        )

    def cached(self, path, inputs, compute):
        signature = digest(inputs)
        saved = load_checkpoint(path)
        if saved:
            if saved.get("input_hash") != signature:
                raise ValueError(f"checkpoint inputs changed: {path}")
            return saved["value"]
        value = compute()
        save_checkpoint(path, {"input_hash": signature, "value": value})
        return value

    def episode(self, question, skill, split, version):
        qid = str(question["id"])
        key = digest({"id": qid})[:20]
        directory = self.root / "episodes" / split / skill_sha256(skill) / key
        directory.mkdir(parents=True, exist_ok=True)
        runtime_question = {"id": qid, "question": question["question"]}
        raw = self.cached(
            directory / "raw.json",
            runtime_question,
            lambda: self.rollout(copy.deepcopy(runtime_question), copy.deepcopy(skill)),
        )
        retrieval_diagnostics = build_retrieval_diagnostics(raw)
        save_checkpoint(
            directory / "retrieval_diagnostics.json",
            {
                "input_hash": digest({"raw": digest(raw)}),
                "value": retrieval_diagnostics,
            },
        )
        if raw.get("question_id") != qid or raw.get("question") != question["question"]:
            raise ValueError("runtime returned a different question")

        def evaluate():
            if raw.get("execution_errors") or raw.get("error"):
                return {"correct": None, "status": "runtime_error"}
            result = self.evaluator(question, raw)
            if result.get("status") != "ok" or type(result.get("correct")) is not bool:
                return {**result, "correct": None, "status": "unavailable"}
            return result

        evaluation = self.cached(
            directory / "evaluation.json",
            {"question": question, "raw": digest(raw)},
            evaluate,
        )
        entry = self.support_map["questions"].get(qid)
        if entry and entry["question"] != question["question"]:
            raise ValueError("support mapping question mismatch")
        view, audit = build_view(
            raw, evaluation, entry if split == "train" else None, version
        )
        save_checkpoint(directory / "view.json", view)
        save_checkpoint(
            directory / "lineage.json",
            {"skill_hash": skill_sha256(skill), "split": split, **audit},
        )
        rendered, render_audit = render_view_with_audit(
            view, plain_language=self.cfg.analysis_protocol == STATE_ACTION_PROTOCOL
        )
        save_checkpoint(directory / "optimizer_render_lineage.json", render_audit)
        (directory / "optimizer_view.txt").write_text(rendered, encoding="utf-8")
        usage = action_usage(raw)
        save_checkpoint(directory / "action_usage.json", usage)
        return {
            "question_id": qid,
            "raw": raw,
            "evaluation": evaluation,
            "view": view,
            "action_usage": usage,
        }

    def messages(self, stage, skill, payload, decision_refs=None):
        # Hashes/provenance fields deliberately excluded; runtime text shown once.
        messages = [
            {
                "role": "system",
                "content": self.prompts["common"] + "\n\n" + self.prompts[stage],
            },
            {
                "role": "user",
                "content": "Current Skill:\n"
                + json.dumps(public_skill(skill), ensure_ascii=False)
                + "\nFixed tool guidance:\n"
                + self.tool_guidance
                + "\n\nAction-space and Phase 1 optimization objective:\n"
                + json.dumps(self.action_objective, ensure_ascii=False, sort_keys=True)
                + "\n\n"
                + payload,
            },
        ]
        if decision_refs is not None and self.cfg.analysis_protocol == "diagnosis-to-edits-v4-single-case":
            messages[-1]["content"] += (
                "\n\nAllowed exact Decision references by case (copy these strings):\n"
                + json.dumps(decision_refs, ensure_ascii=False)
                + "\nUse one observation per case_id within each diagnosis; combine that case's decisions in its decisions list."
            )
        return messages

    def propose(
        self,
        stage,
        skill,
        payload,
        case_ids,
        path,
        limit,
        *,
        diagnostic=False,
        decision_refs=None,
        documented_support_ids=None,
        classification_advisories=None,
    ):
        messages = self.messages(stage, skill, payload, decision_refs=decision_refs if diagnostic else None)
        rule_ids = [r["rule_id"] for r in skill["retrieval_policy"]]
        eligible_ids = set(
            case_ids if documented_support_ids is None else documented_support_ids
        )
        schemas = (
            diagnostic_tools(self.cfg.analysis_protocol)
            if diagnostic
            else tool_schemas(rule_ids, sorted(eligible_ids), self.cfg.min_support_cases,
                              plain_language=self.cfg.analysis_protocol == STATE_ACTION_PROTOCOL)
        )
        inputs = {
            "messages": messages,
            "tools": schemas,
            "limit": limit,
            "case_ids": sorted(case_ids),
            "diagnostic": diagnostic,
            "decision_refs": decision_refs,
            "documented_support_ids": sorted(eligible_ids),
            "classification_advisories": classification_advisories or [],
        }
        input_tokens = count({"messages": messages, "tools": schemas})
        input_limit = (self.cfg.max_merge_input_tokens or self.cfg.max_input_tokens) if stage in {"merge", "meta_merge"} else self.cfg.max_input_tokens

        def invoke():
            if input_tokens > input_limit:
                return {
                    "status": "input_too_long",
                    "edits": [],
                    "diagnoses": [],
                    "budget_input_tokens": input_tokens,
                    "budget_input_limit": input_limit,
                    "request_not_sent": {"messages": messages, "tools": schemas},
                }
            audit = []
            for retry in range(1, self.cfg.max_retries + 1):
                attempt_path = path.with_name(path.stem + f".attempt{retry}.json")

                def request():
                    request_record = {
                        "messages": copy.deepcopy(messages),
                        "tools": schemas,
                    }
                    try:
                        return {
                            "request": request_record,
                            "response": self.client.chat(
                                messages=messages, tools=schemas, temperature=0.0
                            ),
                        }
                    except Exception as exc:  # noqa: BLE001 -- persist provider failures; do not adopt edits
                        return {"request": request_record, "error": str(exc)}

                response = self.cached(attempt_path, inputs, request)
                attempt = {"retry": retry, **response}
                try:
                    if "error" in response:
                        raise ValueError(response["error"])
                    raw_response = response["response"].get("raw_response") or {}
                    choices = raw_response.get("choices") or []
                    finish_reason = (
                        choices[0].get("finish_reason")
                        if choices else raw_response.get("done_reason")
                    )
                    if finish_reason in {"length", "max_tokens"}:
                        raise ValueError(
                            "Optimizer output reached its generation limit; do not apply "
                            "a potentially incomplete diagnostic report or edit set. "
                            "Return a shorter, complete response."
                        )
                    calls = (
                        response["response"].get("message", {}).get("tool_calls") or []
                    )
                    if not 1 <= len(calls) <= limit:
                        raise ValueError(f"expected 1..{limit} native calls")
                    if diagnostic:
                        diagnostic_call = calls[0]
                        if self.cfg.diagnostic_format_repair:
                            diagnostic_call, events = normalize_redundant_case_id(diagnostic_call, case_ids)
                            attempt["diagnostic_normalizations"] = events
                            if events:
                                attempt["normalized_tool_call"] = diagnostic_call
                        report = parse_diagnoses(
                            diagnostic_call, case_ids, rule_ids, decision_refs,
                            protocol=self.cfg.analysis_protocol,
                        )
                        attempt["parsed_diagnoses"] = report
                        audit.append(attempt)
                        return {
                            "status": "ok",
                            "edits": [],
                            **report,
                            "audit": audit,
                            "budget_input_tokens": input_tokens,
                        }
                    edits = [parse_tool_call(c) for c in calls]
                    for edit in edits:
                        if edit["operation"] == "no_change":
                            continue
                        support = set(edit["supporting_case_ids"])
                        unknown = support - set(case_ids)
                        if unknown:
                            raise ValueError(
                                "supporting_case_ids contains IDs outside the supplied "
                                f"train cases: {sorted(unknown)}; eligible SUPPORT "
                                f"case IDs: {sorted(eligible_ids)}"
                            )
                        if len(support) < self.cfg.min_support_cases:
                            minimum = "two" if self.cfg.min_support_cases == 2 else str(self.cfg.min_support_cases)
                            raise ValueError(
                                f"supporting_case_ids requires at least {minimum} DIFFERENT "
                                f"train case IDs; received {len(support)}; eligible "
                                f"SUPPORT case IDs: {sorted(eligible_ids)}"
                            )
                        unsupported = support - eligible_ids
                        if unsupported:
                            if self.cfg.analysis_protocol == STATE_ACTION_PROTOCOL:
                                raise ValueError(
                                    "supporting_case_ids includes questions without observations supporting the finding: "
                                    f"{sorted(unsupported)}; eligible question IDs: {sorted(eligible_ids)}"
                                )
                            raise ValueError(
                                "supporting_case_ids includes cases without a "
                                "documented SUPPORT observation (neutral, "
                                f"counterexample, or undiagnosed): {sorted(unsupported)}; "
                                f"eligible SUPPORT case IDs: {sorted(eligible_ids)}"
                            )
                        rule = edit.get("rule")
                        if (rule and self.cfg.analysis_protocol in {"diagnosis-to-edits-v4-single-case", STATE_ACTION_PROTOCOL}
                                and re.search(r"\bkeyword_search\b", json.dumps(rule))):
                            raise ValueError("rule introduces unavailable action keyword_search")
                        if rule and re.search(
                            r"\b(?:EXPAND|FINISH|E\d+|S\d+|C\d+)\b", json.dumps(rule)
                        ):
                            raise ValueError(
                                "rule uses old interface actions/references"
                            )
                    self.store(skill, limit).apply(edits)
                    cited = {
                        qid for edit in edits for qid in edit.get("supporting_case_ids", [])
                    }
                    advisories = [
                        note for note in classification_advisories or []
                        if note["case_id"] in cited
                    ]
                    attempt["parsed_edits"] = edits
                    attempt["citation_advisories"] = advisories
                    audit.append(attempt)
                    return {
                        "status": "ok",
                        "edits": edits,
                        "audit": audit,
                        "budget_input_tokens": input_tokens,
                        "citation_advisories": advisories,
                    }
                except Exception as exc:  # noqa: BLE001 -- all invalid native replies become auditable no_change
                    attempt["validation_error"] = str(exc)
                    audit.append(attempt)
                    # Corrective feedback is bounded and contains no raw Skill rewrite.
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Invalid diagnostic report: "
                                if diagnostic
                                else "Invalid edit response: "
                            )
                            + str(exc)
                            + (
                                ". Call report_diagnoses once. Explain each supplied case using state, action and result. "
                                "Use analysis, possible_skill_improvement and limitations for the explanation. "
                                "Do not add rule IDs, classification labels or decision-reference fields. "
                                "The diagnosis object has only observations, analysis, possible_skill_improvement, and limitations. "
                                "Put case_id only in each observations item; never put it beside analysis. "
                                "Follow the field path and unexpected/missing field names in the error above. "
                                "An empty diagnoses list is valid."
                                if diagnostic and self.cfg.analysis_protocol == STATE_ACTION_PROTOCOL else
                                ". Call report_diagnoses once; an empty diagnoses list is valid. "
                                "Use one observation per case_id in each diagnosis; combine that case's decisions in one observation. "
                                "Copy only these exact Decision references: "
                                + json.dumps(decision_refs or {}, ensure_ascii=False)
                                if diagnostic
                                else ". Use the allowed native tools; no_change is valid."
                            ),
                        }
                    )
                    if (
                        count({"messages": messages, "tools": schemas})
                        > input_limit
                    ):
                        break
            return {
                "status": "diagnosis_failed_after_retry"
                if diagnostic
                else "no_change_after_retry",
                "edits": [],
                "diagnoses": [],
                "audit": audit,
                "budget_input_tokens": input_tokens,
            }

        # Save actual optimizer request separately, including prompt for reproducibility.
        return self.cached(path, inputs, invoke)

    def diagnose(self, stage, skill, payload, case_ids, path, decision_refs):
        # Exactly one reporting call, up to two diagnoses, zero executable edits.
        return self.propose(
            stage,
            skill,
            payload,
            case_ids,
            path,
            1,
            diagnostic=True,
            decision_refs=decision_refs,
        )

    def reflect_group(self, kind, skill, episodes, directory, name):
        plain = self.cfg.analysis_protocol == STATE_ACTION_PROTOCOL
        payload = render_phase1_reflection_context(
            skill,
            episodes,
            kind,
            action_space=self.action_space,
            plain_language=plain,
        )
        ids = [e["question_id"] for e in episodes]
        path = directory / (name + ".json")
        refs = {e["question_id"]: [str(d["decision"]) for d in e["view"]["decisions"]]
                for e in episodes}
        messages = self.messages(kind, skill, payload, decision_refs=refs)
        tools = diagnostic_tools(self.cfg.analysis_protocol)
        if (
            count({"messages": messages, "tools": tools}) > self.cfg.max_input_tokens
            and len(episodes) > 1
        ):
            middle = len(episodes) // 2
            save_checkpoint(
                path.with_suffix(".split.json"),
                {
                    "kind": kind,
                    "ids": ids,
                    "reason": "input_token_budget",
                    "split_at": middle,
                },
            )
            return self.reflect_group(
                kind, skill, episodes[:middle], directory, name + "a"
            ) + self.reflect_group(
                kind, skill, episodes[middle:], directory, name + "b"
            )
        proposal = self.diagnose(
            kind,
            skill,
            payload,
            ids,
            path,
            refs,
        )
        return [
            {"proposal_id": name, "analysis_kind": kind, "case_ids": ids, **proposal}
        ]

    def merge(
        self,
        skill,
        proposals,
        directory,
        stage="merge",
        edit_limit=None,
        *,
        context_payload=None,
    ):
        # All diagnoses, including neutral/counterexamples, stay available to merge.
        # Do not silently select an early proposal if the combined input is too long.
        edit_limit = self.cfg.max_batch_edits if edit_limit is None else edit_limit
        if self.cfg.analysis_protocol == STATE_ACTION_PROTOCOL:
            for report in proposals:
                if report["status"] == "ok":
                    parse_diagnoses({"function": {"name": "report_diagnoses", "arguments": {
                        "diagnoses": report.get("diagnoses", []), "reason": report.get("reason", "")
                    }}}, report["case_ids"], [], protocol=STATE_ACTION_PROTOCOL)
        compact = [
            {
                "proposal_id": p["proposal_id"],
                "analysis_kind": p["analysis_kind"],
                "case_ids": p["case_ids"],
                "status": p["status"],
                "diagnoses": p.get("diagnoses", []),
                "reason": p.get("reason", ""),
            }
            for p in proposals
        ]
        if self.cfg.analysis_protocol == STATE_ACTION_PROTOCOL:
            # The report's summary repeats the diagnoses. Keep it in audit, and
            # only send reasons for empty/failed reports where they explain a gap.
            for report in compact:
                if report["status"] == "ok" and report["diagnoses"]:
                    report.pop("reason")
        # Only provenance/observation checks are hard gates. Model-assigned
        # layer/basis labels are advisory in BOTH Phase 1 and Meta.
        train_ids = {qid for p in compact for qid in p["case_ids"]}
        ids = support_ids(compact)
        if not ids <= train_ids:
            raise ValueError("diagnostic SUPPORT case is outside supplied train cases")
        advisories = citation_advisories(compact)
        if len(ids) < self.cfg.min_support_cases:
            return self.cached(
                directory / "merge.json",
                {"reports": compact, "skill": skill, "analysis_protocol": self.cfg.analysis_protocol,
                 "min_support_cases": self.cfg.min_support_cases},
                lambda: {
                    "status": ("no_documented_support" if self.cfg.min_support_cases == 1
                               else "insufficient_cross_question_diagnoses"),
                    "edits": [],
                    "supporting_case_count": len(ids),
                    "citation_advisories": advisories,
                },
            )
        # HRO supplies a deterministic Stage B projection here.  The
        # canonical JSON remains in its checkpoint; only the compact labelled
        # rendering is sent to the model.  Ordinary Phase 1 and non-HRO Meta
        # merges use the same sectioned renderer instead of hand-built JSON.
        payload = context_payload
        if payload is None:
            payload = render_phase1_merge_context(
                skill,
                compact,
                sorted(ids),
                action_space=self.action_space,
                state_action=self.cfg.analysis_protocol == STATE_ACTION_PROTOCOL,
            )
        return self.propose(
            stage,
            skill,
            payload,
            train_ids,
            directory / "merge.json",
            edit_limit,
            documented_support_ids=ids,
            classification_advisories=advisories,
        )

    def run(self, train, validation):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".phase1.lock").open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "another Phase 1 runner owns this output directory"
                ) from exc
            return self._run(train, validation)

    def prepare_candidate(self, number, questions, parent, directory):
        """One frozen batch's diagnosis and merge; separate from gate/replay."""
        episodes = self.episodes(questions, parent, "train", f"batch {number} parent")
        groups = {"success": [], "failure": [], "unavailable": []}
        for e in episodes:
            correct = e["evaluation"].get("correct")
            groups["success" if correct is True else "failure" if correct is False else "unavailable"].append(e)
        save_checkpoint(directory / "groups.json", {k: [e["question_id"] for e in v] for k, v in groups.items()})
        jobs = [(kind, groups[kind][i:i + self.cfg.reflection_minibatch_size], f"{kind}-{i:03d}")
                for kind in ("success", "failure")
                for i in range(0, len(groups[kind]), self.cfg.reflection_minibatch_size)]
        nested = self.map_ordered(
            lambda job: self.reflect_group(job[0], parent, job[1], directory / "reflections", job[2]),
            jobs, self.cfg.reflection_workers)
        proposals = [p for group in nested for p in group]
        merged = self.merge(parent, proposals, directory)
        candidate, change = self.store(parent, self.cfg.max_batch_edits).apply_with_receipt(merged["edits"])
        return candidate, change, len(proposals), merged["status"]

    def _run(self, train, validation):
        ids = [[str(q["id"]) for q in rows] for rows in (train, validation)]
        if (
            not all(ids)
            or any(len(x) != len(set(x)) for x in ids)
            or set(ids[0]) & set(ids[1])
        ):
            raise ValueError("train/validation must be nonempty, unique and disjoint")
        self.contract.update(
            train_hash=digest(train), validation_hash=digest(validation)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        previous = load_checkpoint(self.root / "contract.json")
        if previous and previous != self.contract:
            raise ValueError("contract changed: use a NEW output directory")
        if not previous and any(p.name != ".phase1.lock" for p in self.root.iterdir()):
            raise ValueError("nonempty output without Phase 1 contract")
        save_checkpoint(self.root / "contract.json", self.contract)
        active = copy.deepcopy(self.initial)
        receipts = []
        for offset in range(0, len(train), self.cfg.rollout_batch_size):
            number = offset // self.cfg.rollout_batch_size + 1
            directory = self.root / "batches" / f"{number:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            questions = train[offset : offset + self.cfg.rollout_batch_size]
            key = {"parent": skill_sha256(active), "questions": digest(questions)}
            done = load_checkpoint(directory / "completed.json")
            if done:
                if done["input_hash"] != digest(key):
                    raise ValueError("completed batch lineage mismatch")
                active = done["value"]["active_skill"]
                receipts.append(done["value"])
                continue
            parent = copy.deepcopy(active)
            save_checkpoint(directory / "parent_skill.json", parent)
            candidate, change, reflection_count, optimizer_status = self.prepare_candidate(
                number, questions, parent, directory)
            save_checkpoint(directory / "candidate_skill.json", candidate)
            if skill_sha256(parent) == skill_sha256(candidate):
                gate = {
                    "accepted": False,
                    "reason": "no_change",
                    "optimizer_status": optimizer_status,
                }
            else:
                parent_val = self.episodes(
                    validation, parent, "validation", "validation parent"
                )
                candidate_val = self.episodes(
                    validation, candidate, "validation", "validation candidate"
                )
                gate = self.cached(
                    directory / "validation.json",
                    {
                        "parent": skill_sha256(parent),
                        "candidate": skill_sha256(candidate),
                    },
                    lambda a=parent_val, b=candidate_val: self.validation_gate(a, b),
                )
            if gate["accepted"]:
                active = candidate
            replay = None
            if self.replay_for_meta:
                replay = collect_train_branches(
                    self, directory, number, questions, parent, candidate, gate
                )
            receipt = {
                **change,
                "batch": number,
                "train_ids": [q["id"] for q in questions],
                "reflection_count": reflection_count,
                "gate": gate,
                "active_skill": active,
                "active_skill_hash": skill_sha256(active),
                "meta_replay": replay,
            }
            save_checkpoint(
                directory / "completed.json",
                {"input_hash": digest(key), "value": receipt},
            )
            save_checkpoint(
                self.root / "checkpoint.json",
                {"completed_batch": number, "active_skill_hash": skill_sha256(active)},
            )
            receipts.append(receipt)
        summary = {
            "phase": "phase1",
            "completed_batches": len(receipts),
            "accepted_updates": sum(r["gate"]["accepted"] for r in receipts),
            "train_count": len(train),
            "validation_count": len(validation),
            "test_executed": False,
            "meta_executed": False,
            "final_skill_hash": skill_sha256(active),
        }
        save_checkpoint(self.root / "final_skill.json", active)
        (self.root / "final_skill.md").write_text(
            compile_initial_skill(active), encoding="utf-8"
        )
        save_checkpoint(self.root / "summary.json", summary)
        return summary
