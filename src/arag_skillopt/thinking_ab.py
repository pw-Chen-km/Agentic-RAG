"""Replay frozen optimizer requests with ONLY thinking changed.

Both arms are new Ollama native /api/chat calls. Historical answers are reference
material, never the off control. No agent rollout, Judge, gate, or Skill update.
"""

import copy
import fcntl
import hashlib
import json
import re
import time
from pathlib import Path

import requests

from .checkpoint import load_checkpoint, save_checkpoint
from .optimizer_tools import parse_tool_call
from .skill_rules import RuleStore
from .support import digest
from .tokens import count

MODEL = "qwen3.8:27b-q4_K_M"
MODEL_DIGEST = "25b843619e944cd0ae6069f94ff4e5e26a16e109ccbc0a66a0f05979ed70098e"


def saved_request(path):
    """Accept an actual saved attempt, not a reconstructed trajectory/prompt."""
    path = Path(path)
    source = json.loads(path.read_text(encoding="utf-8"))
    value = source.get("value", source)
    request = value.get("request", value.get("request_not_sent"))
    if not isinstance(request, dict) or set(request) != {"messages", "tools"}:
        raise ValueError(f"need saved messages/tools request: {path}")
    if not isinstance(request["messages"], list) or not isinstance(
        request["tools"], list
    ):
        raise TypeError("saved request lists required")
    if not request["messages"] or not request["tools"]:
        raise ValueError("empty saved optimizer request")
    return {
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "request": copy.deepcopy(request),
        "request_hash": digest(request),
        "budget_input_tokens": count(request),
        "historical_response": value.get("response"),
    }


def parse_edit_response(raw, request):
    """Report structural validity only; it is NOT evidence of better diagnosis."""
    calls = raw.get("message", {}).get("tool_calls") or []
    is_merge = any(
        "Batch proposals (all groups):" in m.get("content", "")
        or "Batch diagnoses (all groups):" in m.get("content", "")
        for m in request["messages"]
    )
    if not 1 <= len(calls) <= (3 if is_merge else 2):
        raise ValueError("missing or too many edit tool calls")
    allowed = {t["function"]["name"] for t in request["tools"]}
    if any(c.get("function", {}).get("name") not in allowed for c in calls):
        raise ValueError("model called a tool outside the saved tool schema")
    edits = [parse_tool_call(c) for c in calls]
    if any(e["operation"] == "no_change" for e in edits) and len(edits) != 1:
        raise ValueError("no_change must be exclusive")
    ids = set(
        re.findall(
            r"^Case ID: (.+)$",
            "\n".join(m.get("content", "") for m in request["messages"]),
            flags=re.MULTILINE,
        )
    )
    if ids:
        for edit in edits:
            if edit["operation"] != "no_change":
                support = set(edit["supporting_case_ids"])
                if len(support) < 2 or not support <= ids:
                    raise ValueError("edit requires two supplied independent cases")
    # Frozen requests include the actual editable Skill once. Check the old
    # interface/budgets without ever applying a result to the active Skill.
    for message in request["messages"]:
        if "Current Skill:\n" in message.get("content", ""):
            skill, _ = json.JSONDecoder().raw_decode(
                message["content"].split("Current Skill:\n", 1)[1]
            )
            RuleStore(skill, max_edits=3 if is_merge else 2).apply(edits)
            break
    return edits


class ThinkingAB:
    def __init__(
        self,
        endpoint,
        output,
        model=MODEL,
        expected_digest=MODEL_DIGEST,
        num_ctx=32768,
        num_predict=16384,
        seed=42,
        timeout=900,
        session=None,
        thinking_on="auto",
    ):
        self.url = endpoint.removesuffix("/v1").rstrip("/")
        self.root = Path(output)
        self.model, self.expected_digest = model, expected_digest
        self.num_ctx, self.num_predict, self.seed, self.timeout = (
            num_ctx,
            num_predict,
            seed,
            timeout,
        )
        if num_ctx <= 0 or num_predict <= 0 or timeout <= 0 or not expected_digest:
            raise ValueError(
                "positive budgets/timeout and an expected model digest required"
            )
        self.http = session or requests.Session()
        self.thinking_on = thinking_on

    def verify_model(self):
        response = self.http.get(self.url + "/api/tags", timeout=15)
        response.raise_for_status()
        matches = [m for m in response.json()["models"] if m["name"] == self.model]
        if len(matches) != 1 or matches[0]["digest"] != self.expected_digest:
            raise ValueError("model digest differs from frozen experiment")
        response = self.http.post(
            self.url + "/api/show", json={"model": self.model}, timeout=15
        )
        response.raise_for_status()
        shown = response.json()
        self.last_environment = {"digest": matches[0]["digest"], "show": shown}
        values = shown.get("thinking", {}).get("values")
        if values is not None:
            if False not in values:
                raise ValueError(
                    f"model cannot disable thinking: thinking.values={values!r}"
                )
            on = self.thinking_on
            if on == "auto":
                on = True if True in values else shown["thinking"].get("default")
            if on not in values or on is False or on is None:
                raise ValueError(
                    f"no supported thinking-on control: thinking.values={values!r}"
                )
        elif "thinking" not in shown.get("capabilities", []):
            raise ValueError(
                "cannot confirm this model supports thinking; no A/B calls sent"
            )
        else:
            on = True if self.thinking_on == "auto" else self.thinking_on
        if hasattr(self, "on_control") and self.on_control != on:
            raise ValueError("thinking controls changed during comparison")
        self.on_control = on
        self.last_environment["resolved_thinking_controls"] = {"off": False, "on": on}
        return self.last_environment

    def payload(self, source, think):
        return {
            "model": self.model,
            **copy.deepcopy(source["request"]),
            "stream": False,
            "think": self.on_control if think else False,
            "options": {
                "temperature": 0,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }

    def run(self, paths, execute=False, progress=None):
        sources = [saved_request(p) for p in paths]
        if not sources or len({s["request_hash"] for s in sources}) != len(sources):
            raise ValueError("choose nonempty, distinct saved requests")
        contract = {
            "protocol": "frozen-optimizer-thinking-ab-v1",
            "runner_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "endpoint": self.url,
            "model": self.model,
            "expected_digest": self.expected_digest,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "seed": self.seed,
            "temperature": 0,
            "transport": "ollama_api_chat",
            "inputs": [
                {
                    k: s[k]
                    for k in (
                        "source",
                        "source_sha256",
                        "request_hash",
                        "budget_input_tokens",
                    )
                }
                for s in sources
            ],
            "sole_treatment": "think",
            "thinking_on_request": self.thinking_on,
            "new_calls_in_both_arms": True,
            "historical_responses_are_not_controls": True,
            "agent_rollouts": 0,
            "skill_updates": 0,
            "budget_tokenizer": "o200k_base (not the model tokenizer)",
        }
        if not execute:
            return {
                "status": "dry_run",
                "contract": contract,
                "planned_calls": 2 * len(sources),
            }
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".thinking-ab.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            previous = load_checkpoint(self.root / "contract.json")
            if previous and previous != contract:
                raise ValueError("A/B contract changed: use a new output directory")
            if not previous and any(
                p.name != ".thinking-ab.lock" for p in self.root.iterdir()
            ):
                raise ValueError("nonempty A/B output without contract")
            save_checkpoint(self.root / "contract.json", contract)
            try:
                environment = self.verify_model()  # fail before any generation
            except (requests.RequestException, ValueError, TypeError) as exc:
                save_checkpoint(
                    self.root / "preflight.json",
                    {
                        "status": "failed",
                        "error": str(exc),
                        "generation_calls_in_this_attempt": 0,
                        "environment": getattr(self, "last_environment", None),
                    },
                )
                raise
            save_checkpoint(self.root / "preflight.json", {"status": "passed"})
            previous_environment = load_checkpoint(self.root / "environment.json")
            if (
                previous_environment
                and previous_environment.get("resolved_thinking_controls")
                != environment["resolved_thinking_controls"]
            ):
                raise ValueError(
                    "thinking controls changed: use a new output directory"
                )
            save_checkpoint(self.root / "environment.json", environment)
            trials = []
            for i, source in enumerate(sources):
                save_checkpoint(self.root / f"{i:03d}" / "frozen_source.json", source)
                pair = []
                # Alternate pair order to reduce the always-cold-off-arm confound.
                for think in (False, True) if i % 2 == 0 else (True, False):
                    request = self.payload(source, think)
                    path = self.root / f"{i:03d}" / ("on.json" if think else "off.json")
                    signature = digest(request)
                    saved = load_checkpoint(path)
                    if saved:
                        if saved["input_hash"] != signature:
                            raise ValueError("thinking trial inputs changed")
                        result = saved["value"]
                    else:
                        self.verify_model()  # no silent model replacement on resume
                        if progress:
                            progress(f"sample {i + 1}/{len(sources)}, think={think}")
                        started = time.monotonic()
                        result = {
                            "think": think,
                            "request": request,
                            "source_request_hash": source["request_hash"],
                        }
                        try:
                            response = self.http.post(
                                self.url + "/api/chat",
                                json=request,
                                timeout=self.timeout,
                            )
                            response.raise_for_status()
                            raw = response.json()
                            if not isinstance(raw, dict) or not isinstance(
                                raw.get("message"), dict
                            ):
                                raise TypeError("provider returned no message object")
                            result.update(
                                raw_response=raw,
                                status="ok",
                                thinking=raw.get("message", {}).get("thinking", ""),
                                input_tokens=raw.get("prompt_eval_count"),
                                output_tokens=raw.get("eval_count"),
                            )
                            try:
                                result["parsed_edits"] = parse_edit_response(
                                    raw, source["request"]
                                )
                                result["parse_status"] = "ok"
                            except (ValueError, TypeError, KeyError) as exc:
                                result.update(
                                    parse_status="invalid", parse_error=str(exc)
                                )
                            if (
                                not raw.get("done")
                                or raw.get("done_reason") == "length"
                            ):
                                result["status"] = "incomplete_generation"
                            if think and not result["thinking"]:
                                result["thinking_status"] = (
                                    "not_observed; do not assume enabled"
                                )
                            elif not think and result["thinking"]:
                                result["thinking_status"] = "unexpected_in_off_arm"
                            else:
                                result["thinking_status"] = (
                                    "observed" if think else "not_observed"
                                )
                        except (
                            requests.RequestException,
                            ValueError,
                            TypeError,
                        ) as exc:
                            result.update(
                                status="provider_error",
                                error=str(exc),
                                parse_status="unavailable",
                            )
                        result["elapsed_seconds"] = time.monotonic() - started
                        save_checkpoint(
                            path, {"input_hash": signature, "value": result}
                        )
                    pair.append(
                        {
                            k: result.get(k)
                            for k in (
                                "think",
                                "status",
                                "parse_status",
                                "parse_error",
                                "thinking_status",
                                "input_tokens",
                                "output_tokens",
                                "elapsed_seconds",
                            )
                        }
                    )
                save_checkpoint(
                    self.root / "checkpoint.json",
                    {"completed_pairs": i + 1, "total_pairs": len(sources)},
                )
                trials.append({"source": source["source"], "arms": pair})
            summary = {
                "status": "complete"
                if all(
                    a["status"] == "ok" and a["parse_status"] == "ok"
                    for p in trials
                    for a in p["arms"]
                )
                else "completed_with_failures",
                "pairs": trials,
                "skill_updates": 0,
                "quality_conclusion": "Requires grounded case review; parse success is not diagnostic accuracy.",
            }
            save_checkpoint(self.root / "summary.json", summary)
            return summary
