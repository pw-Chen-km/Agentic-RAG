"""Observe actual requests without changing the upstream A-RAG tool loop."""

import copy
import time


class AuditClient:
    def __init__(self, client):
        self.client = client
        self.records = []
        self.context_checks = []

    @property
    def enforces_context_window(self):
        return getattr(self.client, "enforces_context_window", False)

    def inspect_context(self, **kwargs):
        result = self.client.inspect_context(**kwargs)
        self.context_checks.append(copy.deepcopy(result))
        return result

    def chat(self, **kwargs):
        record = {"decision": len(self.records) + 1, "request": copy.deepcopy(kwargs)}
        self.records.append(record)
        started = time.monotonic()
        try:
            response = self.client.chat(**kwargs)
            record["response"] = copy.deepcopy(response)
            return response
        except Exception as exc:
            record["error"] = str(exc)
            if hasattr(exc, "details"):
                record["error_details"] = copy.deepcopy(exc.details)
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started


def attach_audit(raw, client):
    result = copy.deepcopy(raw)
    result["audit_version"] = "arag-decisions-v1"
    result["decision_audit"] = copy.deepcopy(client.records)
    if client.context_checks:
        result["context_checks"] = copy.deepcopy(client.context_checks)
    complete = all(
        "response" in r
        and r["response"].get("input_tokens") is not None
        and r["response"].get("output_tokens") is not None
        and r["response"].get("usage_complete", True)
        for r in client.records
    )
    result["usage"] = {
        "complete": complete,
        "input_tokens": sum(r["response"]["input_tokens"] for r in client.records)
        if complete
        else None,
        "output_tokens": sum(r["response"]["output_tokens"] for r in client.records)
        if complete
        else None,
    }
    result["usage"]["total_tokens"] = (
        (result["usage"]["input_tokens"] + result["usage"]["output_tokens"])
        if complete
        else None
    )
    result["policy_calls"] = len(client.records)
    result["tool_calls"] = len(raw.get("trajectory", []))
    errors = [r["error"] for r in client.records if "error" in r]
    result["execution_errors"] = errors
    if raw.get("error") and raw["error"] not in errors:
        errors.append(raw["error"])
    result["termination_reason"] = (
        "service_error_then_forced_answer"
        if errors
        else "token_budget_forced_answer"
        if raw.get("token_budget_exceeded")
        else "loop_limit_forced_answer"
        if raw.get("max_loops_exceeded")
        else "model_answered"
    )
    return result
