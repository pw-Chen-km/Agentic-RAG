"""Provider-neutral implementation of GraphRAG-Benchmark evaluation metrics.

The evaluator runs after an episode and receives only the text that was visible
to Policy.  A small client protocol keeps the metric logic testable without a
live Ollama server.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np


class SemanticJudgeClient(Protocol):
    def complete(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]: ...
    def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], dict[str, Any]]: ...


def _lcs(a: list[str], b: list[str]) -> int:
    row = [0] * (len(b) + 1)
    for x in a:
        previous = 0
        for j, y in enumerate(b, 1):
            old = row[j]
            row[j] = previous + 1 if x == y else max(row[j], row[j - 1])
            previous = old
    return row[-1]


def rouge_l(answer: str, reference: str) -> float:
    if not answer.strip() or not reference.strip():
        return 0.0
    try:
        from rouge_score import rouge_scorer
        return float(rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(reference, answer)["rougeL"].fmeasure)
    except ImportError:
        tokenize = lambda value: re.findall(r"\w+", value.casefold())
        a, r = tokenize(answer), tokenize(reference)
        overlap = _lcs(r, a)
        if not overlap or not a or not r:
            return 0.0
        precision, recall = overlap / len(a), overlap / len(r)
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _json_prompt(task: str, payload: Mapping[str, Any]) -> str:
    return (
        "You are a strict evaluation component. Return JSON only.\n"
        f"Task: {task}\nInput:\n{json.dumps(payload, ensure_ascii=False)}\n"
    )


def _statements(client: SemanticJudgeClient, text: str) -> tuple[list[str], dict[str, Any]]:
    if not text.strip():
        return [], {"status": "ok", "calls": 0}
    value, usage = client.complete(_json_prompt(
        "Split the text into self-contained atomic factual statements. Do not add facts.",
        {"text": text},
    ))
    statements = None
    if isinstance(value, Mapping):
        statements = value.get("statements") or value.get("facts") or value.get("atomic_statements")
        if statements is None and value and all(isinstance(key, str) for key in value):
            statements = list(value.keys())
    if not isinstance(statements, list) or not all(isinstance(item, str) for item in statements):
        raise ValueError("judge response missing statements list")
    return [item.strip() for item in statements if item.strip()], usage


def _classify(client: SemanticJudgeClient, answer: list[str], reference: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    value, usage = client.complete(_json_prompt(
        "Classify each answer statement against reference statements. verdict must be TP, FP, or FN; include reasons.",
        {"answer_statements": answer, "reference_statements": reference},
    ))
    rows = None
    if isinstance(value, Mapping):
        rows = value.get("items") or value.get("classifications") or value.get("results") or value.get("verdicts")
        if rows is None and value.get("verdict") is not None:
            rows = [value]
    if not isinstance(rows, list):
        raise ValueError("judge response missing classification items")
    counts = {"TP": 0, "FP": 0, "FN": 0}
    for item in rows:
        verdict = str(item.get("verdict", "")).upper() if isinstance(item, Mapping) else ""
        if verdict in counts:
            counts[verdict] += 1
    return counts, {"response": value, "usage": usage}


def _fraction_judged(client: SemanticJudgeClient, task: str, items: list[str], context: str) -> tuple[float | None, dict[str, Any]]:
    if not items:
        return 1.0, {"status": "ok", "items": []}
    value, usage = client.complete(_json_prompt(task, {"items": items, "context": context}))
    rows = None
    if isinstance(value, Mapping):
        rows = value.get("items") or value.get("verdicts") or value.get("results")
        if rows is None and any(key in value for key in ("attributed", "supported", "verdict")):
            rows = [value]
    if not isinstance(rows, list) or len(rows) != len(items):
        raise ValueError("judge response item count mismatch")
    verdicts = [int(bool(item.get("attributed", item.get("supported", 0)))) for item in rows if isinstance(item, Mapping)]
    if len(verdicts) != len(items):
        raise ValueError("judge response has invalid verdict rows")
    return sum(verdicts) / len(verdicts), {"response": value, "usage": usage}


class GraphRAGSemanticEvaluator:
    """Compute the official generation/retrieval metric family."""

    def __init__(self, client: SemanticJudgeClient) -> None:
        self.client = client

    def evaluate(self, *, question: Mapping[str, Any], answer: str | None, contexts: Sequence[str], evidence: Sequence[str] = ()) -> dict[str, Any]:
        answer_text = answer or ""
        reference = str(question.get("answer") or "")
        qtype = str(question.get("question_type") or "").casefold().replace(" ", "_")
        applicable = {
            "rouge_l": qtype in {"fact_retrieval", "complex_reasoning"} or not qtype,
            "answer_correctness": True,
            "coverage": qtype in {"contextual_summarize", "creative_generation"},
            "faithfulness": qtype == "creative_generation",
            "context_relevancy": True,
            "evidence_recall": bool(evidence),
        }
        result: dict[str, Any] = {"status": "ok", "applicable": applicable, "metrics": {}, "details": {}, "judge_usage": {"calls": 0, "input_tokens": None, "output_tokens": None, "total_tokens": None}}
        def run(name: str, fn: Any) -> None:
            if not applicable.get(name):
                result["metrics"][name] = None
                result["details"][name] = {"status": "not_applicable"}
                return
            try:
                value, detail = fn()
                result["metrics"][name] = value
                result["details"][name] = detail
            except Exception as exc:
                result["metrics"][name] = None
                result["details"][name] = {"status": "unavailable", "error": str(exc)}
        run("rouge_l", lambda: (rouge_l(answer_text, reference), {"status": "ok"}))
        def correctness() -> tuple[float, dict[str, Any]]:
            a, ad = _statements(self.client, answer_text); r, rd = _statements(self.client, reference)
            counts, detail = _classify(self.client, a, r)
            tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            vectors, emb_usage = self.client.embed([answer_text, reference])
            va, vr = np.asarray(vectors[0], dtype=float), np.asarray(vectors[1], dtype=float)
            denom = float(np.linalg.norm(va) * np.linalg.norm(vr))
            similarity = ((float(np.dot(va, vr)) / denom) + 1) / 2 if denom else 0.0
            return 0.75 * f1 + 0.25 * similarity, {"status": "ok", "counts": counts, "factuality_f1": f1, "semantic_similarity": similarity, "answer_statements": a, "reference_statements": r, "statement_details": {"answer": ad, "reference": rd, "classification": detail}, "embedding_usage": emb_usage}
        run("answer_correctness", correctness)
        context = "\n".join(dict.fromkeys(str(item) for item in contexts if str(item).strip()))
        run("coverage", lambda: _fraction_judged(self.client, "For each reference fact, decide whether the answer covers it. Use attributed=1 or 0.", _statements(self.client, reference)[0], answer_text))
        def faithfulness() -> tuple[float, dict[str, Any]]:
            statements, detail = _statements(self.client, answer_text)
            if not statements:
                return 1.0, {"status": "ok", "statements": []}
            if not context.strip():
                return 0.0, {"status": "ok", "statements": statements, "reason": "empty_context"}
            value, judged = _fraction_judged(self.client, "For each answer statement, decide whether it is directly supported by the context. Use supported=1 or 0.", statements, context)
            return value, {"statement_extraction": detail, "judgement": judged}
        run("faithfulness", faithfulness)
        def relevance() -> tuple[float, dict[str, Any]]:
            if not context.strip():
                return 0.0, {"status": "ok", "reason": "empty_context"}
            scores = []
            details = []
            for _ in range(2):
                value, usage = self.client.complete(_json_prompt("Rate context relevance to the question as 0, 1, or 2.", {"question": question.get("question", ""), "context": context}))
                score = int(value.get("score"))
                if score not in (0, 1, 2): raise ValueError("invalid relevance score")
                scores.append(score); details.append({"response": value, "usage": usage})
            return sum(scores) / len(scores) / 2, {"status": "ok", "scores": scores, "calls": details}
        run("context_relevancy", relevance)
        def evidence_recall() -> tuple[float, dict[str, Any]]:
            if not context.strip():
                return 0.0, {"status": "ok", "reason": "empty_context"}
            return _fraction_judged(self.client, "For each reference evidence statement, decide whether the context supports it. Use attributed=1 or 0.", list(evidence), context)
        run("evidence_recall", evidence_recall)
        if any(detail.get("status") == "unavailable" for detail in result["details"].values()):
            result["status"] = "partial"
        def collect(value: Any) -> tuple[int, int, int]:
            if isinstance(value, Mapping):
                calls = 1 if ("prompt_eval_count" in value or "eval_count" in value) else 0
                ins = int(value.get("prompt_eval_count") or 0) if calls else 0
                outs = int(value.get("eval_count") or 0) if calls else 0
                for child in value.values():
                    c, i, o = collect(child); calls += c; ins += i; outs += o
                return calls, ins, outs
            if isinstance(value, list):
                totals = [collect(item) for item in value]
                return tuple(sum(item[index] for item in totals) for index in range(3))  # type: ignore[return-value]
            return 0, 0, 0
        calls, input_tokens, output_tokens = collect(result["details"])
        result["judge_usage"] = {"calls": calls, "input_tokens": input_tokens or None, "output_tokens": output_tokens or None, "total_tokens": (input_tokens + output_tokens) or None}
        return result


class OllamaSemanticJudge:
    def __init__(self, *, model: str = "qwen3.5:4b", host: str = "http://localhost:11434", embedding_model: str = "nomic-embed-text", timeout: float = 600.0, max_retries: int = 2) -> None:
        try:
            from ollama import Client
        except ImportError as exc:
            raise RuntimeError("Install the ollama package to use semantic evaluation") from exc
        self.client = Client(host=host, timeout=timeout)
        self.model, self.embedding_model, self.max_retries = model, embedding_model, max_retries

    def complete(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat(model=self.model, messages=[{"role": "user", "content": prompt}], format={"type": "object"}, options={"temperature": 0, "top_p": 1, "num_ctx": 32768, "seed": 42, "num_predict": 2048}, think=False, stream=False)
                content = response.message.content if hasattr(response, "message") else response["message"]["content"]
                return json.loads(content), {"model": self.model, "attempt": attempt + 1, "prompt_eval_count": getattr(response, "prompt_eval_count", None), "eval_count": getattr(response, "eval_count", None)}
            except Exception as exc:
                last = exc; time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(str(last))

    def embed(self, texts: Sequence[str]) -> tuple[list[list[float]], dict[str, Any]]:
        response = self.client.embed(model=self.embedding_model, input=list(texts), options={"seed": 42})
        vectors = response.embeddings if hasattr(response, "embeddings") else response["embeddings"]
        return vectors, {"model": self.embedding_model}


__all__ = ["GraphRAGSemanticEvaluator", "OllamaSemanticJudge", "SemanticJudgeClient", "rouge_l"]
