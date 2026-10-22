"""Pinned Ollama transport with exact GGUF token accounting and no truncation.

Ollama v0.35.0 renders the *actual* chat template, including tool definitions,
without generation. Its own llama-server then tokenizes that rendered prompt.
No Python approximation of Qwen's template/tokenizer and no extra LLM inference
is used. This local adapter deliberately fails closed on an unsupported server.
"""

import copy
import hashlib
import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from .llm import LLMClient

VERSION = "ollama-exact-context-v1"
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def endpoint_lock(url):
    key = url.removesuffix("/v1").rstrip("/")
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


class OllamaRequestError(RuntimeError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


class ContextOverflowError(OllamaRequestError):
    pass


def runner_url(endpoint, proc_root=Path("/proc"), model_path=None):
    """Find only the local runner owned by the requested Ollama listener.

    Ports are ephemeral after model reloads. Never guess a runner port, accept
    a remote tokenizer, or choose a runner belonging to the other replica.
    """
    parsed = urlparse(endpoint)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("exact GGUF tokenization requires co-located Ollama")
    found = []
    # Some deployments run the Ollama daemon under a dedicated service
    # account.  A caller running as an ordinary user can then inspect the
    # runner command line and model path but cannot read the daemon parent's
    # /proc/<pid>/environ.  Keep the normal listener-ownership check whenever
    # it is readable; otherwise retain a narrowly bounded fallback that still
    # requires exactly one runner for the exact GGUF path reported by
    # /api/show.  Multiple runners remain a hard failure.
    permission_fallback = []
    for proc in proc_root.iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\x00")
            args = [a.decode() for a in argv if a]
            if not args or Path(args[0]).name != "llama-server":
                continue
            if model_path is not None:
                if args[args.index("--model") + 1] != model_path:
                    continue
            status = (proc / "status").read_text()
            ppid = next(line.split()[1] for line in status.splitlines() if line.startswith("PPid:"))
            parent = proc_root / ppid
            try:
                env = dict(e.split(b"=", 1) for e in (parent / "environ").read_bytes().split(b"\x00") if b"=" in e)
            except PermissionError:
                if model_path is not None:
                    port = int(args[args.index("--port") + 1])
                    permission_fallback.append((port, int(proc.name)))
                continue
            host = env.get(b"OLLAMA_HOST", b"").decode()
            host = urlparse(host if "://" in host else "http://" + host)
            if host.port != parsed.port or host.hostname not in {"localhost", "127.0.0.1"}:
                continue
            port = int(args[args.index("--port") + 1])
            found.append((port, int(proc.name)))
        except (OSError, ValueError, StopIteration, UnicodeError):
            continue
    if len(found) == 1 and not permission_fallback:
        return f"http://127.0.0.1:{found[0][0]}", found[0][1]
    if not found and len(permission_fallback) == 1:
        return f"http://127.0.0.1:{permission_fallback[0][0]}", permission_fallback[0][1]
    raise RuntimeError(
        f"expected one exact tokenizer runner for {endpoint}, found {found or permission_fallback}"
    )


def native_messages(messages):
    result = copy.deepcopy(messages)
    for msg in result:
        if "reasoning_content" in msg:
            msg["thinking"] = msg.pop("reasoning_content")
        for call in msg.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
            if not isinstance(function.get("arguments"), dict):
                raise ValueError("tool arguments must be an object")
    return result


class GuardedOllamaClient(LLMClient):
    enforces_context_window = True

    def __init__(self, *, context_length, expected_digest, input_token_limit=128000,
                 safety_tokens=256, read_timeout=300, total_timeout=1800,
                 max_transport_attempts=2, expected_version="0.35.0",
                 require_complete_output=False, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = self.base_url.removesuffix("/v1")
        self.context_length = context_length
        self.expected_digest = expected_digest
        self.input_token_limit = input_token_limit
        self.safety_tokens = safety_tokens
        self.read_timeout, self.total_timeout = read_timeout, total_timeout
        self.max_transport_attempts = max_transport_attempts
        self.expected_version = expected_version
        self.require_complete_output = require_complete_output
        self._verified = False
        if not (context_length > self.max_tokens + safety_tokens > 0 and input_token_limit > 0):
            raise ValueError("context must reserve generation and safety tokens")
        if not 1 <= max_transport_attempts <= 3 or read_timeout <= 0 or total_timeout <= 0:
            raise ValueError("invalid bounded transport settings")

    def payload(self, messages, tools=None, temperature=None, max_tokens=None, format=None):
        think = False if self.reasoning_effort in {None, "none"} else self.reasoning_effort
        body = {
            "model": self.model, "messages": native_messages(messages),
            "stream": True, "think": think, "truncate": False, "shift": False,
            "keep_alive": -1,
            "options": {"num_ctx": self.context_length,
                        "num_predict": self.max_tokens if max_tokens is None else max_tokens,
                        "temperature": self.temperature if temperature is None else temperature},
        }
        if self.seed is not None:
            body["options"]["seed"] = self.seed
        if tools:
            body["tools"] = copy.deepcopy(tools)
        if format:
            body["format"] = copy.deepcopy(format)
        return body

    def _verify_version(self):
        if not self._verified:
            version = requests.get(self.endpoint + "/api/version", timeout=15)
            version.raise_for_status()
            if version.json().get("version") != self.expected_version:
                raise RuntimeError("Ollama version changed; exact renderer adapter must be revalidated")
            self._verified = True

    def measure(self, payload):
        # Every preflight failure is typed and recorded before generation.  A
        # plain RuntimeError would make the legacy BaseAgent fall through to
        # its emergency "force an answer" path, which is precisely what an
        # unavailable tokenizer, version, or model must never do.
        try:
            self._verify_version()
            debug = {**payload, "stream": False, "_debug_render_only": True}
            r = requests.post(self.endpoint + "/api/chat", json=debug, timeout=(15, self.read_timeout))
            if not r.ok:
                raise OllamaRequestError("prompt rendering failed: " + r.text, {"stage": "render", "http_status": r.status_code, "body": r.text, "generation_attempts": []})
            prompt = r.json().get("_debug_info", {}).get("rendered_template")
            if not isinstance(prompt, str) or not prompt:
                raise RuntimeError("exact rendered prompt unavailable; refusing estimated tokenization")
            ps = requests.get(self.endpoint + "/api/ps", timeout=15)
            ps.raise_for_status()
            loaded = [m for m in ps.json()["models"] if m["name"] == self.model]
            if len(loaded) != 1 or loaded[0]["digest"] != self.expected_digest:
                raise RuntimeError("loaded tokenizer/model digest mismatch")
            if loaded[0].get("context_length") != self.context_length:
                raise RuntimeError("requested and allocated contexts differ")
            show = requests.post(self.endpoint + "/api/show", json={"model": self.model}, timeout=15)
            show.raise_for_status()
            match = re.search(r"(?m)^FROM (.+)$", show.json().get("modelfile", ""))
            if not match:
                raise RuntimeError("loaded GGUF path unavailable")
            url, pid = runner_url(self.endpoint, model_path=match[1].strip())
            r = requests.post(url + "/tokenize", json={"content": prompt, "add_special": False}, timeout=(15, 60))
            r.raise_for_status()
            tokens = r.json().get("tokens")
            if not isinstance(tokens, list) or not all(type(t) is int for t in tokens):
                raise RuntimeError("loaded GGUF tokenizer returned invalid tokens")
            return {"version": VERSION, "tokenizer": "loaded_qwen_gguf", "ollama_version": self.expected_version,
                    "model_digest": self.expected_digest, "context_length": self.context_length,
                    "input_tokens": len(tokens), "output_reserve": payload["options"]["num_predict"],
                    "safety_tokens": self.safety_tokens, "input_token_limit": self.input_token_limit,
                    "effective_input_limit": min(self.input_token_limit, self.context_length - payload["options"]["num_predict"] - self.safety_tokens),
                    "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "rendered_prompt_characters": len(prompt), "tokenizer_runner_pid": pid,
                    "truncate": False, "shift": False}
        except OllamaRequestError:
            raise
        except Exception as exc:
            raise OllamaRequestError(
                "context preflight failed: " + str(exc),
                {"stage": "context_preflight", "error": str(exc), "generation_attempts": []},
            ) from exc

    def chat(self, messages, tools=None, temperature=None, max_tokens=None, format=None):
        body = self.payload(messages, tools, temperature, max_tokens, format)
        start_wait = time.monotonic()
        with endpoint_lock(self.endpoint):
            waited = time.monotonic() - start_wait
            count = self.measure(body)
            count["endpoint_wait_seconds"] = waited
            if count["input_tokens"] > count["effective_input_limit"]:
                raise ContextOverflowError(
                    f"input_too_long: {count['input_tokens']} Qwen tokens > {count['effective_input_limit']} allowed; no generation or truncation",
                    {"stage": "context_preflight", "context": count, "generation_attempts": []},
                )
            return self._generate(body, count)

    def inspect_context(self, messages, tools=None, **kwargs):
        """Measure the actual rendered request without generating or shortening it."""
        body = self.payload(messages, tools, **kwargs)
        with endpoint_lock(self.endpoint):
            return self.measure(body)

    def _generate(self, body, count):
        attempts = []
        for attempt in range(1, self.max_transport_attempts + 1):
            response, chunks, started = None, [], time.monotonic()
            item = {"attempt": attempt, "stage": "generation", "endpoint": self.endpoint}
            try:
                response = requests.post(self.endpoint + "/api/chat", json=body, stream=True,
                                         timeout=(15, self.read_timeout))
                item["http_status"] = response.status_code
                if not response.ok:
                    item["body"] = response.text
                    item["error"] = "http_error"
                    if response.status_code in {429, 502, 503, 504} and attempt < self.max_transport_attempts:
                        attempts.append(item)
                        time.sleep(attempt)
                        continue
                    raise OllamaRequestError("Ollama HTTP error: " + response.text, {"stage": "generation", "context": count, "generation_attempts": attempts + [item]})
                for line in response.iter_lines():
                    if time.monotonic() - started > self.total_timeout:
                        raise TimeoutError("generation exceeded total wall time limit")
                    if not line:
                        continue
                    chunk = json.loads(line)
                    chunks.append(chunk)
                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"])
                if not chunks or chunks[-1].get("done") is not True:
                    raise RuntimeError("incomplete generation stream")
                result = copy.deepcopy(chunks[-1])
                result["message"] = {"role": "assistant", "content": "".join(c.get("message", {}).get("content", "") for c in chunks)}
                thinking = "".join(c.get("message", {}).get("thinking", "") for c in chunks)
                calls = [call for c in chunks for call in c.get("message", {}).get("tool_calls", [])]
                message = copy.deepcopy(result["message"])
                if thinking:
                    message["reasoning_content"] = thinking
                    result["message"]["thinking"] = thinking
                if calls:
                    normalized = []
                    for i, call in enumerate(calls):
                        fn = call["function"]
                        args = fn.get("arguments", {})
                        if not isinstance(args, dict):
                            raise ValueError("native tool arguments are not an object")
                        normalized.append({"id": call.get("id") or "call_" + count["rendered_prompt_sha256"][:20] + "_" + str(i),
                                           "type": "function", "function": {"name": fn["name"], "arguments": json.dumps(args, ensure_ascii=False)}})
                    message["tool_calls"] = normalized
                    result["message"]["tool_calls"] = calls
                item.update(elapsed_seconds=time.monotonic() - started, complete=True,
                            input_tokens=result.get("prompt_eval_count"), output_tokens=result.get("eval_count"))
                attempts.append(item)
                measured = result.get("prompt_eval_count")
                if measured is None or measured > count["input_tokens"] + self.safety_tokens or (
                    self.require_complete_output and measured < count["input_tokens"] - self.safety_tokens
                ):
                    raise OllamaRequestError("token accounting mismatch", {"context": count, "generation_attempts": attempts, "raw_response": result})
                count["server_input_tokens"] = measured
                count["server_token_difference"] = measured - count["input_tokens"]
                if self.require_complete_output and result.get("done_reason") != "stop":
                    raise OllamaRequestError("incomplete_agent_output: " + str(result.get("done_reason")),
                        {"stage": "output_integrity", "context": count,
                         "generation_attempts": attempts, "raw_response": result})
                # A response after an uncertain generation timeout cannot hide
                # the cost of the failed attempt from the validation gate.
                usage_complete = all(a.get("complete") or a.get("http_status") in {429, 502, 503, 504} for a in attempts)
                return {"message": message, "input_tokens": result.get("prompt_eval_count"),
                        "output_tokens": result.get("eval_count"), "cost": 0,
                        "raw_response": result, "context_accounting": count,
                        "provider_attempts": attempts, "usage_complete": usage_complete,
                        "generation_attempt_count": len(attempts), "transport": VERSION}
            except OllamaRequestError:
                raise
            except (requests.RequestException, TimeoutError, RuntimeError, ValueError) as exc:
                item.update(error=str(exc), elapsed_seconds=time.monotonic() - started,
                            received_chunks=len(chunks), usage_unknown=True)
                attempts.append(item)
                # Retry only a transient network timeout before any output.
                transient = isinstance(exc, (requests.RequestException, TimeoutError))
                if transient and not chunks and attempt < self.max_transport_attempts:
                    time.sleep(attempt)
                    continue
                raise OllamaRequestError(str(exc), {"stage": "generation", "context": count, "generation_attempts": attempts, "partial_chunks": chunks}) from exc
            finally:
                if response is not None:
                    response.close()
        raise AssertionError("bounded attempts exhausted without a result")


