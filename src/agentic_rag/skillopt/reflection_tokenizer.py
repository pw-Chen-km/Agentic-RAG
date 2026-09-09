"""Read-only exact token counting through a pinned AMD Ollama backend.

No model is loaded, unloaded, or asked to generate. The remote helper verifies
the service's loaded model, identifies the listener-owning ``ollama serve``
process, and only considers matching llama-server descendants of that process.
Prompts travel on SSH stdin, never in command arguments or exception messages.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from typing import Any, Mapping, Sequence


MODEL = "qwen3.6:35b-a3b-bf16"
MODEL_DIGEST = "94061ddd23a7de9c902f7bc468455aff03564d26e929b7e2e9ce62c5e6a3a492"
MODEL_BLOB_DIGEST = "b2e1285b022c507b1c369ed296465cdfd1743c81010c207a2fb1fcda69c87107"
METHOD = "llama_server_apply_template_and_tokenize"
CONTEXT_LENGTH = 262144
TIMEOUT_SECONDS = 120


class ReflectionTokenizationError(RuntimeError):
    """A count was not verified; callers must stop rather than guess or truncate."""


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# This constant contains program text only. Runtime prompts are sent separately
# as JSON on stdin, and the helper does not create any remote files.
_REMOTE_PROGRAM = r'''
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


class VerificationError(Exception):
    pass


def fail(code):
    raise VerificationError(code)


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def flag(args, name):
    values = []
    for index, value in enumerate(args):
        if value == name:
            if index + 1 >= len(args):
                fail('invalid_backend_arguments')
            values.append(args[index + 1])
        elif value.startswith(name + '='):
            values.append(value.split('=', 1)[1])
    if len(values) != 1:
        fail('missing_or_ambiguous_backend_argument')
    return values[0]


def process_snapshot():
    records = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            args = (directory / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            records[int(directory.name)] = {'pid': int(directory.name), 'ppid': int(fields[1]),
                                            'start': fields[19], 'args': args}
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    return records


def listener_owners(port):
    inodes = set()
    for name in ('tcp', 'tcp6'):
        try:
            lines = (Path('/proc/net') / name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) >= 10 and fields[3] == '0A' and int(fields[1].rsplit(':', 1)[1], 16) == port:
                inodes.add(fields[9])
    if not inodes:
        fail('listener_not_found')
    owners = set()
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            for fd in (directory / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith('socket:[') and target[8:-1] in inodes:
                    owners.add(int(directory.name))
        except OSError:
            continue
    return owners


def http_json(port, path, payload, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        fail('tokenization_deadline_exceeded')
    request = urllib.request.Request('http://127.0.0.1:' + str(port) + path,
        data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode(),
        headers={'Content-Type': 'application/json'}, method='GET' if payload is None else 'POST')
    with urllib.request.urlopen(request, timeout=min(remaining, 110)) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        fail('invalid_http_response')
    return value


def loaded_model(value, expected):
    models = value.get('models')
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        fail('expected_one_loaded_model_no_auto_load')
    model = models[0]
    if (model.get('digest', '').removeprefix('sha256:') != expected['model_digest'] or
            expected['model'] not in (model.get('name'), model.get('model'))):
        fail('loaded_model_mismatch_no_replacement')
    if model.get('context_length') != expected['context_length']:
        fail('loaded_context_mismatch')
    size, vram = model.get('size'), model.get('size_vram')
    if type(size) is not int or type(vram) is not int or size <= 0 or vram < size:
        fail('cpu_offload_or_unknown_gpu_residency')
    return model


def descends_from(pid, parent_pid, processes):
    seen = set()
    while pid in processes and pid not in seen:
        seen.add(pid)
        pid = processes[pid]['ppid']
        if pid == parent_pid:
            return True
    return False


def count(payload, http=None, snapshot=None, listeners=None):
    http, snapshot, listeners = http or http_json, snapshot or process_snapshot, listeners or listener_owners
    messages = payload['messages']
    if sha(messages) != payload.get('prompt_sha256'):
        fail('prompt_hash_mismatch')
    deadline = time.monotonic() + 110
    service_port = payload['amd_port']
    loaded_model(http(service_port, '/api/ps', None, deadline), payload)
    processes = snapshot()
    owners = listeners(service_port)
    # Port ownership, not inherited environment variables, identifies the service.
    services = [processes[pid] for pid in owners if pid in processes and processes[pid]['args']
                and Path(processes[pid]['args'][0]).name == 'ollama' and 'serve' in processes[pid]['args'][1:]]
    if len(services) != 1 or owners != {services[0]['pid']}:
        fail('missing_or_ambiguous_ollama_service')
    service = services[0]
    descendants = [row for row in processes.values() if row['args']
                   and Path(row['args'][0]).name == 'llama-server'
                   and descends_from(row['pid'], service['pid'], processes)]
    expected_names = {payload['model_blob_digest'], 'sha256-' + payload['model_blob_digest']}
    backends = [row for row in descendants if Path(flag(row['args'], '--model')).name in expected_names]
    if len(backends) != 1:
        fail('missing_or_ambiguous_matching_backend')
    backend = backends[0]
    try:
        backend_port = int(flag(backend['args'], '--port'))
    except ValueError:
        fail('invalid_backend_port')
    if not 1 <= backend_port <= 65535 or listeners(backend_port) != {backend['pid']}:
        fail('backend_listener_ownership_mismatch')
    template = flag(backend['args'], '--chat-template')
    applied = http(backend_port, '/apply-template', {
        'messages': messages, 'add_generation_prompt': True, 'enable_thinking': True,
    }, deadline)
    rendered = applied.get('prompt')
    if not isinstance(rendered, str) or not rendered:
        fail('backend_did_not_return_complete_chat_template')
    result = http(backend_port, '/tokenize', {'content': rendered, 'add_special': False}, deadline)
    tokens = result.get('tokens')
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
        fail('backend_did_not_return_exact_tokens')
    # Refuse a count if the chosen process/model changed while it was measured.
    loaded_model(http(service_port, '/api/ps', None, deadline), payload)
    after = snapshot()
    for previous in (service, backend):
        current = after.get(previous['pid'])
        if not current or any(current.get(key) != previous.get(key) for key in ('start', 'ppid', 'args')):
            fail('service_or_backend_changed_during_count')
    if listeners(service_port) != {service['pid']} or listeners(backend_port) != {backend['pid']}:
        fail('listener_changed_during_count')
    return {'input_tokens': len(tokens), 'model_digest': payload['model_digest'],
            'model_blob_digest': payload['model_blob_digest'], 'model': payload['model'],
            'context_length': payload['context_length'], 'method': 'llama_server_apply_template_and_tokenize',
            'prompt_sha256': payload['prompt_sha256'], 'backend_pid': backend['pid'], 'backend_port': backend_port,
            'backend_start_time': backend['start'], 'service_pid': service['pid'], 'service_start_time': service['start'],
            'amd_port': service_port, 'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest(),
            'rendered_prompt_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
            'rendered_prompt_chars': len(rendered), 'enable_thinking': True,
            'add_generation_prompt': True, 'verified_before_and_after': True, 'generation_calls': 0}


if __name__ == '__main__':
    try:
        result = count(json.load(sys.stdin))
        print(json.dumps({'ok': True, 'result': result}, ensure_ascii=False))
    except VerificationError as error:
        print(json.dumps({'ok': False, 'error_code': str(error)}))
        sys.exit(2)
    except Exception as error:
        print(json.dumps({'ok': False, 'error_code': 'remote_tokenizer_error', 'error_type': type(error).__name__}))
        sys.exit(2)
'''


class SshOllamaTokenCounter:
    def __init__(self, config: Mapping[str, Any]):
        self.host = config.get("reflection_tokenizer_ssh_host")
        self.ssh_port = config.get("reflection_tokenizer_ssh_port")
        self.key = config.get("reflection_tokenizer_ssh_key")
        self.amd_port = config.get("reflection_tokenizer_amd_port")
        self.model_digest = config.get("expected_model_digest", MODEL_DIGEST)
        if not isinstance(self.host, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.:\[\]-]+", self.host):
            raise ValueError("reflection_tokenizer_ssh_host must be a user@host, without shell options")
        if not isinstance(self.key, str) or not self.key.startswith("/") or any(char in self.key for char in ("\0", "\n", "\r")):
            raise ValueError("reflection_tokenizer_ssh_key must be an absolute key path")
        if any(type(port) is not int or not 1 <= port <= 65535 for port in (self.ssh_port, self.amd_port)):
            raise ValueError("SSH and AMD Ollama ports must be explicit valid integers")
        if not isinstance(self.model_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", self.model_digest):
            raise ValueError("expected_model_digest must be a complete SHA-256")

    def __call__(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("Token counting requires complete nonempty chat messages")
        copied = []
        for message in messages:
            if not isinstance(message, Mapping) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
                raise ValueError("Reflection messages must contain role and complete text content")
            copied.append(dict(message))
        try:
            prompt_hash = _sha(copied)
        except (TypeError, ValueError) as error:
            raise ValueError("Reflection messages must be JSON serializable") from error
        payload = {"messages": copied, "prompt_sha256": prompt_hash, "model": MODEL,
                   "model_digest": self.model_digest, "model_blob_digest": MODEL_BLOB_DIGEST,
                   "amd_port": self.amd_port, "context_length": CONTEXT_LENGTH}
        argv = ["ssh", "-i", self.key, "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=20", "-p", str(self.ssh_port),
                self.host, "python3 -c " + shlex.quote(_REMOTE_PROGRAM)]
        try:
            completed = subprocess.run(argv, input=json.dumps(payload, ensure_ascii=False), capture_output=True,
                                       text=True, timeout=TIMEOUT_SECONDS, check=False)
        except subprocess.TimeoutExpired as error:
            raise ReflectionTokenizationError("ssh_tokenizer_timeout_no_count") from None
        except OSError as error:
            raise ReflectionTokenizationError("ssh_tokenizer_unavailable_no_count") from None
        try:
            response = json.loads(completed.stdout)
        except (TypeError, ValueError):
            raise ReflectionTokenizationError("invalid_ssh_tokenizer_response_no_count") from None
        if not isinstance(response, dict) or completed.returncode != 0 or response.get("ok") is not True:
            code = response.get("error_code") if isinstance(response, dict) else None
            if not isinstance(code, str) or not re.fullmatch(r"[a-z0-9_]{1,100}", code):
                code = "ssh_tokenizer_failed_no_count"
            raise ReflectionTokenizationError(code)
        record = response.get("result")
        expected = {"model_digest": self.model_digest, "model_blob_digest": MODEL_BLOB_DIGEST, "model": MODEL,
                    "context_length": CONTEXT_LENGTH, "method": METHOD, "prompt_sha256": prompt_hash,
                    "amd_port": self.amd_port, "enable_thinking": True, "add_generation_prompt": True,
                    "verified_before_and_after": True, "generation_calls": 0}
        if not isinstance(record, dict) or any(record.get(key) != value for key, value in expected.items()):
            raise ReflectionTokenizationError("returned_token_count_contract_mismatch")
        if (type(record.get("input_tokens")) is not int or record["input_tokens"] <= 0
                or any(type(record.get(key)) is not int or record[key] <= 0 for key in ("backend_pid", "service_pid"))
                or type(record.get("backend_port")) is not int or not 1 <= record["backend_port"] <= 65535):
            raise ReflectionTokenizationError("returned_token_count_is_unverified")
        for key in ("chat_template_sha256", "rendered_prompt_sha256"):
            if not isinstance(record.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", record[key]):
                raise ReflectionTokenizationError("returned_template_hash_is_unverified")
        # Whitelist metadata so even an unexpected remote field cannot leak text.
        fields = set(expected) | {"input_tokens", "backend_pid", "backend_port", "backend_start_time", "service_pid",
                                  "service_start_time", "chat_template_sha256", "rendered_prompt_sha256", "rendered_prompt_chars"}
        return {key: record[key] for key in sorted(fields) if key in record}
