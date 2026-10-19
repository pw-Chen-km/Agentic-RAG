"""Run the unchanged dense tool in a bounded native-library child process.

The public schema/text and index/model are unchanged. Native crashes are logged,
retried once in a fresh child, then surfaced as errors, never empty successes.
"""
import atexit
import copy
import json
import os
import selectors
import subprocess
import sys
import threading
import time
from pathlib import Path

from arag.tools.base import BaseTool


class SemanticProcessTool(BaseTool):
    def __init__(self, chunks_file, index_dir='index', model_name='sentence-transformers/all-MiniLM-L6-v2',
                 device=None, query_prompt_name=None, *, _worker_command=None, timeout=120):
        self.spec = dict(chunks_file=chunks_file, index_dir=index_dir, model_name=model_name,
                         device=device, query_prompt_name=query_prompt_name)
        self.command = _worker_command or [sys.executable, '-X', 'faulthandler', '-u', '-m',
                                           'arag.tools.semantic_process', '--worker', json.dumps(self.spec)]
        self.timeout, self.lock, self.process, self.schema = timeout, threading.Lock(), None, None
        self.failures = []
        self._start()
        atexit.register(self.close)

    @property
    def name(self): return 'semantic_search'

    def get_schema(self): return copy.deepcopy(self.schema)

    def _receive(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout): raise TimeoutError('semantic worker timed out')
        line = self.process.stdout.readline()
        if not line:
            code = self.process.wait(timeout=5)
            raise ChildProcessError(f'semantic worker exited {code}')
        result = json.loads(line)
        if 'error' in result: raise RuntimeError(result['error'])
        return result

    def _start(self):
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   TOKENIZERS_PARALLELISM='false', PYTHONFAULTHANDLER='1')
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get('PYTHONPATH', '')
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1, env=env)
        try: ready = self._receive()['schema']
        except BaseException:
            self.close()
            raise
        if self.schema is not None and ready != self.schema:
            self.close(); raise RuntimeError('restarted semantic worker changed schema')
        self.schema = ready

    def close(self):
        p = self.process
        if p is not None:
            if p.poll() is None:
                p.terminate()
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired: p.kill(); p.wait(timeout=5)
            for stream in (p.stdin, p.stdout):
                if stream: stream.close()
            self.process = None

    def execute(self, context, query, top_k=5):
        with self.lock:
            for attempt in (1, 2):
                try:
                    self.process.stdin.write(json.dumps({'query': query, 'top_k': top_k}) + '\n')
                    self.process.stdin.flush()
                    result = self._receive()
                    text, metadata = result['text'], result['metadata']
                    context.add_retrieval_log(tool_name=self.name, tokens=metadata['retrieved_tokens'],
                                              metadata={'query': query, 'chunks_found': metadata['chunks_found']})
                    return text, metadata
                except (ChildProcessError, BrokenPipeError, TimeoutError) as exc:
                    event = {'event': 'semantic_worker_failure', 'attempt': attempt,
                             'error': str(exc), 'time': time.time()}
                    self.failures.append(event)
                    print(json.dumps(event), file=sys.stderr, flush=True)
                    self.close()
                    if attempt == 2: raise RuntimeError('semantic worker failed twice; no result fabricated') from exc
                    self._start()


def worker(spec):
    # Native torch/numpy/tokenizer imports and state stay outside the coordinator.
    from arag.tools.semantic_search import SemanticSearchTool
    from arag.core.context import AgentContext
    tool = SemanticSearchTool(**spec)
    print(json.dumps({'schema': tool.get_schema()}), flush=True)
    for line in sys.stdin:
        try:
            query = json.loads(line)
            text, metadata = tool.execute(AgentContext(), **query)
            print(json.dumps({'text': text, 'metadata': metadata}), flush=True)
        except Exception as exc:
            print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}), flush=True)


if __name__ == '__main__' and sys.argv[1] == '--worker': worker(json.loads(sys.argv[2]))
