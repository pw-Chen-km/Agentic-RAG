"""Independent original-prompt baseline; never edits the frozen training/test runner."""

import argparse
import copy
import fcntl
import hashlib
import json
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import requests
import yaml
from run_phase1 import file_hash
from arag.agent.base import BaseAgent
from arag_skillopt.checkpoint import load_checkpoint, save_checkpoint
from arag_skillopt.final_test import FinalTestRunner
from arag_skillopt.phase1_runtime import AnswerJudge, client, load_questions, make_tools
from arag_skillopt.runtime_audit import AuditClient, attach_audit
from arag_skillopt.support import digest
from arag_skillopt.trajectory import build_view


class OriginalRollout:
    """Same audited A-RAG BaseAgent, with the unmodified official prompt."""

    def __init__(self, config, tools, prompt):
        self.config, self.tools, self.prompt = config, tools, prompt
        self.llm = client(config["llm"])

    def __call__(self, question):
        if set(question) != {"id", "question"}:
            raise ValueError("only ID/question may enter original Agent")
        audited = AuditClient(self.llm)
        agent = BaseAgent(
            llm_client=audited, tools=self.tools, system_prompt=self.prompt,
            max_loops=self.config["agent"]["max_loops"],
            max_token_budget=self.config["agent"]["token_budget"],
        )
        start = time.monotonic()
        try:
            result = agent.run(question["question"])
        except Exception as exc:
            result = {"answer": "", "trajectory": [], "error": str(exc)}
        result = attach_audit(result, audited)
        result.update(question_id=question["id"], question=question["question"],
                      elapsed_seconds=time.monotonic() - start)
        return result


class OriginalTestRunner(FinalTestRunner):
    def __init__(self, output, rollout, evaluator, runtime, prompt, workers=2,
                 support_map=None):
        super().__init__(output, rollout, evaluator, runtime, workers, support_map)
        self.prompt = prompt
        self.prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

    def episode(self, question):
        qid = str(question["id"])
        folder = self.root / "episodes" / self.prompt_hash / digest(qid)[:20]
        runtime_question = {"id": qid, "question": question["question"]}
        raw = self.cached(folder / "raw.json", runtime_question,
                          lambda: self.rollout(copy.deepcopy(runtime_question)))
        if raw.get("question_id") != qid or raw.get("question") != question["question"]:
            raise ValueError("original test question mismatch")

        def evaluate():
            if raw.get("error") or raw.get("execution_errors"):
                return {"correct": None, "status": "runtime_error"}
            result = self.evaluator(question, raw)
            if result.get("status") != "ok" or type(result.get("correct")) is not bool:
                return {**result, "correct": None, "status": "unavailable"}
            return result

        evaluation = self.cached(folder / "evaluation.json",
                                  {"question": question, "raw_hash": digest(raw)}, evaluate)
        support = self.support_map["questions"].get(qid)
        if support and support["question"] != question["question"]:
            raise ValueError("supporting-fact question mismatch")
        view, audit = build_view(raw, evaluation, support, "original final test")
        save_checkpoint(folder / "view.json", view)
        save_checkpoint(folder / "lineage.json", {"split": "test", **audit})
        return {"question_id": qid, "raw": raw, "evaluation": evaluation, "view": view}

    def run(self, questions, training_ids, validation_ids):
        ids = [str(q["id"]) for q in questions]
        if not ids or len(ids) != len(set(ids)) or set(ids) & (set(training_ids) | set(validation_ids)):
            raise ValueError("test IDs must be unique and disjoint")
        contract = {
            "purpose": "post_training_original_prompt_test_only", "variant": "original",
            "test_feedback_to_optimizer": False, "test_hash": digest(questions),
            "original_prompt_sha256": self.prompt_hash, "workers": self.workers,
            "support_map_hash": digest(self.support_map), "runtime": self.runtime,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".original_test.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            saved = load_checkpoint(self.root / "contract.json")
            if saved and saved != contract:
                raise ValueError("original test contract changed; use NEW output")
            if not saved and any(x.name != ".original_test.lock" for x in self.root.iterdir()):
                raise ValueError("nonempty original output without contract")
            save_checkpoint(self.root / "contract.json", contract)
            (self.root / "original_prompt.txt").write_text(self.prompt)
            save_checkpoint(self.root / "checkpoint.json", {"status": "running", "arm": "original"})
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                episodes = list(pool.map(self.episode, questions))
            summary = {"status": "complete", "variant": "original", "question_count": len(ids),
                       "original": self.arm_summary(episodes), "test_feedback_to_optimizer": False}
            save_checkpoint(self.root / "summary.json", summary)
            save_checkpoint(self.root / "checkpoint.json", {"status": "complete", "arm": "original"})
            return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--paired-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    expected_host = cfg.get("deployment", {}).get("expected_hostname")
    if expected_host and socket.gethostname() != expected_host:
        raise ValueError("wrong deployment hostname")
    training, paired, output = [Path(x).resolve() for x in (args.training_output, args.paired_output, args.output)]
    if output in (training, paired) or any(base in output.parents for base in (training, paired)):
        raise ValueError("original baseline needs independent output")
    training_contract = load_checkpoint(training / "contract.json")
    training_summary = load_checkpoint(training / "summary.json")
    tc = load_checkpoint(training / "checkpoint.json")
    paired_contract = load_checkpoint(paired / "contract.json")
    if not training_summary or training_summary.get("status") != "complete" or not tc or tc.get("stage") != "complete":
        raise ValueError("training must complete before baseline")
    runtime = training_contract["runtime"]
    if cfg != runtime["settings"] or paired_contract["runtime"]["settings"] != cfg:
        raise ValueError("all arms must share identical settings")
    for key, expected in runtime["data_hashes"].items():
        path = Path(cfg["data"]["index_dir"]) / "sentence_index.pkl" if key == "index" else cfg["data"][key]
        if file_hash(path) != expected:
            raise ValueError("data changed: " + key)
    for name, expected in runtime["source_hashes"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("frozen source changed: " + name)
    train, val, test = [load_questions(cfg["data"][k]) for k in ("train", "validation", "test")]
    ids = [{q["id"] for q in rows} for rows in (train, val, test)]
    if len(test) != 600 or any(len(i) != len(r) for i, r in zip(ids, (train, val, test))) or any(ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("requires identical disjoint test600")
    if digest(test) != paired_contract["test_hash"] or digest(train) != training_contract["train_hash"] or digest(val) != training_contract["validation_hash"]:
        raise ValueError("split hash mismatch")
    original = (ROOT / "src/arag/agent/prompts/default.txt").read_text()
    print(json.dumps({"preflight": "passed", "variant": "original", "questions": len(test),
                      "formatted_skill_injected": False, "execute": args.execute}), flush=True)
    if not args.execute:
        return
    ps, pc = [load_checkpoint(paired / n) for n in ("summary.json", "checkpoint.json")]
    if not ps or ps.get("status") != "complete" or not pc or pc.get("status") != "complete":
        raise ValueError("wait for the existing paired test to finish")
    for name in ("llm", "judge"):
        for endpoint, expected in runtime["model_digests"][name].items():
            response = requests.get(endpoint + "/api/tags", timeout=15)
            response.raise_for_status()
            matches = [m for m in response.json()["models"] if m["name"] == cfg[name]["model"]]
            if len(matches) != 1 or matches[0]["digest"] != expected:
                raise ValueError("model digest mismatch")
    recorded = {
        "training_contract_hash": digest(training_contract), "paired_contract_hash": digest(paired_contract),
        "model_digests": runtime["model_digests"], "data_hashes": runtime["data_hashes"],
        "source_hashes": runtime["source_hashes"], "settings": cfg,
        "extension_sha256": file_hash(__file__),
    }
    runner = OriginalTestRunner(output, OriginalRollout(cfg, make_tools(cfg), original),
                                AnswerJudge(cfg["judge"]), recorded, original,
                                cfg["phase1"]["episode_workers"],
                                json.loads(Path(cfg["data"]["support_map"]).read_text()))
    print(json.dumps(runner.run(test, ids[0], ids[1]), indent=2), flush=True)


if __name__ == "__main__":
    main()
