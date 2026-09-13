"""Installed CLI for portable staged SkillOpt training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .config import WorkflowConfig
from .runner import WorkflowRunner, digest, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Workflow-Aware SkillOpt: Retrieval -> Meta -> Answer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without contacting a model")
    parser.add_argument("--demo", action="store_true", help="Use deterministic fake models, NOT experiment results")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    base = args.config.resolve().parent
    resolve = lambda value: (base / value).resolve()
    train_path, val_path = resolve(config["train"]), resolve(config["validation"])
    test_path = resolve(config["test"]) if config.get("test") else None
    train = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
    val = [json.loads(line) for line in val_path.read_text().splitlines() if line.strip()]
    test = ([json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
            if test_path else [])
    for rows in (train, val, test):
        if any(not all(key in r for key in ("id", "question", "answer", "scope_id")) for r in rows):
            raise ValueError("each question needs id, question, answer, scope_id")
    seed = resolve(config["skill"]).read_text()
    workflow = WorkflowConfig.from_mapping(config.get("workflow", {}))
    source_dir = Path(__file__).parent
    # Include all runtime source files, not just the new runner, in lineage.
    source_files = sorted(source_dir.parents[1].rglob("*.py"))
    source_files += sorted(source_dir.rglob("*.md"))
    contract = {"demo": args.demo, "settings": config,
                "source_hash": digest({str(p.relative_to(source_dir.parents[1])):
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})}
    backend = None
    if args.demo:
        from .demo import DemoBackend
        backend = DemoBackend()
    else:
        from agentic_rag.agent.config import AgentConfig, OllamaPolicyConfig
        from agentic_rag.substrate.storage import Substrate
        agent_path = resolve(config["agent_config"])
        agent = AgentConfig.from_yaml(agent_path)
        optimizer = OllamaPolicyConfig.model_validate(config["optimizer"])
        judge = OllamaPolicyConfig.model_validate(config["judge"])
        substrate = resolve(config["substrate"])
        Substrate.open(substrate)
        contract["agent"] = agent.effective_dict()
        # Hash every existing index/corpus artifact; never rebuild embeddings.
        def file_hash(path):
            h = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1024*1024), b""):
                    h.update(block)
            return h.hexdigest()
        contract["substrate"] = {str(p.relative_to(substrate)): file_hash(p)
            for p in sorted(substrate.rglob("*")) if p.is_file()}
        if not args.dry_run:
            from ollama import Client
            from .runtime import OllamaBackend
            inventories = {}
            for settings in (agent.policy, optimizer, judge):
                if not isinstance(settings, OllamaPolicyConfig):
                    raise ValueError("workflow runner currently requires Ollama target/optimizer/judge")
                models = Client(host=settings.host, timeout=settings.timeout_seconds).list().models
                found = next((m for m in models if m.model == settings.model), None)
                if found is None:
                    raise ValueError(f"model {settings.model} is not installed at {settings.host}")
                inventories[f"{settings.host}/{settings.model}"] = found.digest
            contract["model_digests"] = inventories
            backend = OllamaBackend(substrate=substrate, agent=agent, optimizer=optimizer,
                                    judge=judge, dataset=config["dataset"])
    runner = WorkflowRunner(backend=backend, output=resolve(config["output"]), config=workflow,
        contract=contract, train=train, validation=val, test=test, skill=seed)
    if args.dry_run:
        print(json.dumps({"status": "validated", "train": len(train), "validation": len(val),
                          "test": len(test),
                          "model_calls": 0, "demo": args.demo, "contract": runner.contract}, indent=2))
    else:
        # One writer per output. OS releases the lock on process termination.
        import fcntl
        runner.output.mkdir(parents=True, exist_ok=True)
        with (runner.output / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(json.dumps(runner.run(), indent=2))


if __name__ == "__main__":
    main()
