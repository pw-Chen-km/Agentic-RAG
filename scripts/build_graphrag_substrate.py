"""Build an isolated substrate for one GraphRAG-Benchmark dataset."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from agentic_rag.config import BuildConfig
from agentic_rag.substrate.builder import SubstrateBuilder

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--dataset", choices=("novel","medical"), required=True); p.add_argument("--root", type=Path, default=Path("data/graphrag_benchmark")); p.add_argument("--output", type=Path); p.add_argument("--config", type=Path); a=p.parse_args()
    root=a.root/a.dataset; output=a.output or root/"substrate"; config=a.config or Path(f"configs/graphrag_{a.dataset}_substrate.yaml")
    manifest=SubstrateBuilder(BuildConfig.from_yaml(config)).build(root, output)
    print(json.dumps({"dataset":a.dataset,"output":output.as_posix(),"records":manifest.record_counts}, ensure_ascii=False))
if __name__ == "__main__": main()
