#!/usr/bin/env python3
"""Build a LinearRAG-style corpus graph from existing offline indices.

The command only reads chunks, the existing sentence embedding index and the
existing corpus-only entity index.  It never accepts questions, answers or
gold evidence, and it never downloads an embedding model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arag.tools.linear_graph_index import build_linear_graph, write_linear_graph  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--sentence-index", required=True)
    parser.add_argument("--entity-index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    graph = build_linear_graph(args.chunks, args.sentence_index, args.entity_index)
    write_linear_graph(graph, args.output)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "version": graph["version"],
        "graph_sha256": graph["graph_sha256"],
        "source": graph["source"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
