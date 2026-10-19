"""Build evaluator-only exact supporting-fact spans. Does not touch the index."""

import argparse
import json
from collections import Counter
from pathlib import Path

from arag_skillopt.checkpoint import save_checkpoint
from arag_skillopt.support import build_support_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise SystemExit("Output exists; use a new path to preserve prior mappings")
    mapping = build_support_map(
        json.loads(Path(args.source).read_text()),
        json.loads(Path(args.chunks).read_text()),
    )
    save_checkpoint(target, mapping)
    stats = Counter(
        f["status"] for q in mapping["questions"].values() for f in q["facts"]
    )
    report = {
        "questions": len(mapping["questions"]),
        "evaluable_questions": sum(
            q["evaluable"] for q in mapping["questions"].values()
        ),
        "fact_status": dict(stats),
        "unevaluable_question_ids": [
            i for i, q in mapping["questions"].items() if not q["evaluable"]
        ],
    }
    save_checkpoint(target.with_suffix(".report.json"), report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "unevaluable_question_ids"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
