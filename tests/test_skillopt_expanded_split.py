from __future__ import annotations

import json
from pathlib import Path

from agentic_rag.skillopt.expanded_split import prepare_expanded_hotpotqa_splits


def test_expanded_split_preserves_base_and_excludes_heldout(tmp_path: Path) -> None:
    questions = [
        {
            "id": f"q{index:02d}",
            "question": f"Question {index}?",
            "answer": f"Answer {index}",
            "question_type": "comparison" if index % 5 == 4 else "bridge",
            "scope_id": "hotpotqa:benchmark_exact:dev",
            "source": "hotpotqa",
        }
        for index in range(30)
    ]
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps(questions), encoding="utf-8")

    base = tmp_path / "base"
    base.mkdir()
    base_ids = {"train": "q00", "validation": "q04", "test": "q01"}
    for split, question_id in base_ids.items():
        row = next(item for item in questions if item["id"] == question_id)
        (base / f"{split}.jsonl").write_text(
            json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
        )
    (base / "split_manifest.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "subset": "hotpotqa",
                    "scope_id": "hotpotqa:benchmark_exact:dev",
                }
            }
        ),
        encoding="utf-8",
    )

    excluded = tmp_path / "heldout.json"
    excluded.write_text(json.dumps([questions[2], questions[9]]), encoding="utf-8")
    output = tmp_path / "expanded"
    manifest = prepare_expanded_hotpotqa_splits(
        questions_path=questions_path,
        base_split_dir=base,
        excluded_questions_path=excluded,
        output_dir=output,
        train_size=4,
        validation_size=4,
        test_size=4,
        seed=42,
    )

    split_ids: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in (output / f"{split}.jsonl").read_text().splitlines()
        ]
        split_ids[split] = {str(row["id"]) for row in rows}
        assert len(rows) == 4
        assert base_ids[split] in split_ids[split]
        assert manifest["splits"][split]["question_type_counts"] == {
            "bridge": 3,
            "comparison": 1,
        }

    assert not (split_ids["train"] & split_ids["validation"])
    assert not (split_ids["train"] & split_ids["test"])
    assert not (split_ids["validation"] & split_ids["test"])
    assert not ({"q02", "q09"} & set().union(*split_ids.values()))
    assert manifest["selection"]["excluded_questions"]["count"] == 2
