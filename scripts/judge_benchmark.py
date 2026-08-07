"""Judge persisted benchmark predictions with the Luna judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_rag.evaluation import (
    EpisodeEvaluator,
    JudgeUsage,
    OpenAIResponsesJudge,
)
from agentic_rag.evaluation.profiles import get_dataset_profile


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=SUMMARY_JSON",
        help="Label and path to one persisted run summary; repeat as needed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--dataset", default="hotpotqa")
    return parser.parse_args()


def _parse_runs(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label.strip() or not raw_path.strip():
            raise ValueError(f"invalid --run value: {value!r}")
        normalized_label = label.strip()
        if normalized_label in labels:
            raise ValueError(f"duplicate run label: {normalized_label}")
        labels.add(normalized_label)
        parsed.append((normalized_label, Path(raw_path.strip())))
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(payload + "\n", encoding="utf-8")


def _add_usage(total: JudgeUsage, current: JudgeUsage) -> JudgeUsage:
    return total + current


def _judge_run(
    *,
    label: str,
    summary_path: Path,
    evaluator: EpisodeEvaluator,
) -> tuple[dict[str, Any], JudgeUsage]:
    source = _load_json(summary_path)
    contract = source.get("run_contract")
    declared_dataset = (
        contract.get("dataset") if isinstance(contract, dict) else None
    )
    if (
        declared_dataset is not None
        and declared_dataset != evaluator.profile.key
    ):
        raise ValueError(
            f"summary dataset is {declared_dataset!r}; expected "
            f"{evaluator.profile.key!r}: {summary_path}"
        )
    source_results = source.get("results")
    if not isinstance(source_results, list):
        raise ValueError(f"summary has no results list: {summary_path}")

    items: list[dict[str, Any]] = []
    usage = JudgeUsage()
    for row in source_results:
        if not isinstance(row, dict):
            raise ValueError(f"summary result is not an object: {summary_path}")
        evaluation = evaluator.evaluate(
            question=str(row["question"]),
            predicted_answer=str(row.get("predicted_answer") or ""),
            gold_answer=str(row["gold_answer"]),
        )
        usage = _add_usage(usage, evaluation.judge_usage)
        items.append(
            {
                "id": str(row["id"]),
                "question": evaluation.question,
                "gold_answer": evaluation.gold_answer,
                "predicted_answer": evaluation.predicted_answer,
                "verdict": "CORRECT" if evaluation.llm_acc else "INCORRECT",
                "llm_acc": evaluation.llm_acc,
                "contain_acc": evaluation.contain_acc,
                "normalized_exact": int(
                    evaluation.normalized_prediction == evaluation.normalized_gold
                ),
                "judge_status": evaluation.status,
                "judge_model": evaluation.judge_model,
                "judge_messages": [
                    message.model_dump(mode="json")
                    for message in evaluation.judge_messages
                ],
                "raw_judge_output": evaluation.raw_judge_output,
                "judge_usage": evaluation.judge_usage.model_dump(mode="json"),
                "judge_request_options": evaluation.judge_request_options,
            }
        )

    return (
        {
            "label": label,
            "source_summary": summary_path.as_posix(),
            "question_count": len(items),
            "luna_correct": sum(item["llm_acc"] for item in items),
            "contain_correct": sum(item["contain_acc"] for item in items),
            "normalized_exact_correct": sum(
                item["normalized_exact"] for item in items
            ),
            "blank_auto_incorrect": sum(
                item["judge_status"] == "failed" for item in items
            ),
            "judge_usage": usage.model_dump(mode="json"),
            "items": items,
        },
        usage,
    )


def main() -> None:
    args = _arguments()
    profile = get_dataset_profile(args.dataset)
    runs = _parse_runs(args.run)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)

    evaluator = EpisodeEvaluator(
        OpenAIResponsesJudge(model=str(args.model)),
        profile=profile,
    )
    results: list[dict[str, Any]] = []
    total_usage = JudgeUsage()
    for index, (label, summary_path) in enumerate(runs, start=1):
        print(f"START {index}/{len(runs)} {label}", flush=True)
        judged, usage = _judge_run(
            label=label,
            summary_path=summary_path,
            evaluator=evaluator,
        )
        results.append(judged)
        total_usage = _add_usage(total_usage, usage)
        _write_json(args.output / f"{label}.json", judged)
        print(
            f"DONE {label} correct={judged['luna_correct']}/"
            f"{judged['question_count']} judge_calls={usage.calls}",
            flush=True,
        )

    aggregate = {
        "judge_contract": {
            "model": str(args.model),
            "provider": "openai_responses",
            "profile": evaluator.profile.key,
            "comparison_input": "generated_answer_and_gold_answer",
            "blank_answers": "automatic_incorrect_without_provider_call",
            "supporting_evidence_visible_to_judge": False,
        },
        "run_count": len(results),
        "judge_usage": total_usage.model_dump(mode="json"),
        "results": results,
    }
    _write_json(args.output / "summary.json", aggregate)
    print(
        f"SUMMARY runs={len(results)} judge_calls={total_usage.calls} "
        f"total_tokens={total_usage.total_tokens}",
        flush=True,
    )


if __name__ == "__main__":
    main()
