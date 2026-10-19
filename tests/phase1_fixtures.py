"""Synthetic, deterministic examples. NOT experimental Qwen results."""

import copy
import json


def full(cid, text):
    return (
        "\n"
        + "=" * 80
        + f"\n[Chunk {cid}]\n"
        + "-" * 80
        + "\n"
        + text
        + "\n"
        + "=" * 80
    )


def raw_example(qid="q1"):
    author = "The River was written by Alice."
    birthplace = "Alice was born in York."
    selected = [
        [
            (
                "keyword_search",
                {"keywords": ["The River"]},
                "Chunk ID: 17, Matched keywords in chunk: ... "
                + author
                + " ...\n\nChunk ID: 31, Matched keywords in chunk: ... Alice biography ...",
            ),
            (
                "semantic_search",
                {"query": "The River author"},
                "Chunk ID: 17 (Similarity: 0.900)\nMatched: ... " + author + " ...",
            ),
        ],
        [
            (
                "read_chunk",
                {"chunk_ids": ["17", "31"]},
                full("17", author + " It was published in spring.")
                + full("31", birthplace),
            )
        ],
    ]
    messages = [
        {"role": "system", "content": "Fixed runtime prompt."},
        {"role": "user", "content": "Where was the author of The River born?"},
    ]
    raw = {
        "question_id": qid,
        "question": messages[-1]["content"],
        "answer": "York",
        "trajectory": [],
        "decision_audit": [],
        "usage": {
            "complete": True,
            "input_tokens": 400,
            "output_tokens": 100,
            "total_tokens": 500,
        },
        "policy_calls": 3,
        "total_retrieved_tokens": 60,
        "termination_reason": "model_answered",
        "execution_errors": [],
    }
    for n, group in enumerate(selected, 1):
        calls = [
            {
                "id": f"call{n}-{i}",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(args)},
            }
            for i, (tool, args, _) in enumerate(group)
        ]
        response = {"role": "assistant", "content": None, "tool_calls": calls}
        raw["decision_audit"].append(
            {
                "request": {"messages": copy.deepcopy(messages)},
                "response": {
                    "message": response,
                    "input_tokens": 100,
                    "output_tokens": 25,
                },
            }
        )
        messages.append(response)
        for call, (tool, args, body) in zip(calls, group):
            raw["trajectory"].append(
                {
                    "loop": n,
                    "tool_name": tool,
                    "arguments": args,
                    "tool_result": body,
                    "retrieved_tokens": 20,
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": body}
            )
    raw["decision_audit"].append(
        {
            "request": {"messages": copy.deepcopy(messages)},
            "response": {
                "message": {"role": "assistant", "content": "York"},
                "input_tokens": 200,
                "output_tokens": 50,
            },
        }
    )
    return raw


def source_example():
    return [
        {
            "_id": "q1",
            "question": raw_example()["question"],
            "answer": "York",
            "context": [
                ["The River", ["The River was written by Alice."]],
                ["Alice", ["Alice was born in York."]],
            ],
            "supporting_facts": [["The River", 0], ["Alice", 0]],
        }
    ]


def alternative_example(qid="q1"):
    """Synthetic alternative: omit the redundant simultaneous semantic search."""
    raw = raw_example(qid)
    raw["trajectory"].pop(1)
    for decision in raw["decision_audit"]:
        messages = decision["request"]["messages"]
        messages[:] = [m for m in messages if m.get("tool_call_id") != "call1-1"]
        for message in messages + [decision["response"]["message"]]:
            if "tool_calls" in message:
                message["tool_calls"] = [
                    call for call in message["tool_calls"] if call["id"] != "call1-1"
                ]
    raw["usage"].update(input_tokens=300, output_tokens=80, total_tokens=380)
    for decision, (inputs, outputs) in zip(
        raw["decision_audit"], ((80, 20), (80, 20), (140, 40))
    ):
        decision["response"].update(input_tokens=inputs, output_tokens=outputs)
    raw["total_retrieved_tokens"] = 40
    return raw


def chunks_example():
    return [
        "17:The River was written by Alice. It was published in spring.",
        "31:Alice was born in York.",
    ]
