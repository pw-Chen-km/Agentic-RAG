import json
from pathlib import Path

ids = [
    "5ae608c15542992663a4f246", "5ae72ab25542991e8301cb77",
    "5a76c98855429972597f13d0", "5abfaa445542990832d3a18c",
    "5ae0ce4e554299603e41845f", "5abd6be55542993062266c89",
    "5a8835dc5542994846c1ce2b", "5a90b1635542990a984936a9",
    "5adc1ae5554299438c868d5c", "5ae618325542996de7b71b56",
    "5abc7bf5554299114383a129", "5ae2120c5542997283cd23ac",
    "5ac15c1c5542994ab5c67cf7", "5a8a410655429970aeb70279",
    "5a8b987f5542997f31a41d7a", "5a7fb7985542994857a767d2",
    "5a77b6535542995d83181274", "5a7b24fe55429931da12c9f7",
    "5a7c8a3b55429935c91b5204", "5adfc77b554299603e4183ab",
]

source = json.loads(
    Path("data/rag_test/hotpotqa/questions.json").read_text(encoding="utf-8")
)
by_id = {str(row["id"]): row for row in source}
out = Path("data/evaluations/hotpotqa_resample20_seed20260805")
out.mkdir(parents=True, exist_ok=True)
records = [
    {
        "answer": by_id[qid]["answer"],
        "id": qid,
        "question": by_id[qid]["question"],
        "question_type": by_id[qid]["question_type"],
        "scope_id": "hotpotqa:benchmark_exact:dev",
        "source": "hotpotqa",
    }
    for qid in ids
]
payload = "".join(
    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records
)
(out / "questions.jsonl").write_text(payload, encoding="utf-8")
(out / "questions.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
