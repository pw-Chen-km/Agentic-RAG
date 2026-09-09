#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jj/PW/Agentic-RAG
SAMPLE_DIR="$ROOT/data/evaluations/hotpotqa_sample100_seed42"
AGENTIC_CONFIG="$ROOT/configs/hotpotqa_qwen36_amd_nothink.yaml"
ARAG_CONFIG="$ROOT/A-RAG/configs/hotpotqa_qwen36_amd_nothink.yaml"
AGENTIC_OUT="$ROOT/runs/qwen36_amd_nothink_hotpotqa_sample100_agentic"
ARAG_OUT="$ROOT/A-RAG/results/qwen36_amd_nothink_hotpotqa_sample100_arag"
COMPARE_OUT="$ROOT/runs/comparisons/qwen36_amd_nothink_hotpotqa_sample100"
LOG_DIR="$ROOT/runs/qwen36_amd_nothink_hotpotqa_sample100_logs"
mkdir -p "$LOG_DIR" "$COMPARE_OUT"

curl -fsS --max-time 20 http://127.0.0.1:11435/api/tags | jq -e '.models[] | select(.name == "qwen3.6:35b-a3b-bf16")' >/dev/null
cd "$ROOT"
uv run python scripts/create_hotpotqa_sample100.py
jq -e '.embedding_model.name == "sentence-transformers/all-MiniLM-L6-v2" and .embedding_dimension == 384' artifacts/hotpotqa_benchmark_exact/manifest.json >/dev/null
test -f A-RAG/data/hotpotqa/index/sentence_index.pkl

SMOKE_QUESTION=$(jq -r '.[0].question' "$SAMPLE_DIR/questions.json")
if [ ! -e "$ROOT/runs/qwen36_amd_nothink_hotpotqa_smoke_agentic/smoke/episode.json" ]; then
  uv run agentic-rag run artifacts/hotpotqa_benchmark_exact "$SMOKE_QUESTION" \
    --scope-id hotpotqa:benchmark_exact:dev \
    --skill-file skills/skillopt_seed.md \
    --config "$AGENTIC_CONFIG" \
    --output runs/qwen36_amd_nothink_hotpotqa_smoke_agentic \
    --episode-id smoke >"$LOG_DIR/agentic_smoke.log" 2>&1
fi

cd "$ROOT/A-RAG"
if [ ! -s results/qwen36_amd_nothink_hotpotqa_smoke_arag/predictions.jsonl ]; then
  ARAG_API_KEY=ollama-local uv run python scripts/batch_runner.py \
    --config "$ARAG_CONFIG" \
    --questions "$SAMPLE_DIR/questions.json" \
    --output results/qwen36_amd_nothink_hotpotqa_smoke_arag \
    --limit 1 --workers 1 >"$LOG_DIR/arag_smoke.log" 2>&1
fi

cd "$ROOT"
AGENTIC_RESUME=()
if [ -f "$AGENTIC_OUT/progress.json" ]; then AGENTIC_RESUME=(--resume); fi
/usr/bin/time -f '%e' -o "$LOG_DIR/agentic_seconds.txt" \
  uv run python scripts/run_benchmark_eval.py \
    --substrate artifacts/hotpotqa_benchmark_exact \
    --split "$SAMPLE_DIR/questions.jsonl" \
    --config "$AGENTIC_CONFIG" \
    --skill skills/skillopt_seed.md \
    --output "$AGENTIC_OUT" \
    --expected-count 100 --dataset hotpotqa "${AGENTIC_RESUME[@]}" \
    >"$LOG_DIR/agentic_full.log" 2>&1

cd "$ROOT/A-RAG"
/usr/bin/time -f '%e' -o "$LOG_DIR/arag_seconds.txt" \
  env ARAG_API_KEY=ollama-local uv run python scripts/batch_runner.py \
    --config "$ARAG_CONFIG" \
    --questions "$SAMPLE_DIR/questions.json" \
    --output "$ARAG_OUT" --workers 1 \
    >"$LOG_DIR/arag_full.log" 2>&1

cd "$ROOT"
uv run python scripts/summarize_qwen36_sample100.py \
  --agentic-summary "$AGENTIC_OUT/summary.json" \
  --arag-predictions "$ARAG_OUT/predictions.jsonl" \
  --sample "$SAMPLE_DIR/questions.json" \
  --agentic-seconds "$LOG_DIR/agentic_seconds.txt" \
  --arag-seconds "$LOG_DIR/arag_seconds.txt" \
  --output "$COMPARE_OUT" >"$LOG_DIR/comparison.log" 2>&1

echo COMPLETE
