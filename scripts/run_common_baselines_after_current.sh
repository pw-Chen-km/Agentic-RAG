#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jj/PW/Agentic-RAG
BASE=/home/jj/Large_Space_A/chiu/PW/Agentic-RAG-data/hotpotqa_provenance/runs/skillopt_qwen36_full_agent_policy_v1_bestskills_common_heldout100
PILOT=/home/jj/Large_Space_A/chiu/PW/Agentic-RAG-data/hotpotqa_provenance/runs/skillopt_qwen36_full_agent_policy_v1_100_50_50_parallel4
SAMPLE_DIR="$ROOT/data/evaluations/hotpotqa_sample100_seed42"
SUBSTRATE=/home/jj/Large_Space_A/chiu/PW/Agentic-RAG-data/hotpotqa_provenance/substrate
CONFIG="$ROOT/configs/hotpotqa_qwen36_amd_nothink.yaml"
INITIAL_OUT="$BASE/initial"
ARAG_OUT="$BASE/a_rag"

while pgrep -f "run_skillopt_heldout100_comparison.py.*$BASE" >/dev/null; do
  sleep 20
done

cd "$ROOT"
initial_resume=()
if [ -f "$INITIAL_OUT/progress.json" ]; then
  initial_resume=(--resume)
fi
if [ ! -f "$INITIAL_OUT/summary.json" ]; then
  /usr/bin/time -f '%e' -o "$BASE/timing/initial_seconds.txt" \
    .venv/bin/python scripts/run_benchmark_eval.py \
      --substrate "$SUBSTRATE" \
      --split "$SAMPLE_DIR/questions.jsonl" \
      --config "$CONFIG" \
      --skill "$PILOT/raw/skills/skill_v0000.md" \
      --output "$INITIAL_OUT" \
      --expected-count 100 \
      --dataset hotpotqa "${initial_resume[@]}" \
      >> "$BASE/logs/initial.log" 2>&1
fi

cd "$ROOT/A-RAG"
/usr/bin/time -f '%e' -o "$BASE/timing/a_rag_seconds.txt" \
  env ARAG_API_KEY=ollama-local .venv/bin/python scripts/batch_runner.py \
    --config configs/hotpotqa_qwen36_amd_nothink.yaml \
    --questions "$SAMPLE_DIR/questions.json" \
    --output "$ARAG_OUT" \
    --workers 1 \
    >> "$BASE/logs/a_rag.log" 2>&1

cd "$ROOT"
.venv/bin/python scripts/evaluate_arag_heldout.py \
  --predictions "$ARAG_OUT/predictions.jsonl" \
  --sample "$SAMPLE_DIR/questions.json" \
  --config "$CONFIG" \
  --output "$ARAG_OUT/summary.json" \
  --elapsed-seconds-file "$BASE/timing/a_rag_seconds.txt" \
  >> "$BASE/logs/a_rag_judge.log" 2>&1

.venv/bin/python scripts/build_extended_heldout_comparison.py \
  --root "$BASE" \
  --initial-summary "$INITIAL_OUT/summary.json" \
  --optimized-summary "$BASE/summary.json" \
  --arag-summary "$ARAG_OUT/summary.json" \
  >> "$BASE/logs/extended_comparison.log" 2>&1

echo COMPLETE
