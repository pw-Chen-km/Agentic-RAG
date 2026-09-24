#!/usr/bin/env bash
set -euo pipefail

REPO="/home/jj/PW/agenticRAG-native-v1-20260923-r2"
DATA="/home/jj/Large_Space_A/chiu/PW/Agentic-RAG-data/interface_study_v2"
OUT="$DATA/runs/native-v1-full-20260924-r2"
CONFIG="$REPO/configs/interface_study_qwen27b_ollama.yaml"
SKILL="$REPO/skills/interface_study.md"
PYTHON="/home/jj/PW/Agentic-RAG/.venv/bin/python"
export PYTHONPATH="$REPO/src"

test ! -e "$OUT"
mkdir -p "$OUT"

"$PYTHON" "$REPO/scripts/run_interface_study.py" \
  --dataset hotpotqa \
  --substrate "$DATA/substrates-native-v1/hotpotqa-r2" \
  --questions "$DATA/sources/hotpotqa/hotpot_dev_distractor_v1.selected.json" \
  --source-manifest "$DATA/sources/hotpotqa/source_manifest.json" \
  --config "$CONFIG" --skill "$SKILL" \
  --output "$OUT/hotpotqa" \
  --conditions C0 C1 C2 C3 C5 C4 A1 --seed 20260805

"$PYTHON" "$REPO/scripts/run_interface_study.py" \
  --dataset novel \
  --substrate "$DATA/substrates-native-v1/novel-r2" \
  --questions "$DATA/sources/graphrag_benchmark/novel/questions.json" \
  --source-manifest "$DATA/sources/graphrag_benchmark/novel/source_manifest.json" \
  --config "$CONFIG" --skill "$SKILL" \
  --output "$OUT/novel" \
  --conditions C0 C1 C2 C3 C5 C4 A1 --seed 20260805

"$PYTHON" "$REPO/scripts/run_interface_study.py" \
  --dataset medical \
  --substrate "$DATA/substrates-native-v1/medical-r2" \
  --questions "$DATA/sources/graphrag_benchmark/medical/questions.json" \
  --source-manifest "$DATA/sources/graphrag_benchmark/medical/source_manifest.json" \
  --config "$CONFIG" --skill "$SKILL" \
  --output "$OUT/medical" \
  --conditions C0 C1 C2 C3 C5 C4 A1 --seed 20260805
