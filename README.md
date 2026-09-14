# Agentic RAG Interface Study

這個工作區只保留 Agentic RAG 介面研究需要的 runtime、Ollama provider、global
HotpotQA provenance substrate、evaluation 與 pilot scripts。原本的
`agenticRAG_研究` 沒有被修改。

## Data lineage

`data/interface_study/source_manifest.json` 固定 HotpotQA distractor dev 檔案的
來源、版本、授權與 SHA-256。`prepare_interface_study.py` 會先把官方資料與本機
1,000 題逐筆比對，再以 seed `20260805` 抽出 24 bridge 與 6 comparison；比對失敗
會直接停止。

## Build and run

先建立環境並安裝本專案（Python 3.12）：

```powershell
pip install -e ".[dev]"
```

準備資料與 provenance：

```powershell
python scripts/prepare_interface_study.py `
  --official data/hotpotqa/hotpot_dev_distractor_v1.json `
  --local-questions C:\Users\Patrick\Desktop\agenticRAG_研究\data\arag_hotpotqa\hotpotqa\questions.json `
  --output data/interface_study
python scripts/build_interface_substrate.py
```

Pilot 使用本機 Ollama 的 `qwen3.5:4b`、temperature 0、`think=false`、
`num_ctx=32768`、每次最多 2,048 output tokens、每題 15 次正常 policy decisions
與最多一次 FINISH-only finalize。執行五個條件（C0–C4，共 150 episodes）：

```powershell
python scripts/run_interface_study.py `
  --substrate data/interface_study/substrate `
  --questions data/interface_study/pilot_questions.json `
  --source-manifest data/interface_study/source_manifest.json
python scripts/analyze_interface_study.py --run runs/interface-study-v1
```

A1 是 annotation-only control，可額外以 `--conditions C0 C1 C2 C3 C4 A1` 執行。
每個 episode 的 `episode.json` 保存實際 messages、schema digest、visible source
spans、exposed refs、provider token metadata 與錯誤原因；gold answer/support 只在
evaluation sidecar，不會進入 Policy context。

## Acceptance criteria

所有正常決策（valid、invalid、duplicate、empty）都計入 15 次上限；finalize 不得
搜尋、擴展或讀取新資料。resume 會拒絕 source/config/substrate/renderer hash 改變。
Pilot 的主要輸出是流程、provenance、instrumentation 與成本的可重現性；4B 結果不
直接作為正式研究結論。

## GraphRAG-Benchmark semantic evaluation

下載並固定官方 Novel、Medical corpus/questions，產生 source manifest 與固定大小的
normalized chunks：

```powershell
python scripts/fetch_graphrag_benchmark.py --download
python scripts/build_graphrag_substrate.py --dataset novel
python scripts/build_graphrag_substrate.py --dataset medical
```

兩個資料集各自建立 substrate。Semantic judge 使用獨立 Ollama `qwen3.5:4b` 與
`nomic-embed-text`，依官方題型計算 ROUGE-L、Answer Correctness、Coverage、
Faithfulness、Context Relevancy、Evidence Recall。它讀取已完成的 episode，不會改變
target Policy input：

```powershell
python scripts/evaluate_semantics.py `
  --episodes runs/interface-study-v1/episodes `
  --substrate data/interface_study/substrate `
  --output runs/interface-study-v1/semantic_evaluations `
  --model qwen3.5:4b `
  --embedding-model nomic-embed-text
```

Novel、Medical、HotpotQA 的 aggregate 分開保存；所有 judge prompt、parsed verdict、
provider usage、耗時與 unavailable/not_evaluable 原因都保留在 semantic evaluation
artifact。GraphRAG-Benchmark 的 indexing structural metrics 不在本研究的主要輸出中。

要對 GraphRAG-Benchmark 全量執行六個介面條件，可使用：

```powershell
python scripts/run_interface_study.py `
  --dataset novel `
  --substrate data/graphrag_benchmark/novel/substrate `
  --questions data/graphrag_benchmark/novel/questions.json `
  --source-manifest data/graphrag_benchmark/novel/source_manifest.json `
  --judge-config configs/semantic_judge_qwen.yaml `
  --semantic-eval `
  --conditions C0 C1 C2 C3 C4 A1 `
  --output runs/graphrag-novel-v1
```

Medical 只需把 dataset、substrate、questions、source manifest 與 output 換成
`medical` 對應路徑。若 target episodes 已完成，可用 `evaluate_semantics.py` 只補做
judge，避免重跑 Agent。
