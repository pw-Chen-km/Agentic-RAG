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
python scripts/run_interface_study.py
python scripts/analyze_interface_study.py
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
