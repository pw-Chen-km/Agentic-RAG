# Agentic RAG Interface Study

這個工作區是 Agentic RAG retrieval-interface study 的 v2 實驗工作區。
`runs/interface-study-v1` 是保留的舊 pilot；v2 使用新的 manifest、neutral skill、
schema-constrained single-decision protocol 與 observation renderer，不覆寫舊結果。原本的
`agenticRAG_研究` 沒有被修改。

目前的正式候選 protocol 為 `constrained-single-decision-v4`。模型每輪回傳一個符合
JSON Schema 的 decision，其中只有一個 action；assessment 開啟時，`supported_facts`
與 `missing_information` 是同一 decision 的頂層陣列。這避免 provider 回傳多個 tool calls，
也避免 assessment object 被輸出成字串。設計、JJ calibration 與 smoke 結果見
[constrained_single_decision_v4.md](docs/constrained_single_decision_v4.md)。

Context 採用 `sectioned-context-v5-action-guide`：分開上一輪操作結果、新來源與舊來源，
保留完整文字與可用 references。`agent.require_evidence_assessment` 預設為 `true`，
要求每次工具呼叫附上簡短的已支持事實與資訊缺口；設為 `false` 可做對照。
完整行為與 JJ 測試說明見 [context_assessment_v3.md](docs/context_assessment_v3.md)。
JJ 使用本機 SSH 與既有 `qwen3.8:27b-q4_K_M`，不同於下方 Brev 的 FP8 設定。

每輪 system prompt 依當前 condition 與已顯示的 E# 產生 `AVAILABLE ACTIONS THIS TURN`：
列出模型須輸出的 exact action 名稱，說明全域 embedding search 與局部 entity links 的
候選範圍、回傳單位和上下文差異。Prompt 不列 backend `top_k`。共用 skill 不指定檢索
順序；action guide 與實際 decision schema 由同一當輪 action space 生成。設計與測試見
[action_guide_v5.md](docs/action_guide_v5.md)。

```bash
python scripts/run_context_comparison.py --data-root /path/to/interface_study_v2 --output /path/to/new-comparison-run
python scripts/analyze_context_comparison.py --run /path/to/new-comparison-run --output /path/to/comparison.json
```

比較程式依序執行 assessment 關閉／開啟，各三資料集 × 七配置，合計 42 episodes；
只做 smoke，不會自動啟動完整實驗。

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

部署到 Brev 後先對每套已下載的 substrate 執行：

```powershell
python scripts/validate_v2_substrate.py --substrate data/.../substrate
```

這個檢查會拒絕非 `qwen3-embedding:4b`、缺少 dense passage/sentence/entity
index，或 index dimension 不一致的 substrate；它不會覆寫既有資料。
遠端部署流程使用本機 WSL 的 Brev CLI SSH；不使用 Jupyter。

v2 的條件是 C0、C1、C2、C3、C5、C4 與 A1。C0/C1 比較全域 passage 與
全域 sentence search；C2/C3 比較 entity 導向的 passage/sentence landing；C5/C4
則測試 global sentence search 加入 local navigation 後的額外效果。A1 是
annotation-only control，不列入主要 factorial contrasts。

正式設定使用 `Qwen/Qwen3.8-27B-FP8` 的 vLLM OpenAI-compatible server。
temperature 0、關閉 thinking、`num_ctx=32768`、每次最多 2,048 output tokens、
每題 15 次正常 policy decisions 與最多一次 FINISH-only finalize。執行前先完成
deterministic contract tests 與遠端 calibration。

```powershell
python scripts/run_interface_study.py `
  --substrate data/interface_study/substrate `
  --questions data/interface_study/pilot_questions.json `
  --source-manifest data/interface_study/source_manifest.json `
  --config configs/interface_study_v2_qwen38_vllm.yaml `
  --output runs/interface-study-v2 `
  --conditions C0 C1 C2 C3 C5 C4 A1
python scripts/analyze_interface_study.py --run runs/interface-study-v2
```

每個 episode 的 `episode.json` 保存實際 messages、agent-visible capability registry、
實際送給 provider 的 decision schema 與 digest、raw structured decision、visible source
spans、exposed refs、provider token metadata 與錯誤原因。gold answer/support 只在
evaluation sidecar，不會進入 Policy context。每輪只能形成一個 decision；schema 不包含
當前 configuration 不可用的 action。

## Acceptance criteria

所有正常決策（valid、protocol-invalid、state-invalid、duplicate、empty）都計入
15 次上限；finalize 不得搜尋、擴展或讀取新資料。resume 會拒絕
source/config/substrate/renderer/skill/provider-protocol hash 改變。舊的 v1 pilot
只作流程診斷；v2 才是重新校準後的研究 run。

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

要對 GraphRAG-Benchmark 全量執行七個介面條件，可使用：

```powershell
python scripts/run_interface_study.py `
  --dataset novel `
  --substrate data/graphrag_benchmark/novel/substrate `
  --questions data/graphrag_benchmark/novel/questions.json `
  --source-manifest data/graphrag_benchmark/novel/source_manifest.json `
  --judge-config configs/semantic_judge_qwen.yaml `
  --semantic-eval `
  --conditions C0 C1 C2 C3 C5 C4 A1 `
  --output runs/graphrag-novel-v1
```

Medical 只需把 dataset、substrate、questions、source manifest 與 output 換成
`medical` 對應路徑。若 target episodes 已完成，可用 `evaluate_semantics.py` 只補做
judge，避免重跑 Agent。
