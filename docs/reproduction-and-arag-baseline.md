# 新電腦完整重現與 A-RAG baseline 指南

本文件是目前 Agentic RAG V3.2、五套 benchmark、Ollama Qwen、SkillOpt 與
A-RAG baseline 的單一重現入口。`data/`、`artifacts/`、`runs/`、`.env`
及 `A-RAG/` 都被 Git 忽略；只 clone 主 repository 不會自動取得這些
大型資料或本機 secret。

## 1. 固定實驗契約

| 項目 | 固定值 |
|---|---|
| Agentic RAG repository | `https://github.com/pw-Chen-km/Agentic-RAG.git` |
| A-RAG upstream | `https://github.com/Ayanami0730/arag.git` |
| A-RAG commit | `a44de6b2216bf6791979c4b6ac4ae106212fa1a6` |
| Dataset collection | `Ayanami0730/rag_test`（五個 subsets） |
| Dataset revision | `b9198a5a8702cc35c6df7542529357a9af95d928` |
| Dataset scopes | `<dataset>:benchmark_exact:dev` |
| Qwen | Ollama `qwen3.5:9b` |
| Qwen digest | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` on CPU |
| Embedding dimension | 384 |
| Reference machine | Windows, RTX 4070 12 GB, NVIDIA driver 581.57 |
| Python | 3.12.8 (`pyproject.toml` permits Python 3.12.x) |
| HotpotQA held-out set | fixed resample-20, 16 bridge + 4 comparison |
| SkillOpt | HotpotQA 20/6/6；其他 profile smoke 6/6/6；seed 42 |

### 完整 raw dataset registry

五套資料都來自同一個 pinned revision。`chunks`／`questions` 是原始資料筆數，
不是 SkillOpt 或 held-out 題數。

| Dataset | Chunks | Questions | `chunks.json` SHA-256 | `questions.json` SHA-256 |
|---|---:|---:|---|---|
| 2WikiMultiHopQA | 658 | 1,000 | `e92b8bcfcd2748100d60ad86819abe2e4cb318dc8d5f33b1aebdb0b4735c8aa8` | `246e43fb624413e38e11e3a582d5945185a3290efe4c2adbe32af2f112b70ab8` |
| HotpotQA | 1,311 | 1,000 | `cb76f6fdb54e7b2853d51d400bacdba01c814baf43b74207bc79c2a06474d231` | `ecc641d532a4d2518f1ceb57627f2e41044e0c4fd07012bf0aaa02327dc770a9` |
| Medical (GraphRAG-Bench) | 225 | 2,062 | `ffbc583386e145f807755d338c55201f4c2a924ba4beec7d363537a08b0c652e` | `022f41d22cd618c7d5f10c5056cf5385f75d8bd4f763b70cc093abb2e0d87753` |
| MuSiQue | 1,354 | 1,000 | `41d439ad258a09b602cce4b4b4151747c0682f30ba64486fec819fc965e84630` | `42dfd487e7e08d0892ed94bd9e0d0e56744cd92239f41b190e156f563ab49fb4` |
| Novel (GraphRAG-Bench) | 1,117 | 2,010 | `34f05833f0cab3956f2066f709b697ec4ee7b4b46c5474cd92d7d95e30e9d738` | `9d7a58fa0d613e46a96b85f461f0920d4f3ad47a322c85e392e2071362d5d963` |

### 五套資料現在都由同一個 V3.2 runtime 支援

目前 `main` 只有一套 V3.2 agent workflow，但 dataset layer 正式支援
`2wikimultihop`、`hotpotqa`、`medical`、`musique`、`novel`。五者共用
Semantic Memory、Controller、Retriever 與 action contract；差異集中在 profile：

- pinned row counts 與 scope；
- question/task-type normalization；
- short/long answer mode；
- 評估 metric contract；
- Novel 重複來源 ID 的 deterministic disambiguation。

奇異點前的 branch/tag 只用於查看已移除的舊 agent workflow，不再需要它來
build、評估或跑這五套資料。

### 派生資料 registry

| 資料 | 用途 | 大小 | SHA-256 |
|---|---|---:|---|
| HotpotQA SkillOpt train | optimizer training | 20 | `79f4e165a390af2ac7752e9aa4ca7765f94dc01821a828cb4f40e02f830826ea` |
| HotpotQA SkillOpt validation | selection/gate | 6 | `67c9fcf7a25da604c976e74eeef4dd83b5b7d1b068da946e26cb6727db6d02fd` |
| HotpotQA SkillOpt test | final SkillOpt test | 6 | `aa2107867a8f9faa406dae77a3aabc17780af093f239c6f334fef969f087e9a6` |
| HotpotQA held-out resample | pre/post skill evaluation | 20 | `57fd2bf9871cd7fd310f5937810caa715539f55f7ffc8134103d0755865fec75` |
| 2Wiki SkillOpt train/validation/test | profile smoke | 6/6/6 | `cdcc8f84abaed31a541a7ca8c8fd3a0517def593f2bf5161f691081d64df2b8c` / `64f5874df22e8d2d5b26f0f7f237334c8cde68bac57b838ed7e8b24eb9f0182f` / `bb9fc06552188d0fc4a55e398788208f84fb24f6523f589b4f1d937fddf4de20` |
| Medical SkillOpt train/validation/test | profile smoke | 6/6/6 | `a7d9876e61620892de74e02ebe83b90f65ec2e623214a995e66369324a3884bb` / `74f5df0c2dab6f862b3089ea21c2a45582d4225e2c1ed430877d406f384054cc` / `9d9ba03cf802add8640996fa0dc347be14ee7ef67ee64bcbc47a7445433380fd` |
| MuSiQue SkillOpt train/validation/test | profile smoke | 6/6/6 | `2d76bbfd88ea845e38d58321cb882db144630e1c8e48a390c30d45b8dcfc999f` / `abdd623845b3786ccaf6f61dcfaee0b95e03282225a73491aa7ea16897a5e87a` / `b462d6293a7219d81f5f07c3c3d94a8294bfd2784adc4ff3174cd1f744bfcacf` |
| Novel SkillOpt train/validation/test | profile smoke | 6/6/6 | `675559e05c358c76d5a473f6ef6ad670ec10f470c8e3140d670113832eb0830e` / `e1df00d2e5010a31a1af92c168449d3124c8e983fcf2adcd74ea1d3d2671d8d0` / `785c76b574fa6b970790c260b1ea296cbbcf32645bc0fc898408b9581dd0c8d3` |

非 HotpotQA split 的完整 hash 保存在各自的 `split_manifest.json`；複製後應以
manifest 驗證，不要只比對上表的縮寫。

每次正式實驗另記錄 `git rev-parse HEAD`、Ollama model digest、config hash
與 skill hash。不要只寫「使用 main」，因為 main 會繼續演進。
不同 GPU 可以得到相同功能設定，但 latency、VRAM 壓力及底層數值運算未必
逐位元一致；要重現效能數據時也應記錄 GPU、driver 與 Ollama version。

## 2. 新電腦 prerequisite

安裝 Git、Git LFS、uv、Ollama、Python 3.12。Windows PowerShell 範例：

```powershell
git lfs install
ollama pull qwen3.5:9b
ollama list

git clone https://github.com/pw-Chen-km/Agentic-RAG.git Agentic-RAG
Set-Location Agentic-RAG
uv sync --frozen --extra dev --extra skillopt
uv run pytest -q
uv run python -m compileall -q src\agentic_rag
```

Ollama 必須能由本機 HTTP endpoint 存取：

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

`.env` 只能留在本機，不得 commit。純 Ollama 流程不需要真正 API key。

## 3. 取得相同五套原始資料

```powershell
git clone https://huggingface.co/datasets/Ayanami0730/rag_test data\rag_test
git -C data\rag_test checkout b9198a5a8702cc35c6df7542529357a9af95d928
git -C data\rag_test lfs pull

Get-ChildItem data\rag_test -Recurse -Filter *.json |
  Get-FileHash -Algorithm SHA256
```

十個 raw-file hash 必須與第 1 節完全相同。不同就停止，不要建立 index。

### 最快搬機方式

若舊電腦仍可使用，直接複製下列 ignored directories 可省下重建時間：

```text
data/rag_test/                         # 五個 raw datasets
data/skillopt/                         # 五套 deterministic splits
data/evaluations/hotpotqa_resample20_seed20260805/
artifacts/*_benchmark_exact/           # 五套正式 V3.2 substrates
A-RAG/data/*/index/                    # 若已建立；目前本機只有 HotpotQA
```

搬完仍要驗證 raw-data hash、`agentic-rag validate`，以及 A-RAG index 使用的
embedding 名稱；不可只相信檔案成功複製。

### 目前本機已有的五套 Agentic RAG substrates

| Artifact | Scope | Chunks | Sentences | Entities | 約略大小 |
|---|---|---:|---:|---:|---:|
| `2wikimultihop_benchmark_exact` | `2wikimultihop:benchmark_exact:dev` | 658 | 24,316 | 24,130 | 92 MB |
| `hotpotqa_benchmark_exact` | `hotpotqa:benchmark_exact:dev` | 1,311 | 46,935 | 36,380 | 162 MB |
| `medical_benchmark_exact` | `medical:benchmark_exact:dev` | 225 | 13,493 | 1,794 | 29 MB |
| `musique_benchmark_exact` | `musique:benchmark_exact:dev` | 1,354 | 47,709 | 38,239 | 166 MB |
| `novel_benchmark_exact` | `novel:benchmark_exact:dev` | 1,117 | 47,510 | 17,569 | 125 MB |

`artifacts/hotpotqa-mini/` 只有 5 chunks，是開發測試 fixture，不是正式研究
dataset。若直接搬 substrate，逐一執行：

```powershell
Get-ChildItem artifacts -Directory | ForEach-Object {
  uv run agentic-rag validate $_.FullName
}
```

Manifest 內含建立時間與舊電腦的絕對 source path，所以「重新 build」後整份
manifest 的 hash 不一定相同；真正必須相同的是 raw source hashes、embedding
model/dimension、record counts、schema 與 scope。直接位元複製 artifact 時才適合
再比對整個 directory 或 manifest hash。

## 4. 建立五套 V3.2 substrate

五份設定都使用 canonical `benchmark_exact` source format、相同 embedding，並
對照 pinned profile 驗證資料筆數。從 `data\rag_test` 一次建立：

```powershell
$datasets = @("2wikimultihop", "hotpotqa", "medical", "musique", "novel")
foreach ($dataset in $datasets) {
  uv run agentic-rag build data\rag_test "artifacts\${dataset}_benchmark_exact" `
    --config "configs\datasets\${dataset}.yaml"
  uv run agentic-rag validate "artifacts\${dataset}_benchmark_exact"
}
```

每份 manifest 應是 schema 2.0、embedding dimension 384，dataset/scope/counts
與第 1、3 節的 registry 相符。Gold questions/answers 只存在 evaluation sidecar，
不進入 runtime retrieval tables。

## 5. 建立 SkillOpt 固定 split

```powershell
uv run agentic-rag skillopt-prepare `
  --dataset-dir data\rag_test `
  --split-dir data\skillopt\hotpotqa_smoke `
  --dataset hotpotqa `
  --seed 42 `
  --split-size 6 `
  --train-size 20
```

`split_manifest.json` 應顯示 train=20、validation=6、test=6。Validation
與 test 是固定舊六題；train 20 不得和它們重疊。

其餘四套建立 profile-stratified 6/6/6 smoke splits：

```powershell
$datasets = @("2wikimultihop", "medical", "musique", "novel")
foreach ($dataset in $datasets) {
  uv run agentic-rag skillopt-prepare `
    --dataset-dir data\rag_test `
    --split-dir "data\skillopt\${dataset}_smoke" `
    --dataset $dataset `
    --seed 42 `
    --split-size 6
}
```

每份 `split_manifest.json` 都保存 raw source hashes、selection algorithm、task
quotas、metric contract 和三個 split hashes。V3.2 會依 manifest 自動設定
SkillOpt 的 train/validation/test 數量。

## 6. 建立固定 held-out 20 題

將以下程式另存為暫存檔並執行，或直接從舊電腦複製已驗證的
`questions.jsonl`。這 20 題排除 SkillOpt 的 32 個 train/validation/test ID。

```python
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
```

```powershell
Get-FileHash data\evaluations\hotpotqa_resample20_seed20260805\questions.jsonl `
  -Algorithm SHA256
```

JSONL hash 必須為 `57fd2bf...fec75`。

## 7. 跑 V3.2 Qwen

No-thinking 使用 `configs/hotpotqa_qwen.yaml`；thinking-high 使用
`configs/hotpotqa_qwen_thinking.yaml`。先做一題 smoke，再跑 20 題：

```powershell
uv run agentic-rag run artifacts\hotpotqa_benchmark_exact `
  "What magazine was started in 1925 and features the work of Barry Werth?" `
  --scope-id hotpotqa:benchmark_exact:dev `
  --skill-file skills\skillopt_seed.md `
  --config configs\hotpotqa_qwen_thinking.yaml `
  --output runs\qwen_thinking_smoke `
  --episode-id smoke

uv run python scripts\run_benchmark_eval.py `
  --substrate artifacts\hotpotqa_benchmark_exact `
  --split data\evaluations\hotpotqa_resample20_seed20260805\questions.jsonl `
  --config configs\hotpotqa_qwen_thinking.yaml `
  --skill skills\skillopt_seed.md `
  --output runs\qwen_thinking_heldout20_initial `
  --expected-count 20 `
  --dataset hotpotqa
```

中斷後在同一命令加 `--resume`。不要刪除 `progress.json`。

五套 substrate/split 都準備好後，可用相同 Policy、Skill 一次跑完 test matrix：

```powershell
uv run python scripts\run_benchmark_matrix.py `
  --config configs\hotpotqa_qwen_thinking.yaml `
  --skill skills\skillopt_seed.md `
  --output runs\qwen_thinking_all_datasets
```

輸出會按 dataset 隔離，並建立 `matrix_summary.json`；中斷續跑時加
`--resume`。不要把五套不同 answer-mode 的分數直接 pool 成單一 accuracy。

## 8. 跑本機 Qwen SkillOpt

此命令的 target Agent、SkillOpt analyst/optimizer、semantic judge 都是本機
`qwen3.5:9b`，thinking 已開啟。它不是 Luna-as-judge，報告時必須標成
`local-Qwen-as-judge`。

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

uv run python scripts\run_local_qwen_skillopt.py `
  --substrate artifacts\hotpotqa_benchmark_exact `
  --split-dir data\skillopt\hotpotqa_smoke `
  --agent-config configs\hotpotqa_qwen_thinking.yaml `
  --skillopt-config configs\hotpotqa_skillopt_qwen.yaml `
  --skill skills\skillopt_seed.md `
  --output runs\skillopt_qwen_thinking
```

最佳 skill 通常位於 `runs/skillopt_qwen_thinking/best_skill.md`。完成後用
第 7 節相同 20 題與 config 再跑一次，只替換 `--skill`，才能得到真正的
pre/post SkillOpt 比較。

## 9. 安裝獨立 A-RAG baseline

A-RAG 不放回 V3.2 `src/`。它是被 Git 忽略、可獨立 pin 的 sibling checkout：

```powershell
git clone https://github.com/Ayanami0730/arag.git A-RAG
git -C A-RAG checkout a44de6b2216bf6791979c4b6ac4ae106212fa1a6
Set-Location A-RAG
uv sync --extra full
```

建立 `A-RAG/configs/hotpotqa_resample20_qwen.yaml`：

```yaml
llm:
  model: "qwen3.5:9b"
  base_url: "http://127.0.0.1:11434/v1"
  temperature: 0.0
  max_tokens: 16384
  reasoning_effort: "high"

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 64

agent:
  max_loops: 10
  max_token_budget: 128000
  verbose: false

data:
  chunks_file: "../data/rag_test/hotpotqa/chunks.json"
  index_dir: "data/hotpotqa/index"
```

建立 A-RAG index：

```powershell
uv run python scripts\build_index.py `
  --chunks ..\data\rag_test\hotpotqa\chunks.json `
  --output data\hotpotqa\index `
  --model sentence-transformers/all-MiniLM-L6-v2 `
  --device cpu
```

### A-RAG token instrumentation

原 upstream prediction 沒有足夠的 per-episode token totals。公平比較前，
在 `src/arag/core/llm.py` 為 `LLMClient` 加入三個累積欄位：

```python
self.total_input_tokens = 0
self.total_output_tokens = 0
self.total_calls = 0
```

取得每次 response usage 後累加：

```python
input_tokens = usage.get("prompt_tokens", 0)
output_tokens = usage.get("completion_tokens", 0)
self.total_input_tokens += input_tokens
self.total_output_tokens += output_tokens
self.total_calls += 1
```

並在 `scripts/batch_runner.py` 的 success/error prediction record 寫入：

```python
"llm_calls": agent.llm.total_calls,
"llm_input_tokens": agent.llm.total_input_tokens,
"llm_output_tokens": agent.llm.total_output_tokens,
"llm_total_tokens": agent.llm.total_input_tokens + agent.llm.total_output_tokens,
```

上述 `reasoning_effort: "high"` 會由 A-RAG 原生的 `LLMClient` 傳到 Ollama
OpenAI-compatible endpoint。V3.2 則透過 native Ollama API 傳送 `think: high`；
兩者意圖相同，但 provider interface 不同，報告中仍必須揭露。正式跑 20 題前，
先用 `--limit 1` 做 smoke test，並確認 prediction 的 output tokens 非零。

### 執行 A-RAG 20 題

```powershell
$env:ARAG_API_KEY = "ollama-local"

uv run python scripts\batch_runner.py `
  --config configs\hotpotqa_resample20_qwen.yaml `
  --questions ..\data\evaluations\hotpotqa_resample20_seed20260805\questions.json `
  --output results\hotpotqa_resample20_qwen_thinking `
  --workers 1
```

`workers=1` 可避免單張 GPU 上的並行 request 改變 latency、OOM 與 token 行為。
Batch runner 原生支援 checkpoint resume，重跑同一 output 會跳過已完成 ID。

### A-RAG 的五資料集 baseline

A-RAG 本身可以分別為五套 raw chunks 建 index；不要讓不同 dataset 共用同一個
`index_dir`。為每套建立一份 config，只替換：

```yaml
data:
  chunks_file: "../data/rag_test/<dataset>/chunks.json"
  index_dir: "data/<dataset>/index"
```

然後以相同 dataset 的完整 questions 或固定 split 執行：

```powershell
uv run python scripts\batch_runner.py `
  --config configs\<dataset>_qwen.yaml `
  --questions ..\data\rag_test\<dataset>\questions.json `
  --output results\<dataset>_qwen `
  --workers 1
```

`<dataset>` 分別是 `2wikimultihop`、`hotpotqa`、`medical`、`musique`、
`novel`。V3.2 與 A-RAG 必須逐資料集使用同一 raw chunks 和同一 question
split；每列比較都保留 dataset profile。Medical／Novel 必須用長答案的 LLM
judge contract，不能拿 HotpotQA contain contract 代替。

## 10. 轉成共同 summary 並評分

先回到 Agentic-RAG root。舊 A-RAG normalization adapter 保存在 archive
branch，不混入正式 V3.2 runtime：

```powershell
Set-Location ..
New-Item -ItemType Directory -Force runs\tools | Out-Null
git show origin/codex/pre-singularity-version-archive:scripts/summarize_arag_predictions.py `
  | Set-Content -Encoding utf8 runs\tools\summarize_arag_predictions.py

uv run python runs\tools\summarize_arag_predictions.py `
  --predictions A-RAG\results\hotpotqa_resample20_qwen_thinking\predictions.jsonl `
  --output runs\arag_qwen_thinking_summary.json `
  --model qwen3.5:9b
```

若要沿用論文實驗的 Luna-as-judge，`.env` 提供 `OPENAI_API_KEY` 後：

```powershell
uv run python scripts\judge_benchmark.py `
  --run v32=runs\qwen_thinking_heldout20_initial\summary.json `
  --run arag=runs\arag_qwen_thinking_summary.json `
  --output runs\judge_v32_vs_arag `
  --model gpt-5.6-luna `
  --dataset hotpotqa
```

若不使用外部 API，則兩邊都使用同一個 local-Qwen judge；不可拿一邊的
Luna judge 與另一邊的 contain accuracy 直接比較。

## 11. 公平比較表

至少同時報告：

| 類別 | 指標 |
|---|---|
| Accuracy | semantic judge、contain、normalized exact |
| Termination | completed、blank、budget exhausted、error |
| Cost | calls、input/output/total tokens、retrieved tokens、wall-clock |
| Retrieval | SEARCH/EXPAND/READ；A-RAG keyword/semantic/chunk_read |
| Interface | invalid attempts、reference errors、tool-call errors |
| Contract | model digest、thinking mode、context size、budgets、workers |

重要限制：V3.2 的 `max_retrieved_tokens=12000` 與 A-RAG 的
`max_token_budget=128000` 不是同一種 budget。歷史結果應保留各系統 native
budget；若另做 matched-budget ablation，必須單獨命名，不能覆蓋 native
baseline。

## 12. 最終重現檢查清單

- `git status` 沒有把 `.env`、data、artifact、run 或 A-RAG 加入追蹤。
- 五套 raw datasets 共十個 SHA-256 正確。
- held-out JSONL SHA-256 正確且恰好 20 題。
- HotpotQA SkillOpt split 為 20/6/6；其他四套歷史 split 為 6/6/6；seed 皆為 42。
- V3.2 substrate validation 通過且 record counts 正確。
- Ollama 顯示 `qwen3.5:9b`，並記錄 model digest。
- V3.2 與 A-RAG 都使用同一 raw chunks、20 questions、embedding model、workers。
- thinking／no-thinking 不混在同一欄位比較。
- judge provider 與 judge tokens 和 target inference tokens 分開統計。
- 每個 output directory 唯一；需要接續時使用 resume，不手動拼接結果。
