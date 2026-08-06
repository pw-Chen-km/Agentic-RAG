# 新電腦完整重現與 A-RAG baseline 指南

本文件是目前 Agentic RAG V3.2、HotpotQA、Ollama Qwen、SkillOpt 與
A-RAG baseline 的單一重現入口。`data/`、`artifacts/`、`runs/`、`.env`
及 `A-RAG/` 都被 Git 忽略；只 clone 主 repository 不會自動取得這些
大型資料或本機 secret。

## 1. 固定實驗契約

| 項目 | 固定值 |
|---|---|
| Agentic RAG repository | `https://github.com/pw-Chen-km/Agentic-RAG.git` |
| A-RAG upstream | `https://github.com/Ayanami0730/arag.git` |
| A-RAG commit | `a44de6b2216bf6791979c4b6ac4ae106212fa1a6` |
| Dataset | `Ayanami0730/rag_test` |
| Dataset revision | `b9198a5a8702cc35c6df7542529357a9af95d928` |
| HotpotQA scope | `hotpotqa:benchmark_exact:dev` |
| Qwen | Ollama `qwen3.5:9b` |
| Qwen digest | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` on CPU |
| Embedding dimension | 384 |
| Reference machine | Windows, RTX 4070 12 GB, NVIDIA driver 581.57 |
| Python | 3.12.8 (`pyproject.toml` permits Python 3.12.x) |
| Held-out set | fixed resample-20, 16 bridge + 4 comparison |
| SkillOpt | train 20 + validation 6 + test 6, seed 42 |

Raw-data hashes:

```text
hotpotqa/chunks.json
cb76f6fdb54e7b2853d51d400bacdba01c814baf43b74207bc79c2a06474d231

hotpotqa/questions.json
ecc641d532a4d2518f1ceb57627f2e41044e0c4fd07012bf0aaa02327dc770a9

held-out questions.jsonl
57fd2bf9871cd7fd310f5937810caa715539f55f7ffc8134103d0755865fec75
```

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

## 3. 取得相同 HotpotQA 原始資料

```powershell
git clone https://huggingface.co/datasets/Ayanami0730/rag_test data\arag_hotpotqa
git -C data\arag_hotpotqa checkout b9198a5a8702cc35c6df7542529357a9af95d928
git -C data\arag_hotpotqa lfs pull

Get-FileHash data\arag_hotpotqa\hotpotqa\chunks.json -Algorithm SHA256
Get-FileHash data\arag_hotpotqa\hotpotqa\questions.json -Algorithm SHA256
```

兩個 hash 必須與第 1 節完全相同。不同就停止，不要建立 index。

### 最快搬機方式

若舊電腦仍可使用，直接複製下列 ignored directories 可省下重建時間：

```text
data/arag_hotpotqa/
data/skillopt/hotpotqa_smoke/
data/evaluations/hotpotqa_resample20_seed20260805/
artifacts/hotpotqa_benchmark_exact/
A-RAG/data/hotpotqa/index/        # 只供 A-RAG
```

搬完仍要驗證 raw-data hash、`agentic-rag validate`，以及 A-RAG index 使用的
embedding 名稱；不可只相信檔案成功複製。

## 4. 建立 V3.2 substrate

```powershell
uv run agentic-rag build data\arag_hotpotqa\hotpotqa artifacts\hotpotqa_benchmark_exact `
  --corpus-id hotpotqa_benchmark_exact `
  --split dev `
  --dataset hotpotqa `
  --source-format hotpotqa_benchmark_exact `
  --benchmark-scope-id hotpotqa:benchmark_exact:dev `
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 `
  --embedding-device cpu `
  --validate-benchmark-profile

uv run agentic-rag validate artifacts\hotpotqa_benchmark_exact
```

正確 manifest 至少應包含：1,311 chunks、46,935 sentences、36,380
entities、embedding dimension 384、schema version 2.0。

## 5. 建立 SkillOpt 固定 split

```powershell
uv run agentic-rag skillopt-prepare `
  --dataset-dir data\arag_hotpotqa `
  --split-dir data\skillopt\hotpotqa_smoke `
  --dataset hotpotqa `
  --seed 42 `
  --split-size 6 `
  --train-size 20
```

`split_manifest.json` 應顯示 train=20、validation=6、test=6。Validation
與 test 是固定舊六題；train 20 不得和它們重疊。

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
    Path("data/arag_hotpotqa/hotpotqa/questions.json").read_text(encoding="utf-8")
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

uv run python scripts\run_hotpotqa_eval.py `
  --substrate artifacts\hotpotqa_benchmark_exact `
  --split data\evaluations\hotpotqa_resample20_seed20260805\questions.jsonl `
  --config configs\hotpotqa_qwen_thinking.yaml `
  --skill skills\skillopt_seed.md `
  --output runs\qwen_thinking_heldout20_initial `
  --expected-count 20
```

中斷後在同一命令加 `--resume`。不要刪除 `progress.json`。

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
  chunks_file: "../data/arag_hotpotqa/hotpotqa/chunks.json"
  index_dir: "data/hotpotqa/index"
```

建立 A-RAG index：

```powershell
uv run python scripts\build_index.py `
  --chunks ..\data\arag_hotpotqa\hotpotqa\chunks.json `
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
uv run python scripts\judge_hotpotqa.py `
  --run v32=runs\qwen_thinking_heldout20_initial\summary.json `
  --run arag=runs\arag_qwen_thinking_summary.json `
  --output runs\judge_v32_vs_arag `
  --model gpt-5.6-luna
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
- raw HotpotQA 兩個 SHA-256 正確。
- held-out JSONL SHA-256 正確且恰好 20 題。
- SkillOpt split 為 20/6/6，seed 42。
- V3.2 substrate validation 通過且 record counts 正確。
- Ollama 顯示 `qwen3.5:9b`，並記錄 model digest。
- V3.2 與 A-RAG 都使用同一 raw chunks、20 questions、embedding model、workers。
- thinking／no-thinking 不混在同一欄位比較。
- judge provider 與 judge tokens 和 target inference tokens 分開統計。
- 每個 output directory 唯一；需要接續時使用 resume，不手動拼接結果。
