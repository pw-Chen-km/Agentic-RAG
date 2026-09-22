# Constrained single-decision v4

## 為什麼更換 protocol

v3 使用 provider 的 native tool calling。JJ 的 42-episode comparison 出現兩種格式錯誤：

1. 同一輪回傳兩個 tool calls；
2. 把 assessment object 回傳成字串。

兩者都被後端正確拒絕，但代表 native tool API 沒有在 provider 邊界可靠地強制「每輪一個
decision」及 assessment 的型別。只在 prompt 補一句規則不能消除這個問題，也會把
provider 格式服從混入 retrieval policy 的量測。

v4 改用一個 JSON Schema 限制整個 decision。每輪輸出包含：

- `supported_facts`：最多五個字串；
- `missing_information`：最多三個字串；
- `action`：且只有一個當前 configuration 可用的 action。

Assessment 關閉時，前兩個欄位不出現在 schema。Entity reference 的 enum 只包含目前
observation 可見的 E#。Budget finalize 的 schema 只允許 `finish`。不存在 raw JSON
fallback，也不會默默把非法輸出改成合法 action。

Policy-visible action 使用 `find_passages`、`find_sentences`、
`follow_entity_to_passages`、`follow_entity_to_sentences` 與 `finish`。內部的 action model
只在 schema 驗證成功後才轉換；模型不需要輸出 `DENSE`、`CHUNK` 或 `EXPAND`。

Agent-visible capability registry 仍保存在 artifact，供 audit 判斷當輪可用能力；真正送到
provider 的約束則完整保存在 `decision_schema`。兩者分開，避免把說明用的 registry
誤當成 provider 實際執行的 protocol。

## 驗證結果

本機 regression tests：91/91 通過。測試包含多 action、assessment 字串、未知欄位、
不可見 E#、condition isolation、context 分區、budget finalize、Ollama 及
OpenAI-compatible provider request。

JJ provider probe 使用 `qwen3.8:27b-q4_K_M`，對七個 configurations 各測 initial 與
entity-visible state，共 14 calls：

- 14/14 provider status `ok`；
- 14/14 只有一個 decision；
- 14/14 selected action 在當輪 allowlist；
- 14/14 assessment 型別正確；
- 0 raw native tool calls。

這是小型 protocol smoke，不是 99% reliability 的統計證明，也沒有強迫模型在自然軌跡
選用每種 retrieval action。Action branch 與 backend execution 由 deterministic tests
覆蓋；正式部署仍需在目標 FP8/vLLM stack 做更大規模 calibration。

JJ workflow smoke 路徑：

```text
/home/jj/Large_Space_A/chiu/PW/Agentic-RAG-data/interface_study_v2/runs/single-decision-v4-smoke-20260922T121817Z
```

同一份 artifacts 已下載至：

```text
runs/single-decision-v4-smoke-20260922T121817Z
```

HotpotQA、Novel、Medical 各一題，七個 configurations，共 21 episodes：

- 21/21 產生 terminal artifact 並以 `finish` 結束；
- 44 次 policy decisions，0 protocol-invalid；
- 1 state-invalid：HotpotQA C1 重複相同 passage search，被後端拒絕；
- 0 backend execution error；
- artifact audit 無 capability、budget、schema、visible-span 或 missing-file 違規。

自然工具使用為 22 次 passage search、1 次 sentence search、21 次 finish；本次沒有自然
選用 entity navigation。這不代表 entity navigation 無效或不可操作，只表示三個 smoke
questions 不提供 action-uptake coverage。Medical 多數 configurations 第一輪直接 finish，
也是 policy 選擇，不是 protocol failure。因此這 21 episodes 只能驗證 workflow，不能用來
比較研究效果或推論工具偏好。

## 正式實驗 gate

正式 FP8/vLLM run 前仍需：

1. 在完全相同的 target model、served-model digest、schema 與 decoding config 上重跑
   provider calibration；
2. 用足夠 calls 檢驗預先設定的 protocol-valid 門檻，而不是用 14/14 宣稱長期 ≥99%；
3. 執行每個 retrieval branch 的 deterministic backend calibration；
4. 跑固定問題的 dataset × configuration workflow smoke，確認 terminal artifacts、usage、
   source spans、budget/finalize 與 resume hashes；
5. gate 全部通過後才建立新的正式 run，不接續 v1/v3/v4 smoke。

`scripts/calibrate_interface.py --live` 現在會另報 `protocol_valid_rate`、valid/count 與
`live_protocol_smoke_passed/failed`。它不執行 retrieval，因此
`execution_success_rate` 刻意保持 `null`，不能被誤報為 backend success。
`--repetitions N` 可在每個 condition 的 initial/entity-visible state 重複 provider probe；
正式樣本數與通過規則需在跑之前固定，不能看到結果後才調整。
