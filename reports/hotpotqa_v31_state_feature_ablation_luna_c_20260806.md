# V3.1 Policy State Feature Ablation：Luna C Skill

日期：2026-08-06

## 研究問題

針對 V3.1 Policy input 中四個可能重複或無效的 state feature，分別單獨移除：

1. `last_assessment`
2. `latest_event`
3. `attempted_actions`
4. `budget`

這是 one-factor-at-a-time ablation。每一組只拿掉一個欄位，不做「四個同時移除」，因此可以把觀察差異對應到單一 Policy-visible feature。

State Management 仍完整保存 assessment、Observation、Action history 與 budget；Controller 也繼續執行相同上限。Ablation 只決定 Context Builder 是否將該欄位放進 LLM Policy prompt。Raw trajectory 與 audit data 不刪除。

## 固定條件

- Dataset：`hotpotqa_resample20_seed20260805`，固定 20 題
- Workflow：V3.1 `single_agent_v3_typed_refs`
- Policy model：OpenAI `gpt-5.6-luna`
- Skill：只使用 C `hotpotqa_v31_c_posthoc_guarded.md`
- 相同 embedding、substrate、retriever、expansions 與 budgets
- `max_steps=10`、`max_policy_attempts=12`、`max_retrieved_tokens=12000`
- Judge：OpenAI `gpt-5.6-luna`，只看 generated answer 與 gold answer
- Baseline：全部四個 state feature 都顯示

## 主要結果

| Variant | Judge | Contain | Exact | Invalid | Calls | Total tokens |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | **19/20** | 17/20 | 9/20 | 0 | 90 | 328,659 |
| −Last assessment | 18/20 | 17/20 | 9/20 | 0 | 83 | 277,765 |
| −Latest event | **19/20** | 18/20 | 10/20 | 0 | 90 | 319,630 |
| −Attempted actions | **19/20** | 18/20 | 10/20 | **4** | **101** | **342,063** |
| −Remaining budget | 18/20 | 17/20 | 8/20 | 0 | 80 | 272,608 |

## 相對 Baseline 的效率差異

| Variant | Calls | Input tokens | Total tokens | Tokens/call | Retrieved tokens |
|---|---:|---:|---:|---:|---:|
| −Last assessment | -7 (-7.8%) | -15.6% | -50,894 (-15.5%) | -8.4% | -15.9% |
| −Latest event | 0 | -2.6% | -9,029 (-2.7%) | -2.7% | +8.1% |
| −Attempted actions | +11 (+12.2%) | +3.7% | +13,404 (+4.1%) | -7.3% | -13.6% |
| −Remaining budget | -10 (-11.1%) | -17.1% | -56,051 (-17.1%) | -6.7% | -22.6% |

Judge token 不計入 Agent inference usage。

## Action 行為

| Variant | SEARCH | EXPAND | READ | FINISH | SEARCH after no progress |
|---|---:|---:|---:|---:|---:|
| Baseline | 58 | 1 | 11 | 20 | 2 |
| −Last assessment | 54 | 0 | 9 | 20 | 0 |
| −Latest event | 58 | 0 | 12 | 20 | 3 |
| −Attempted actions | **72** | 1 | 8 | 20 | **12** |
| −Remaining budget | 52 | 0 | 8 | 20 | 2 |

所有 variant 都完成 20/20 FINISH。

## 各欄位解讀

### 1. Last assessment：有品質訊號，但也提高探索成本

移除後 judge 從 19/20 降為 18/20，calls 減少 7，total tokens 減少 15.5%。錯誤 flip 是麵條翻譯題：Baseline 正確找到 `little hairs`，移除 assessment 後回答 `to cut`。

這符合 assessment 的設計目的：它讓下一輪持續記得「真正缺少的是哪個 fact」，避免有相關但不足的證據時太早收斂。拿掉後更便宜，但這批資料出現 5 percentage-point 的 judge 損失。初步建議保留。

### 2. Latest event：本次最像冗餘欄位

移除後 judge 維持 19/20、calls 不變、invalid 維持 0，total tokens 降低 2.7%。它有兩個相反 flip：修正樂團人數題，但弄錯麵條翻譯題，淨正確率不變。

成功 retrieval 時，latest event 的 action/outcome 已被 `attempted_actions` 記錄，內容結果也已 materialize 到 `semantic_memory`，因此主要是重複資訊。對本次 Luna-C 可以考慮移除；但本批沒有 invalid，尚未測到它的詳細 error message 對 recovery 的價值。

### 3. Attempted actions：對 workflow efficiency 明確有效

移除後 judge 仍是 19/20，但：

- invalid：0 → 4
- calls：90 → 101
- SEARCH：58 → 72
- no-progress 後 SEARCH：2 → 12
- total tokens：增加 4.1%

四次 invalid 全部都是 `duplicate_action`，而且全部是重複的 SEARCH。這提供直接的 mechanistic evidence：Semantic Memory 能告訴模型「知道什麼」，但不能可靠告訴模型「哪些 query/method 已經試過」。

雖然拿掉 history 讓單次 prompt 平均縮短 7.3%，多出的重複 Policy calls 抵銷了節省，總成本反而上升。建議保留精簡的 Attempted actions。

### 4. Remaining budget：品質與成本的 trade-off

移除後 judge 從 19/20 降為 18/20，但 calls 減少 10，total tokens 減少 17.1%。錯誤 flip 同樣是麵條題。

Policy 看不到 remaining budget 後反而更早 FINISH，而不是盲目使用更多步數。這可能表示 budget display 在 Baseline 中幫助模型知道仍有探索空間；拿掉後模型較快接受不完整證據。不過這也可能包含單次 Luna sampling 差異。若優先追求 answer quality，初步建議保留；若目標是成本，可以再測一個更短、更中性的 budget 表示法，而不是完全刪除。

## 逐題 Judge flip

- `−Last assessment`：Baseline 對、ablation 錯 1 題（麵條翻譯）。
- `−Latest event`：1 題由錯變對、1 題由對變錯，淨值 0。
- `−Attempted actions`：沒有 judge label flip，但 workflow 多出 4 次 duplicate invalid。
- `−Remaining budget`：Baseline 對、ablation 錯 1 題（麵條翻譯）。

## 初步架構建議

依這一次 Luna-C ablation：

1. 保留 `attempted_actions`：它不是純 context 負擔，能直接防止重複搜尋。
2. 保留 `last_assessment`：它可能維持 missing-information focus。
3. `latest_event` 可列為優先移除候選：本次品質持平且節省 2.7%。
4. 暫時保留 `budget`，或後續改成更短的單行表示；完全移除雖省 17.1%，但 judge 少 1 題。

推薦的精簡 Policy state：

```text
last_assessment
semantic_memory
attempted_actions（精簡語意摘要）
budget（精簡）
```

`latest_event` 可不顯示；完整 Observation 繼續留在 audit。

## 限制

這是固定 20 題、每個 variant 單次 persisted run。Luna 的搜尋 query 與 trajectory 並非完全 deterministic，因此 1 題 flip 不能直接宣稱統計因果。`Attempted actions` 的 duplicate trace 是較強的機制證據；其他三項若要形成論文結論，建議至少重複 3–5 runs，報平均值與變異。

## 實作與驗證

- 新增四個獨立 config flags，預設皆為 `true`。
- 新增 Context Builder tests，確認每次只移除指定欄位。
- Targeted tests：50 passed。
- Full regression suite：passed（4 conditional skips）。

Artifacts：

- Baseline：`runs/hotpotqa_resample20_seed20260805/luna/v3_typed_refs/c_posthoc_guarded/`
- −Last assessment：`runs/hotpotqa_resample20_seed20260805/luna/v3_typed_refs_no_last_assessment/c_posthoc_guarded/`
- −Latest event：`runs/hotpotqa_resample20_seed20260805/luna/v3_typed_refs_no_latest_event/c_posthoc_guarded/`
- −Attempted actions：`runs/hotpotqa_resample20_seed20260805/luna/v3_typed_refs_no_attempted_actions/c_posthoc_guarded/`
- −Remaining budget：`runs/hotpotqa_resample20_seed20260805/luna/v3_typed_refs_no_budget/c_posthoc_guarded/`
- New-arm judge：`runs/hotpotqa_resample20_seed20260805/luna_judge_luna_v31_c_state_feature_ablations_20260806/`
