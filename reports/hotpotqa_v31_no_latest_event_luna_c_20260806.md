# V3.1 `latest_event` Ablation：Luna C Skill

日期：2026-08-06

## 實驗問題

V3.1 Policy input 原本同時包含：

- `semantic_memory`：所有目前可見的語意內容
- `attempted_actions`：過去 Action、outcome 與 error code 的精簡歷史
- `latest_event`：最近一次 Action 的 outcome、retrieved token usage 與詳細 error

本 ablation 只從 Policy prompt 移除 `latest_event`。Semantic Memory、Action History、Assessment、budget、structured action schema、retriever、Skill 和執行限制全部保持不變。Raw Observation 仍保存在 trajectory artifact。

## 固定條件

- Dataset：`hotpotqa_resample20_seed20260805`，相同 20 題
- Policy：OpenAI `gpt-5.6-luna`
- Workflow：`single_agent_v3_typed_refs`
- Skill：V3.1-C `hotpotqa_v31_c_posthoc_guarded.md`
- 對照組：`v3_include_latest_event: true`
- Ablation：`v3_include_latest_event: false`
- Judge：OpenAI `gpt-5.6-luna`，只看 answer 與 gold，不看 evidence

## 結果

| 指標 | 有 latest_event | 無 latest_event | 差異 |
|---|---:|---:|---:|
| Luna judge | 19/20 | 19/20 | 0 |
| Contain | 17/20 | 18/20 | +1 |
| Normalized exact | 9/20 | 10/20 | +1 |
| FINISH | 20/20 | 20/20 | 0 |
| Invalid attempts | 0 | 0 | 0 |
| Reference errors | 0 | 0 | 0 |
| Policy calls | 90 | 90 | 0 |
| Input tokens | 310,729 | 302,560 | -8,169 (-2.6%) |
| Output tokens | 17,930 | 17,070 | -860 (-4.8%) |
| Reasoning tokens | 6,921 | 6,399 | -522 (-7.5%) |
| Total inference tokens | 328,659 | 319,630 | -9,029 (-2.7%) |
| Total tokens / call | 3,651.8 | 3,551.4 | -2.7% |
| Retrieved tokens | 31,235 | 33,766 | +2,531 (+8.1%) |
| SEARCH after observation | 38 | 38 | 0 |
| SEARCH after no progress | 2 | 3 | +1 |

Ablation judge 另用 20 calls、3,291 tokens，未計入 Agent inference usage。

## Action 分布

| Action | 有 latest_event | 無 latest_event |
|---|---:|---:|
| SEARCH | 58 | 58 |
| EXPAND | 1 | 0 |
| READ | 11 | 12 |
| FINISH | 20 | 20 |

兩組 calls 相同，因此這次 2.7% token 降幅確實來自每輪 Policy context／generation 較短，不是少呼叫造成。

## 逐題翻轉

兩個版本各有一題由對變錯，彼此抵銷：

1. The Fatima Mansions vs Nude 成立人數
   - 有 latest event：錯誤回答 Nude。
   - 無 latest event：正確回答 The Fatima Mansions（five vs three）。
2. Taglierini 題目所指麵條的英文翻譯
   - 有 latest event：正確回答 `Little hairs`。
   - 無 latest event：錯誤回答 `To cut`。

因此 contain/exact 的提升不能當作語意正確率提升；Luna judge 仍是 19/20。

## 解讀

在本次 Luna-C run，`latest_event` 沒有提供可觀察到的準確率收益，移除後：

- 語意正確率持平。
- Invalid 與 reference error 都維持 0。
- 每 call tokens 下降約 2.7%。
- 無進展後再次 SEARCH 小幅增加 1 次，但沒有造成總 calls 增加。

原因是成功 retrieval 後，`latest_event` 的大部分資訊已經存在別處：Action/outcome 已在 `attempted_actions`，新內容已進入 `semantic_memory`。因此它在沒有錯誤的 Luna trajectory 中主要是重複訊息。

但這個結果尚不能證明應全面刪除 `latest_event`。它比 `attempted_actions` 多保留詳細 `error.message`、retryable 狀態與最近一次 usage；本次 20 題完全沒有 invalid，沒有測到它對 error recovery 的價值。對先前 reference invalid 較多的 Qwen，結果可能不同。

目前最保守的結論是：對這批 Luna-C，拿掉 `latest_event` 可省約 2.7% Policy tokens，品質持平；是否成為正式預設，需要重複 runs 或在 invalid-rich 模型上再驗證。

## 驗證與 Artifact

- V3/V3.1 targeted regression：58 passed
- Full regression suite：passed（4 conditional skips）
- Config：`configs/agentic_hotpotqa_luna_v3_typed_refs_no_latest_event.yaml`
- Inference：`runs/hotpotqa_resample20_seed20260805/luna/v3_typed_refs_no_latest_event/c_posthoc_guarded/`
- Judge：`runs/hotpotqa_resample20_seed20260805/luna_judge_luna_v31_c_no_latest_event_20260806/`
