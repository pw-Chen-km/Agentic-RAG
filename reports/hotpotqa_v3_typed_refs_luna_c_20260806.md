# HotpotQA V3.1 Typed References：Luna C Skill 單次實驗

日期：2026-08-06

## 實驗設定

- Dataset：`hotpotqa_resample20_seed20260805`，固定 20 題
- Policy model：OpenAI `gpt-5.6-luna`
- Workflow：`single_agent_v3_typed_refs`（V3.1）
- Skill：`hotpotqa_v31_c_posthoc_guarded.md`
- Budget：`max_steps=10`、`max_policy_attempts=12`、`max_retrieved_tokens=12000`
- Judge：OpenAI `gpt-5.6-luna`，只看 generated answer 與 gold answer，不看 supporting evidence
- 對照：相同 20 題、相同 Luna、相同 C 策略內容的既有 V3-C persisted run

Judge token 不計入 Agent 推論 token。

## 結果

| 指標 | Luna V3-C | Luna V3.1-C | 差異 |
|---|---:|---:|---:|
| Luna judge | 19/20 | 19/20 | 0 |
| Contain | 17/20 | 17/20 | 0 |
| Normalized exact | 10/20 | 9/20 | -1 |
| 完成 FINISH | 20/20 | 20/20 | 0 |
| Invalid attempts | 0 | 0 | 0 |
| Reference errors | 0 | 0 | 0 |
| Policy calls | 93 | 90 | -3 (-3.2%) |
| Input tokens | 320,443 | 310,729 | -9,714 (-3.0%) |
| Output tokens | 19,033 | 17,930 | -1,103 (-5.8%) |
| Total inference tokens | 339,476 | 328,659 | -10,817 (-3.2%) |
| Retrieved tokens | 32,170 | 31,235 | -935 (-2.9%) |
| Total tokens / policy call | 3,650.3 | 3,651.8 | +0.04% |

V3.1-C 的 judge 額外使用 20 calls、3,273 tokens；這些沒有混入上表的 inference usage。

## Action 分布

| Action | Luna V3-C | Luna V3.1-C |
|---|---:|---:|
| SEARCH | 61 | 58 |
| EXPAND | 1 | 1 |
| READ | 11 | 11 |
| FINISH | 20 | 20 |

V3.1 少了 3 次 SEARCH，正好也少了 3 次 Policy call。因為每 call 的平均 token 幾乎不變，這次總 token 的下降主要是 action trajectory 較短，不是 typed references 本身顯著縮短每輪 prompt。

## 逐題翻轉

兩版不是同一題錯，而是一正一負互相抵銷。

### V3 錯、V3.1 對

- 題目：Taglierini 相似但不是 tagliatelle 的麵，其英文翻譯為何？
- Gold：`little hairs`
- V3：`To cut.`（錯）
- V3.1：`Little hairs.`（對）
- V3 用 8 steps、6 SEARCH、31,417 tokens，檢索方向逐漸鎖在 tagliolini/tagliare，最後把字根翻譯當答案。
- V3.1 用 4 steps、3 SEARCH、13,353 tokens，第二輪即把候選辨認成 capellini，再查到正確翻譯。

這是搜尋假設與 query reformulation 改善，不是 reference parsing 改善；兩邊都沒有 reference error。

### V3 對、V3.1 錯

- 題目：The Fatima Mansions 與 Nude 成立時哪團成員較多？
- Gold：`The Fatima Mansions`
- V3：`The Fatima Mansions ... five, compared with Nude's three.`（對）
- V3.1：`Nude ... three, compared with The Fatima Mansions being formed by Cathal Coughlan.`（錯）
- V3 用 7 steps、4 SEARCH、30,160 tokens，取得兩邊可比較的人數。
- V3.1 用滿 10 environment steps，加上最後 FINISH 共 11 attempts；做了 7 SEARCH、1 EXPAND、2 READ，仍沒有取得 The Fatima Mansions 的完整 founding count，最後把「由 Cathal Coughlan 組成」錯當成一人。

這題的失敗點是 evidence acquisition／比較條件沒有閉合，不是 typed reference；V3.1 全程 `invalid=0` 且 reference error `=0`。

## 解讀

對 Luna 而言，V3.1 typed references 是可用且穩定的：20 題沒有任何 interface invalid，也沒有 `reference_not_available`、type mismatch 或 evidence eligibility 錯誤。但原始 V3-C 本來也已經是零錯誤，因此 typed refs 沒有額外的錯誤率改善空間。

本次品質結果持平（19/20），token 小幅下降（3.2%）。逐題軌跡顯示，結果變化主要來自 Luna 每次生成不同的搜尋假設與 query，而不是「看到什麼引用格式」直接造成。特別是錯誤題能合法執行所有 refs，卻仍找不到足夠證據，說明剩餘瓶頸在 retrieval policy 與 evidence sufficiency guard。

這是一次 persisted-run paired ablation，不能把一正一負的 flip 或 3.2% token 差異解讀成穩定因果效果。若要確認 typed refs 對 Luna 的真實影響，應固定版本後重複多次，統計平均 judge、calls、tokens 與逐題成功率。

## Artifact

- Config：`configs/agentic_hotpotqa_luna_v3_typed_refs.yaml`
- Inference：`runs/hotpotqa_resample20_seed20260805/luna/v3_typed_refs/c_posthoc_guarded/`
- Judge：`runs/hotpotqa_resample20_seed20260805/luna_judge_luna_v3_typed_refs_c_20260806/`
