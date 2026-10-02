# Action guide v5

七個 study conditions 共用一份 `skills/interface_study.md`。System message 依序呈現
`TASK`、`POLICY`、`HOW SEARCH WORKS`、`AVAILABLE ACTIONS THIS TURN`、
`REFERENCE RULES`、`DECISION FORMAT`。原始問題與來源、上一輪結果、新舊證據等
context 在後續 user messages。Action 清單依當輪可用能力生成；還沒有可見 E# 時，
follow action 不在清單或 decision schema 中。

共同機制只解釋一次：query 會轉成代表語意的 embedding，與儲存的 passage 或 sentence
embeddings 比較；不要求字面詞重合。每張 action card 使用 exact action name 與參數：

- `find_passages(query)`：在整個 collection 的 passages 中搜尋，回傳完整段落與周圍
  句子；通常帶入的文字比單句多。
- `find_sentences(query)`：在整個 collection 的 sentences 中搜尋，回傳單句；通常較短，
  但可能缺少周圍上下文。
- `follow_entity_to_passages(entity_ref, query)`：沿目前可見 E# 的同一 entity mention links
  找 passages，只在相連候選中依 query 語意排序；query 為 null 時使用原問題。
- `follow_entity_to_sentences(entity_ref, query)`：相同局部路徑，落點是 sentences。
- `finish(answer, evidence_refs)`：結束 episode，沒有新檢索；只可引用可見 C#/S#。

Prompt 不指定「何時應使用」任何 action，也不暴露 backend 的 `top_k`。Entity follow 的
`query` 只改變已連到候選的順序，不能使沒有同一 entity mention link 的來源進入候選。
這是 agent 選 global search 或 local navigation 時必須知道的範圍限制。

`src/agentic_rag/agent/interface_action_catalog.py` 是 prompt card 與 provider schema action
description 的共同來源。`AvailableActionSpace` 控制實際可用集合；測試逐 condition 比較
prompt 列出的 exact action names 和 JSON Schema，並檢查首輪、E# 已顯示、A1 annotation-only、
assessment off 的情況。`InterfaceContract`、validator 與 router 仍各自執行後端合法性檢查。

此變更只修改 agent 可見的 action 說明與 prompt 版本。它本身可能改變模型的 action
選擇，因此必須建立新的 run；v4 smoke 只可作 workflow 診斷，不能與 v5 當作單一
因果對照。正式 run manifest 的 renderer hash、skill hash 與 target prompt digest 均
包含 v5 檔案，拒絕接續舊 prompt 版本。

## JJ 驗證

程式已同步到 `/home/jj/PW/agenticRAG-interface-study-v2-20260922-smoke`，並在 JJ
通過 97 個 regression tests。單題 HotpotQA × C4 live smoke 保存在：

```text
/home/jj/Large_Space_A/chiu/PW/Agentic-RAG-data/interface_study_v2/runs/v5-action-guide-c4-smoke-20260922
```

該 episode 有 terminal artifact，10 次 policy decisions 均符合單一 decision schema，
實際執行 passage 與 sentence search，0 次 protocol-invalid。它也有 3 次重複搜尋被
validator 拒絕；最後答案未通過 HotpotQA 的 exact／contain 字面比對，但語意上仍提到
Chris Evans 的超級英雄角色，不能單憑字面指標判定為事實錯誤。因此這一題只驗證新版
prompt 確實送達模型並可執行；
它不能證明較好的 action 選擇或研究效果。本機副本位於
`runs/v5-action-guide-c4-smoke-20260922/`。

Live smoke 之後，共同機制文字做了一處條件隔離修訂：它只提「目前 action 搜尋的文字
單位」，不在 C0 額外提 sentence 索引。最終文字在 JJ 的 C0/C4 initial 與 entity-visible
四次 provider probes 均回傳單一合法 decision；C4 entity-visible probe 選中
`follow_entity_to_sentences`。報表位於
`runs/v5-action-guide-final-calibration-20260922.json`。這個 probe 沒有實際執行
retrieval，也不足以證明 99% protocol-valid rate；正式實驗仍需依預先固定的門檻校準。

最終文字另以相同三題、七配置跑完 21 個完整 episodes，遠端結果位於：

```text
/home/jj/Large_Space_A/chiu/PW/Agentic-RAG-data/interface_study_v2/runs/v5-action-guide-21-smoke-20260922T210604Z
```

21/21 都有 terminal artifact，65 次 decisions 中沒有 protocol-invalid 或 backend error，
但有 7 次被拒絕的重複操作，均出現在 HotpotQA。相同題目與 substrate 的 v4 smoke 有
44 次 decisions、1 次重複操作。v5 的首輪直接 Finish 從 7 題降為 0 題，entity navigation
從 0 次增至 3 次；target token 總數也從 111,543 增至 204,116。HotpotQA 的
complete-support 與 contain 仍分別是 6/7 與 2/7。每個資料集僅一題，因此這些是
workflow 診斷，不是對 action guide 效果的統計估計。這次沒有執行 semantic judge；
`runs/` 與資料集不納入 Git 提交。
