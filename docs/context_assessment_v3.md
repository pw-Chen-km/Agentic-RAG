# Context 與 assessment 比較：實作與重測

> 本文件保留 v3 native-tool workflow 的診斷紀錄。v3 發現的兩種 protocol error
> 已由 v4 single-decision protocol 修正；目前候選設計與重測結果見
> [constrained_single_decision_v4.md](constrained_single_decision_v4.md)。

## 實作

`agent.require_evidence_assessment` 預設 `true`。開啟時，各工具和 Finish 共用 `assessment.supported_facts`（最多五項）、`assessment.missing_information`（最多三項）。同次呼叫產生 assessment 與 action，不增加 policy call。關閉時工具 schema 不提供 assessment，decision 記錄為 `null`，狀態為 `not_requested`；要求 assessment 但解析失敗則記為 `unavailable`。

每輪分為：上一份 assessment（含輪次與尚未考慮後續結果的說明）、上一輪 action/result、新來源全文、之前的來源全文、可用 entity 名稱卡、早期歷史、剩餘預算。Assessment 是模型的判斷，不作來源證據；可以修改前次判斷，budget finalize 也可保留尚未解決的缺口。

重複操作明確回覆「未執行、沒有新增來源」，後端失敗則回覆「未完成」。工具成功不代表問題已回答。歷史只保留工具、query/entity、執行狀態及新增來源 references。原始錯誤保留於 artifact。

全文去重依來源 ID。完整 passage 取代其中獨立 sentence 的全文，但曾經可用的 sentence labels 仍顯示在 passage 位置說明中，並繼續可引用。新舊區塊只呈現各 unit 一次；passage 包含先前已見 sentence 時，只把未見 spans 計為新增證據。

## 每輪新增紀錄

- `assessment_status`：provided / not_requested / unavailable。
- `context_audit.previous_assessment`：前一次可解析 assessment 與來源輪次。
- `context_audit.newly_visible_source_spans`：本輪 Policy input 首次真正可見的來源 spans。
- `context_audit.new_section_references` / `old_section_references`：本輪新舊全文分區。
- `context_audit.output_delivery`：當輪工具回傳的 references 與新投影的來源 spans；不等同於當輪 Policy 已看見。
- 原有 messages、schema、provider usage、source mapping 與 raw output 繼續保存。

Renderer 更新為 `sectioned-context-v3`；manifest 納入 context 組裝、renderer、紀錄程式的 hash，以及 assessment 設定、工具 protocol hash、target prompt digest。版本或設定改變時拒絕 resume。

## 重測

JJ 工作區：`/home/jj/PW/agenticRAG-interface-study-v2-20260922-smoke`。

使用 `scripts/run_context_comparison.py` 依序跑 `stage1`（assessment off）與 `stage2`（on）。各階段 HotpotQA、Novel、Medical 各取來源第一題，七種配置 C0/C1/C2/C3/C5/C4/A1，共 42 episodes。Seed 為 20260805，配置執行順序一致。模型為 `qwen3.8:27b-q4_K_M`，embedding 為 `qwen3-embedding:4b`；budget 與解碼設定一致。

最終比較 run 路徑與結果記於下方結果章節。

先前 `context-comparison-v3-20260922T112121Z` 因回傳 references 清單可能暴露 passage 內部 sentence labels 而中止，完整保留作診斷。修正後使用新 run。

`context-comparison-v3-20260922T112309Z` 完成階段一，但階段二模型使用 `Known`/`Needed` 取代 assessment schema 的欄位，被嚴格驗證拒絕。保留失敗 artifacts 後中止；新增兩階段共用的安全格式錯誤回饋，以及只適用 assessment 的正確欄位／陣列型別說明。未加入 JSON fallback 或自動改寫模型參數。修正後 C0 的 live provider probe 成功回傳正確 assessment 和搜尋 query，再開始新的兩階段比較。

分析指令：`python scripts/analyze_context_comparison.py --run <comparison-directory> --output <comparison.json>`。

每一配對報告格式／狀態／執行錯誤、重複操作、最長連續重複、工具使用、query 變化、新來源數量、calls/tokens/time。Query changes 是相鄰已輸出 query 字串不同的次數，不代表改寫語意有效。Assessment 的支持性及 action 是否對應資訊缺口另外做人工閱讀式質性檢查，不宣稱是獨立人類標註或自動 semantic 分數。

一題×七配置只用於 workflow 診斷。兩階段固定先後，wall time 可能受模型暖機影響；先前舊 smoke 另有 prompt 差異，不能用來單獨估計 renderer 的因果效果。完整實驗不自動啟動。

## 最終結果：2026-09-22

42/42 episodes 均以 `finish` 結束，21 組配對完整。本機 88 個 regression tests 通過。實際 artifacts 檢查沒有發現來源 spans 遺失、舊 reference 消失、不可用工具被執行、不可見 entity 被執行、budget 超限或兩階段設定不一致。這是 workflow 檢查通過，不等於已達正式 calibration 的 99% protocol-valid 門檻。

- JJ：`/home/jj/Large_Space_A/chiu/PW/Agentic-RAG-data/interface_study_v2/runs/context-comparison-v3-20260922T113125Z`
- 本機：`runs/context-comparison-v3-20260922T113125Z/`
- 可重算的完整配對數據：該目錄 `comparison.json`；原始輸入、工具定義、來源、輸出、錯誤位於各 stage/dataset/episodes。
- Assessment off/on 使用相同程式碼 snapshot、模型 digest、問題、執行順序、substrate、共同 skill、renderer 與解碼設定；target config 唯一差異為 assessment flag。工具 schema 和必要 assessment 指示隨該 flag 改變。

### 每資料集的操作與成本

每格是七個配置的總和；不作跨資料集研究分數。時間為 episode wall time 加總，不包括部署、載入 substrate 與分析時間。Tokens 是 provider input + output，包含工具 schema 和 assessment。

| Dataset | Assessment | Calls | Protocol invalid | State invalid / duplicate | Backend error | Input tokens | Output tokens | Total tokens | 秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | off | 34 | 1 | 3 | 0 | 90,778 | 1,586 | 92,364 | 101.0 |
| HotpotQA | on | 28 | 1 | 0 | 0 | 64,724 | 3,100 | 67,824 | 128.5 |
| Novel | off | 21 | 0 | 0 | 0 | 144,696 | 1,044 | 145,740 | 99.5 |
| Novel | on | 14 | 0 | 0 | 0 | 70,700 | 1,758 | 72,458 | 86.7 |
| Medical | off | 14 | 0 | 0 | 0 | 56,127 | 683 | 56,810 | 50.9 |
| Medical | on | 14 | 0 | 0 | 0 | 60,287 | 1,536 | 61,823 | 76.6 |

不能解讀成 assessment 必然節省時間。HotpotQA 的 calls/tokens 減少，但總時間增加；Medical 的檢索行為未變，assessment 增加生成量及時間。Novel 在第一次搜尋後即辨認到答案，省去原本的第二次搜尋。

### 全部 21 組配對

以下皆為 off → on。來源數是截至最後 Policy input 真正見過的 unique sentence/span 數，不是有用證據或 gold support 的數量。每題新來源數、query 序列、實際工具、最長連續重複與 tokens/time 差值完整保存於 comparison.json。

| Dataset | Condition | Calls | Duplicate | 來源數 | Tokens |
|---|---|---|---|---|---|
| HotpotQA | C0 | 4 → 5 | 0 → 0 | 32 → 49 | 6,308 → 10,593 |
| HotpotQA | C1 | 3 → 4 | 0 → 0 | 32 → 52 | 4,935 → 9,529 |
| HotpotQA | C2 | 5 → 4 | 1 → 0 | 22 → 20 | 12,344 → 9,494 |
| HotpotQA | C3 | 4 → 4 | 0 → 0 | 20 → 18 | 9,281 → 9,276 |
| HotpotQA | C4 | 4 → 4 | 0 → 0 | 22 → 33 | 8,966 → 11,326 |
| HotpotQA | C5 | 4 → 4 | 0 → 0 | 37 → 20 | 10,711 → 10,068 |
| HotpotQA | A1 | 10 → 3 | 2 → 0 | 54 → 36 | 39,819 → 7,538 |
| Novel | C0 | 3 → 2 | 0 → 0 | 219 → 168 | 16,013 → 8,509 |
| Novel | C1 | 3 → 2 | 0 → 0 | 183 → 168 | 15,283 → 8,783 |
| Novel | C2 | 3 → 2 | 0 → 0 | 362 → 168 | 27,697 → 11,043 |
| Novel | C3 | 3 → 2 | 0 → 0 | 435 → 168 | 28,089 → 11,064 |
| Novel | C4 | 3 → 2 | 0 → 0 | 183 → 168 | 20,205 → 11,401 |
| Novel | C5 | 3 → 2 | 0 → 0 | 183 → 168 | 20,197 → 11,362 |
| Novel | A1 | 3 → 2 | 0 → 0 | 183 → 168 | 18,256 → 10,296 |
| Medical | C0 | 2 → 2 | 0 → 0 | 262 → 262 | 7,107 → 7,757 |
| Medical | C1 | 2 → 2 | 0 → 0 | 262 → 262 | 7,289 → 8,050 |
| Medical | C2 | 2 → 2 | 0 → 0 | 262 → 262 | 8,489 → 9,119 |
| Medical | C3 | 2 → 2 | 0 → 0 | 262 → 262 | 8,488 → 9,114 |
| Medical | C4 | 2 → 2 | 0 → 0 | 262 → 262 | 8,674 → 9,484 |
| Medical | C5 | 2 → 2 | 0 → 0 | 262 → 262 | 8,675 → 9,445 |
| Medical | A1 | 2 → 2 | 0 → 0 | 262 → 262 | 8,088 → 8,854 |

### 錯誤與工具使用

Off 共 69 次 policy calls，1 次 protocol invalid：HotpotQA C0 同輪輸出兩次搜尋，被拒絕；其後恢復。另有 HotpotQA A1 兩次、C2 一次 duplicate，最長連續 duplicate 都只有一次。

On 共 56 次 policy calls，1 次 protocol invalid：HotpotQA C0 把 assessment object 輸出成字串，被拒絕；下一輪恢復。沒有 duplicate、state-invalid 或 backend execution error。沒有額外免費 retry，失敗均消耗 decision。有效的 55 次 calls 都有 assessment，失敗一次記為 unavailable；off 全部記 not_requested。

以整個 smoke 的 calls 計算 protocol-valid，off 為 68/69（98.55%），on 為 55/56（98.21%）；按 C0 條件單獨計算會更低。**因此不能宣稱 99% 門檻已通過，更不能用所有 episodes 最後都成功掩蓋格式失敗。** 本次沒有靜默修正參數或 raw JSON fallback。

實際成功執行的 retrieval tools 如下，不含 Finish，也不把被拒絕的 tool call 算成執行。

| Dataset / phase | Passage search | Sentence search | Entity→passage | Entity→sentence |
|---|---:|---:|---:|---:|
| HotpotQA off | 14 | 3 | 3 | 3 |
| HotpotQA on | 13 | 0 | 4 | 3 |
| Novel off | 10 | 4 | 0 | 0 |
| Novel on | 7 | 0 | 0 | 0 |
| Medical off | 7 | 0 | 0 | 0 |
| Medical on | 7 | 0 | 0 | 0 |

On 的 HotpotQA C2/C3/C4/C5 都實際選過 entity navigation，證明這些自然軌跡不是「工具完全不會使用」。但 on 這三題都未選 sentence search，不能据此判定該工具無效或模型不會用。需要獨立工具操作 calibration，而非透過策略提示強迫主實驗使用。

### Assessment 逐題質性檢查

以下是本次助理閱讀 assessment、相同輪次的可見來源和 action 後的判讀，**不是獨立人類標註，也不是 semantic judge 分數**。檢查涵蓋 on 的 21 個 episodes。所有第一輪 supported_facts 都為空，沒有把未檢索的常識填成來源證據。

| Dataset | Condition | 支持性與下一步是否對應缺口 |
|---|---|---|
| HotpotQA | C0 | 先確認演員名單，再查演員知名作品；最後 Chris Evans 的描述有可見來源。第二輪 assessment 格式錯誤，不能判讀為有效 assessment。 |
| HotpotQA | C1 | 名單→Kate Bosworth→Chris Evans 的 query 對應「因何知名」缺口；Captain America 在 query 中作假設，不應當作當時已證實的事實。 |
| HotpotQA | C2 | 從 Chris Evans 改追 Kate Bosworth 符合剩餘缺口；但將「出演 Straw Dogs」寫成「因該片知名」超出直接來源措辭。Finish 的 missing_information 以一句「不需要更多資訊」代替空陣列，語意不夠一致。 |
| HotpotQA | C3 | Entity 選擇對應演員作品缺口；assessment 保留較準確的「出演 Straw Dogs」，但 Finish 仍改成「因該片知名」。有 assessment 並不保證答案不過度推論。 |
| HotpotQA | C4 | Entity navigation 沒有新文字後仍保留缺口，改用 passage search；新來源支持 Chris Evans 的知名角色後 Finish。 |
| HotpotQA | C5 | 缺口持續到找到 Kate Bosworth 另一部作品；assessment 的「也出演」有支持，Finish 的「known for」仍較強。 |
| HotpotQA | A1 | 名單→Chris Evans→知名角色，兩段來源支持 assessment，沒有重複搜尋。 |
| Novel | C0 | 第一次 passage 結果 C4 已有 Erica vagans / Cornish heath 對應；標記缺口完成後 Finish，符合可見來源。 |
| Novel | C1 | 同上；沒有再選 sentence search，這是停止選擇，不是功能不可用。 |
| Novel | C2 | 同上；已有直接答案，未導航合理，不能由這題檢驗導航能力。 |
| Novel | C3 | 同上；assessment 與 C4 原文一致。 |
| Novel | C4 | 同上；assessment 較長且重複引用原句，增加生成成本，非必需的新證據。 |
| Novel | C5 | 同上；答案識別和 Finish 對應已解決的缺口。 |
| Novel | A1 | 同上；assessment 較長，但命名對應有來源支持。 |
| Medical | C0 | C1 明確支持 basal cell carcinoma 是最常見類型，C2 支持 squamous cell 為第二；Finish 合理。 |
| Medical | C1 | 同上；reference C1/C2 對應可見來源。 |
| Medical | C2 | 只摘要主要答案，來源足夠，沒有額外檢索需求。 |
| Medical | C3 | 同上；簡短 assessment 後 Finish。 |
| Medical | C4 | 摘要第一、第二常見類型，均可在可見來源找到。 |
| Medical | C5 | 同上；缺口清空及 Finish 與來源相符。 |
| Medical | A1 | 同上；沒有因 annotation 而增加導航需求。 |

HotpotQA 的 Chris Evans/Kate Bosworth 軌跡說明，assessment 可讓缺口與下一步清楚可見，但它不是事實驗證器。Novel/Medical 是第一次搜尋就能回答的題目，因此不能證明多步推理能力或一般性的工具偏好。Novel 的原有 substrate 還可看到 `_erica` 與 `vagans_` 被分為兩個 sentence；passage 全文保留了關係，但正式比較 sentence granularity 前值得另外審查既有斷句品質，本次未重建或改動 substrate。

### 可採取的決策

本次指定的 context 修正和兩階段 smoke 已完成；完整來源及既有 references 沒有被 assessment 取代。Assessment on 的這批軌跡沒有重複操作，且 Novel 少一次搜尋，但不能以三題推論為穩定改善，也不能把 off/on 差異完全歸因於 assessment（仍有模型輸出變異與固定階段順序）。

正式 run 暫不開始。先處理並重新校準原生工具輸出中的「同輪多 call」及「assessment object 變字串」，保留嚴格驗證與失敗成本；再以更多、事先固定的多步問題檢查有效率，而不是在這三題上持續調到零錯誤。這次沒有執行獨立 semantic judge，也不把 deterministic answer/evidence 分數當成本次 workflow 的通過條件。
