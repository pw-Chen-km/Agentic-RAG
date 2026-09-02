# SkillOpt 四項修正（2026-09-02）

本版只修正反思輸入與修改建議的篩選。不要求每個建議附題號／步驟，不修改成功、失敗分析或合併指令，也不改 Target Agent、Retriever、Skill 與評分方式。

## 現在的行為

1. **第四組不提供檢索證據全文。** 每步結果和最後引用證據都不附正文；最終引用僅保留 ref、document_id、parent_chunk_id、contained_sentence_ids。問題、搜尋 query、最終答案、訓練用標準答案與標準證據維持原設計。前三組繼續保留檢索文字，原始 episode 完全不改。
2. **第二、第三組按首次取得步驟排序證據。** 同一步按 stable ID 排序。去重文字不刪除取得路徑、READ 時間或能引用的時間。
3. **四組每一步都有相同必要背景。** `decision_context` 包含該次決策的 `missing_information`（缺少決策則為 null）、執行前資料編號與可讀／可引用狀態、該次可用動作、執行前後預算。優先採當次 frozen reference map，其次保存的 Policy input，再其次 state_before；以 `reference_source` 明示來源，不讀取未來狀態來補寫背景。第四組允許「還缺什麼」提及檢索得到的人名，但不加入 supported_facts 摘要或檢索正文。Raw 仍保留完整原始記錄，因此不宣稱四組全部資訊完全一致。
4. **排名真的可以不選任何修改。** 無候選不呼叫模型；任何非空候選池均檢查，包括只有一個候選。`selected_indices: []` 直接保留原 Skill。格式、範圍、重複編號、超過預算或服務錯誤最多嘗試三次；仍失敗就返回空 edits，記為 ranking_error，不取第一個候選。正常選出的修改仍走既有 validation gate。

## 排名整合方式

專案內 `ranking_compat.py` 只在 SkillOpt `train()` 期間暫時替換已安裝套件的兩個函式引用，結束或拋出例外後還原。沒有改套件檔案。修正只針對本實驗的 patch 模式，其他更新模式沿用原行為。

`ranked_edits.json` 的 `ranking_details` 保存狀態、選擇結果、每次回覆與錯誤。`abstained`（明確不選）與 `ranking_error`（無法完成排名）分開記錄。每次模型呼叫的原有內層重試設定為一次，由外層統一限制總嘗試數。

六份核心 prompt 原文不變；執行時的排名要求改成「最多選 L 個，也可全不選」。四組的輸入說明同步更新，解釋共同背景與第四組文字限制。

## 版本與舊結果

- 反思輸入：`agentic-rag-skillopt-reflection-v3`。
- 輸入 manifest：`agentic-rag-skillopt-reflection-manifest-v2`。
- 排名修正：`agentic-rag-skillopt-ranking-v1`。
- 執行整合：`agentic-rag-skillopt-input-ranking-v1`。
- 新訓練輸出會保存 `skillopt_integration.json`，記錄上述版本、六個程式檔與六份 prompt 的 SHA-256；原生 config.json 也記錄版本。
- 偵測到舊訓練輸出，或現有紀錄的版本／程式 hash 不相同時，停止接續並要求新輸出目錄，不覆寫舊資料。

下一輪仍為 rollout 20、reflection 5、accumulation 1、100/50/50。舊四組的 3/3 設定與結果保留不變，不能用它們的輸出目錄接續新版本。

## 驗證範圍

修改前完整測試 117 項通過；新版本完整測試 151 項通過（包含 34 項新增檢查）。測試沒有呼叫 Target、Judge 或 Optimizer 服務。

新增檢查包括：真正交給原生分析者 formatter 的完整輸入不含第四組檢索正文／最終證據全文；允許的標準答案與「還缺什麼」仍保留；四組共同背景一致、不使用未來 references；證據排序與路徑保留；舊輸入拒絕接續；排名正常選取、空選擇、單候選、各種格式錯誤、服務失敗、重試成功；暫時替換還原；原生 validation 拒絕不改善的候選。

另對上一輪四組各 100 條訓練紀錄做唯讀重建檢查，沒有修改任何已保存資料，也沒有重新執行 Agent 或 Judge：

| 組別 | 檢查紀錄數 | 舊輸入含最終證據全文 | 新輸入含最終證據全文 |
|---|---:|---:|---:|
| Raw | 100 | 100 | 100 |
| Result-Driven | 100 | 100 | 100 |
| Organized + Supporting Labels | 100 | 100 | 100 |
| Progress-Abstracted | 100 | 99 | 0 |

第二、第三組的新證據列表均按取得時間排序，逐筆資料與取得路徑和原紀錄一致，只改順序。這是格式／相容性檢查，不是新推理結果，也不能用來宣稱表現已改善。

本次不啟動新訓練、不跑新的 100 題、不 commit、不 push。
