# 四組共用的精簡輸入格式

2026-09-06；輸入版本 `agentic-rag-skillopt-reflection-v5`，送給 Optimizer 的格式為 `agentic-rag-skillopt-compact-v1`。

## 修改內容

Raw、Result-Driven、Organized + Labels、Progress-Abstracted 都使用同一個轉換函式。第三組仍先使用第二組的證據整理，再附上原本的進度數值。

1. 每一步的合法動作清單與預算若完全相同，只保留一份，其他位置明確指向同一步的欄位。如果兩份值不同，兩份都保留。
2. 長資料編號在對照表中保留一次，其餘結構化欄位使用短編號。可以一對一確認的 E/S/C 編號沿用；其他編號用 U 編號對照。U 編號不是新的可執行動作 reference。對照表包含整題紀錄的編號，不代表每一步都能使用它們。
3. 重複的欄位名稱改成表格欄名，每一列保留原始順序與值。不同欄位組合使用各自的欄名清單，不把「欄位不存在」變成 null。JSON 外部排版空白減少，但字串內的空白不改。

例如原本每一列都寫 `ref`、`stable_id`、`node_type`、`has_been_read` 等欄名，現在寫一次欄名，再逐列提供對應值。實際每步的可見、已讀、可引用狀態仍完整存在。

## 資訊保留與版本紀錄

`reflection_conversation.json` 繼續保存展開的完整查核紀錄；`adapter.py` 在產生 native `conversation.json` 時套用精簡格式。這才是 native analyst 真正讀取的版本。

- Raw 的累積輸入 State、原始工具紀錄、重複搜尋、重複正文與順序都保留。
- 第二、第三組的去重證據文字、首次取得順序及所有取得路徑保留。
- 第三、第四組共用原本的進度計算，數值不因格式改動而改變。
- 第四組維持不含檢索正文，原本共同允許的問題、標準參考、缺少資訊及最後答案仍保留。
- 資料編號替換只作用於結構化 ID 欄位，不替換 query、正文、答案、缺少資訊或標準事實中的文字。
- `expand_reflection_input()` 可以將精簡格式完整還原；測試直接比较還原結果與原始物件。

Manifest 記錄展開紀錄與精簡物件各自的 hash；完整 native analyst request 的 hash、實際 token 數與分組仍由既有 `reflection_groups.json` 記錄。Integration contract 記錄新格式、程式 hash 與核心指令 hash。

舊版本可繼續查閱，live training 拒絕用新格式直接接續舊輸出目錄。舊結果與紀錄不覆蓋，尚未更新舊排程的版本合約。

## 實測

使用 Medical 已保存的 Raw 40 份與 Result-Driven 40 份紀錄，共 80 份 trajectory。每一份產生四種呈現，共 320 份完整 native 單題 analyst input。這些是格式檢查，不是 320 道新題目。

使用 AMD 上原本固定 digest 的 Qwen 3.6 chat template 和 tokenizer 實際計數。檢查包含完整 native 成功／失敗分析指令、原始 Skill、共同標準參考、Target prompts 與整題紀錄，沒有呼叫模型生成新答案或 Skill patch。

| 原本停止的紀錄 | 修改前 input tokens | 修改後 input tokens |
|---|---:|---:|
| Raw `medical:benchmark_exact:q:000207` | 259,034 | 132,262 |
| Result-Driven `medical:benchmark_exact:q:001172` | 257,104 | 112,152 |

目前每次完整輸入上限為 244,736 tokens，另保留 16,384 輸出 tokens 與 1,024 安全餘量，合計 262,144。

320 份中有 319 份可放入模型。仍超長的是 Raw 原始 rollout 的 `medical:benchmark_exact:q:000644`，改成 Raw 格式後仍有 **265,805 input tokens**。其 12 筆逐步紀錄保留了反覆呈現的完整 State；單是這些 State 在精簡 JSON 中仍約有 52.9 萬字元。

這份紀錄不能宣稱已解決。既有「最多 5 題，太長拆成更少題」無法再拆小於 1 題；原本的拒絕超長保護保持有效。處理同一題內的分段分析或移除累積 State，均需要額外界定實驗輸入，這次沒有套用。

完整測試：624 通過、0 失敗、0 跳過。320 份轉換皆可還原，原始檔案 hash 不變，第三／第四組進度與共同背景相同。這只證明已保存紀錄的格式與長度，不能保證未來所有題目都不會超長。

JJ 查核輸出：

`/home/jj/Large_Space_A/chiu/PW/Agentic-RAG-data/skillopt_reflection_compact_v1_20260906/final_validation/report.json`

本次尚未恢復正式訓練；剩餘超長紀錄與新舊版本續接仍需要處理。
