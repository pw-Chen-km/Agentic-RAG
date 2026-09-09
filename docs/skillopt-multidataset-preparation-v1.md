# 五個資料集的 SkillOpt 準備（離線版）

本版本只準備資料、計算事後進度及建立未執行的設定。沒有呼叫模型，沒有改搜尋索引、embedding、Agent 動作、Controller、Validator 或動態 schema。

## 三種進度，不混成同一種分數

| 資料 | 進度的意思 | 不能解讀為 |
|---|---|---|
| HotpotQA、2Wiki | 原本指定的句子是否完整呈現 | 已完成推理 |
| MuSiQue | 原本指定段落中，實際呈現的非空白字元比例 | 已找到足以回答的資訊比例 |
| Medical、Novel | 每條標準事實與曾看過片段的最高字詞 F1，再取平均 | 正確證據命中率、語意相同或事實成立 |

每一步分別保存「曾看過」和「合法可引用」的前值、後值與增加量。只採用當輪保存的 PolicyView，不用最後完整記憶回填先前步驟；工具結果若從未進入後續模型輸入，不算已看過。未執行、工具失敗、缺少必要紀錄為不可評估；正常但空白、重複或沒有增加則為零。FINISH 不新增檢索進度。

Medical、Novel 僅使用原始 evidence 列表。answer 不用來計算每步進度；Medical 的 evidence_relations 只封存，不再加入分母。原始文字和位置全部保留，相同正規化文字只計一次並保留所有原始位置。Novel 依原始列位置加上 ID、問題、答案核對，避免重複原始 ID 接錯題。

字詞處理固定為 Unicode NFKC、大小寫统一、保留否定詞及數字的分詞；不做同義詞、詞幹或停用詞處理。檢索長文字按固定標點／換行規則分句，不使用滑動視窗，不拼接不同句子的字詞。分數是 `2 × 多重集合共同詞數 / (事實詞數 + 片段詞數)`，不設成功門檻。

## 取得與後續用途分開

每筆資料以固定 ID 保留第一次取得時間、全部取得路徑、預覽轉全文／可引用時間，以及後來是否作為 EXPAND 起點。

例如 `E1 → EXPAND → C1 → READ`：EXPAND 取得 C1 預覽但沒有提高分數，READ 才取得完整文字，增加量只記在 READ；回看 E1 可以看到這條路徑，但不再給 EXPAND 重複加分。沒有被展開不等於沒有用；也不依 SEARCH query 的字詞巧合推測因果關係。

## 四組輸入

Raw 保留原始紀錄；Result-Driven 整理動作、錯誤與去重文字；Organized + Labels 再加入事後進度；Progress-Abstracted 使用相同進度，但不附檢索正文。

四組共同的訓練標準答案、原始標準事實、問題、SEARCH query、最後答案及「還缺什麼」仍保留。全文來源對應僅供離線評估，不進入共同訓練參考。第四組連最後引用證據的全文與可能回顯正文的自由文字錯誤訊息也不附帶；保留錯誤代碼和執行結果。原始紀錄不刪除。

第三／第四組使用同一份計算結果。第一／第二組沒有額外的 GT 進度標籤。驗證／測試不給分析模型標準答案、標準證據或衍生進度；Target Agent 所有階段都看不到 GT。

## 切分與來源檢查

使用全部現有題目，目標 train/validation/test 為 20/20/60。同問題不同列不可跨集合；歷史用途優先於比例，明確衝突則停止該資料集切分，保留完整問題清單。prepared、executed、analyzed、unknown 分開保存。「沒有分析紀錄」不代表保證從未看過。

2Wiki 只排除已確認座標非法的 9 題；其餘無法映射的標準證據不刪除、不縮小分母、不填零。MuSiQue 不以相同標題猜測段落。只有可靠來源加上可核對文字範圍才接受；沒有來源對應的資料集不得宣稱四組已就緒。

HotpotQA 沿用原本的來源 sidecar。已知 Unicode 正規化差異只在可驗證的既有來源上精確處理，保留原始座標和文字，不重新建立索引，不對無來源的其他資料套用模糊比對。

## 入口（以下命令需自行提供實際路徑）

1. `scripts/prepare_multidataset_skillopt.py --spec SPEC.json --output NEW_PREPARED_ROOT`：準備全量資料、來源／索引 hash、歷史用途、異常、去重及切分。某資料集受阻仍繼續其他資料；受阻時 exit 2，不代表例外資料被排除。
2. `scripts/write_skillopt_progress_examples.py --output NEW_EXAMPLES_ROOT`：產生五組明確標為人工示意的案例，每組包含真正 renderer 產生的四種輸入、逐步分數、路徑和 hash。不讀取保留測試結果。
3. `scripts/write_multidataset_skillopt_configs.py --help`：建立二十份設定與執行清單；blocked 設定沒有虛構實際題數。
4. `scripts/run_local_qwen_skillopt.py ... --dry-run`：只檢查新格式、全部題數、20/5、模型設定與 hash，不建立模型客戶端。正式執行需另外明確授權；本版本不自動執行。
5. `scripts/run_prepared_skillopt_test.py --help`：訓練完成後的獨立測試入口，預設只檢查。必須該資料集四組全部完成並封存後才可 `--execute`，Initial 也不能提前測試。

新資料使用 split schema 3.0、reflection schema v4。舊資料格式仍可讀取，但不能將旧訓練輸出接續成新格式。

每批 20 題使用同版 Skill，每次分析最多 5 題，一批完成至多更新一次；最後不足 20 題不丟棄。設定只有一個 epoch，新訓練中 `eval_test=False`。候選仍使用原有 validation gate，允許不採用任何修改。

## 離線驗收與限制

執行 `pytest -m 'not real_models and not openai_smoke and not ollama_smoke and not skillopt_smoke'`。測試使用假的工具／模型或人工紀錄，不是新的答案評測。測试涵蓋重複、預覽轉 READ、延後呈現、直接 EXPAND 後 READ、不可評估、同字不同意思、重複 ID、第四組正文排除及第三／第四數值一致。

字詞重疊忽略語序，人物關係顛倒仍可能高分；同義改寫則可能低分。Medical 假設病人與 Novel 角色創作可能含原始文件沒有的情境。保留創作題，之後按題型報告，不拿此分數代替最終答案品質，也不因低分刪題。

真實就緒狀態以每次輸出的 `readiness.json`、`history_report.json`、`mapping_gaps.jsonl` 和總報告為準，不以本說明文件推定資料已全部完成。
