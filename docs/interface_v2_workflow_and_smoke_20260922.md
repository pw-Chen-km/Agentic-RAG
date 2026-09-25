# Agentic RAG Interface v2：實際流程、七個配置與 smoke 記錄

此文件依 2026-09-22 的程式實作整理；不是把早期計畫視為已完成的功能。JJ smoke 使用 `qwen3.8:27b-q4_K_M`，與 Brev 計畫的 `Qwen/Qwen3.8-27B-FP8` 不同。這次結果只能證明目前 JJ 模型與執行環境的行為。

## 1. 一題如何執行

1. Runner 讀取題目與 evaluator sidecar，但只把問題文字和資料集的 retrieval scope 傳給 agent。Gold answer、supporting facts 留在 evaluator。
2. 建立該 condition 的工具清單。第一輪沒有已看見的 entity，因此 C2/C3/C4/C5 的 follow 工具尚未出現；`finish` 從第一輪就存在。
3. 模型收到 system prompt、問題、目前完整可見的來源文字、entity 名稱卡（若該配置有）、歷史 action 摘要、剩餘 decisions 與 retrieval token estimate。
4. 模型回傳一個 native tool call。Provider 讀取 tool name/arguments；後端再檢查工具是否合法、reference 是否可見、是否重複、scope 是否相符。
5. 搜尋或 navigation 取得最多五個結果。完整 passage 或 sentence 放入記憶，接著才從可見文字建立 entity references。
6. 下一輪重建累積 observation。已取得完整 passage 時，其中的 sentence 不再另外重複列全文。過往 action 保留摘要；原生對話只保留最近一組 assistant call/tool result。
7. 模型可繼續搜尋、follow entity 或 `finish(answer, evidence_refs)`。Finish 的 references 由後端解析，連回來源文字。
8. 每題最多 15 次正常 decision；耗盡後最多額外一次只提供 finish 的呼叫。Episode 結束後寫入 artifacts，evaluator 才比對 gold。

「全文只呈現一次」指同一輪累積 observation 不重複列相同 unit，不是來源文字此後不再送給模型。累積記憶會在後續呼叫再次送出，成本由 provider input tokens 反映。

Native tools 不代表零格式錯誤。Ollama 收到 tools，仍可能回傳零個或多個 call；目前 provider 要求恰好一個，失敗會記錄為 invalid，沒有 raw JSON fallback。

## 2. 所有配置共同設定

| 項目 | JJ smoke 的實際設定 |
|---|---|
| Policy model | `qwen3.8:27b-q4_K_M`，Ollama `127.0.0.1:11440` |
| Embedding | `qwen3-embedding:4b`，2560 維 |
| Temperature / thinking | `0` / `false` |
| Context / 最大輸出 | 32768 / 2048 tokens |
| Policy budget | 15 decisions + 最多 1 次 finish-only |
| Invalid / duplicate / empty | 仍消耗一次正常 decision；不是免費 retry |
| Provider retry | transient transport error 最多 retry 2 次，與 invalid action 不同 |
| Retrieval budget | 12000，程式的文字 token estimate，不等於 Qwen tokenizer 計數 |
| 每次取回 | 最多 5 個 unit；剩餘 budget 不足時僅保留能完整放入的前綴結果 |
| Timeout | request 600 秒；episode 3600 秒（episode timeout 在輪次間檢查） |
| 排序 | 預計算向量，normalized dot product，穩定 ID 排解同分 |
| Query encoding | 同一 RankingService 快取相同 query；不重新 encode 所有候選 |
| Scope | 每資料集一個 global scope；三套 corpus 分開 |
| 未開放工具 | keyword/BM25、global entity search、entity co-occurrence、adjacency、獨立 READ |

「Global sentence search」就是在該資料集的整個 corpus 中搜尋 sentence；不是只搜尋目前 passage。Local navigation 先選已顯示 entity，將候選限制成含該 entity 的 passages/sentences，再以 query 排序。它是一跳 entity→文字連結，不是關係 triple 推理，也沒有距離最近的地理或文件位置限制。

## 3. 七個配置

| Condition | 從整個 corpus 找 passages | 從整個 corpus 找 sentences | 顯示 entity 名稱卡 | 看見 entity 後可執行 |
|---|---|---|---|---|
| C0 | 是 | 否 | 否 | 無 follow |
| C1 | 是 | 是 | 否 | 無 follow |
| C2 | 是 | 否 | 是 | entity → 完整 passages |
| C3 | 是 | 否 | 是 | entity → 完整 sentences |
| C5 | 是 | 是 | 是 | entity → 完整 passages |
| C4 | 是 | 是 | 是 | entity → 完整 sentences |
| A1 | 是 | 是 | 是 | 無 follow；annotation-only |

每個配置都有 `finish`。C0 可改寫 query 多次找 passages；C1 另外可選 sentence 級搜尋。C2/C3 必須先取得來源文字，才有可 follow 的 entity；C5/C4 則有 passage/sentence 兩種全域入口，之後可以接局部導航。

A1 的 annotation 是離線 entity extraction 對已可見文字產生的名稱卡，例如 `E1 — Marie Curie`。它不是摘要、preview、定義或新的背景知識。A1 可以使用名稱寫下一個 query，但不能呼叫 follow。

主要比較可回答：

- C1 − C0：增加全域 sentence search 的效果。
- C3 − C2、C4 − C5：local landing sentence vs passage；各在無/有 sentence search 下比較。
- C4 − C3、C5 − C2：已有 local navigation 時再增加 sentence search。
- `(C4 − C3) − (C1 − C0)`：sentence landing 設定下的 global/local interaction；passage landing 對應 `(C5 − C2) − (C1 − C0)`。
- A1 − C1：顯示 entity 名稱卡的效果。
- C4 − A1、C5 − A1：已有 annotation 和 sentence search 時，加上可執行 navigation 的效果。

C2/C3 − C0 同時增加 annotation 與 navigation，不能寫成「純 navigation 效果」。目前沒有不含 sentence search 的 annotation-only A0，因此該情況尚不能分離兩者。

## 4. Skill 與模型真正收到的 prompt

更新：本機 `InterfaceContract.protocol` 已改成一般語言，移除下列舊版中的 `DENSE -> CHUNK`、`ENTITY_MENTIONED_IN_*` 等內部術語。七配置由同一份程式產生說明，只依能力增加對應文字；沒有加入工具偏好或行動順序。新增七配置回歸測試後共 60 tests passed。尚未同步 JJ 或執行新的 live run；本節以下保留的是上述 21 episodes 真正使用的舊版 prompt，不能把更新後文字誤當成該次模型輸入。Feedback 與 history 本次未更改。

更新後，每一個 native tool call 也共用一個必要的 `assessment` 物件：`supported_facts` 最多五項，記錄目前顯示來源已支持的事實；`missing_information` 最多三項，記錄回答問題仍需要的資訊。下一輪 context 會顯示最近一次可解析的 assessment；完整 assessment 也保存在 trajectory decision 中。這是簡短的 evidence-state 摘要，不要求完整推理過程，也不影響 backend 執行哪一個 retrieval action。這項更新同樣尚未同步 JJ 或進行新的 live run。

七個配置都載入同一個 `skills/interface_study.md`，原文如下：

```text
# Retrieval interface study policy

Answer the question using only information made available by the retrieval interface.

The current interface provides a set of retrieval actions. The available actions and their input requirements are supplied with the current turn. Each action description states what information that action can return.

At each step, choose one available retrieval action or finish the task. The interface does not prescribe an action order, and no retrieval action is preferred by default.

Use only references that appear in the current observation. Do not invent references, entity names, source text, or unavailable actions. For entity navigation, use only an entity reference that is displayed with its name in the current observation.

You may finish when the available evidence is sufficient to answer the question. The answer must be supported by text that was actually shown to you.
```

System prompt 不只有 skill，前面還會加上 `InterfaceContract.protocol`：

```text
You answer the question using information made available by this retrieval interface.
Choose one available retrieval action or finish at each step. The interface does not prescribe an action order, and no retrieval action is preferred by default.
Available search capabilities: <SEARCH>.
Entity display: <ANNOTATION>. Entity continuation capabilities: <FOLLOW>.
Passage and sentence results contain the complete text returned by that action.
Use only references shown in the current observation. Never invent references, entity names, hidden evidence, or unavailable actions.

Current retrieval skill:
<the shared skill above>
```

| Condition | SEARCH | ANNOTATION | FOLLOW |
|---|---|---|---|
| C0 | DENSE -> CHUNK | no entity annotations | none |
| C1 | DENSE -> CHUNK, DENSE -> SENTENCE | no entity annotations | none |
| C2 | DENSE -> CHUNK | visible entity names | ENTITY_MENTIONED_IN_CHUNK |
| C3 | DENSE -> CHUNK | visible entity names | ENTITY_MENTIONED_IN_SENTENCE |
| C5 | DENSE -> CHUNK, DENSE -> SENTENCE | visible entity names | ENTITY_MENTIONED_IN_CHUNK |
| C4 | DENSE -> CHUNK, DENSE -> SENTENCE | visible entity names | ENTITY_MENTIONED_IN_SENTENCE |
| A1 | DENSE -> CHUNK, DENSE -> SENTENCE | visible entity names | none |

這裡仍有 `DENSE -> CHUNK` 等內部術語，尚未完全符合先前要求的「完全用一般語言解釋」。Skill 沒有指定順序，但不能宣稱整套 prompt 已經證明毫無行為偏差。工具排序也固定為 passage、sentence、follow、finish；尚未做 tool-order robustness。

Native tool definition 是第三部分，模型也會看到這些實際 description：

```text
find_passages(query)
Search the collection and return complete passages related to a query.

find_sentences(query)
Search the collection and return complete sentences related to a query.

follow_entity_to_passages(entity_ref, query?)
Return up to five complete passages that mention the displayed name. Select its entity reference from the visible name list. An optional query ranks these passages; null uses the original question.

follow_entity_to_sentences(entity_ref, query?)
Return up to five complete sentences that mention the displayed name. Select its entity reference from the visible name list. An optional query ranks these sentences; null uses the original question.

finish(answer, evidence_refs)
Return the answer supported by references shown in the current observation.
```

`entity_ref` enum 限制合法 E#，語意來自 observation 的 `E# — name`。重名才補位置。沒有另外給示範 retrieval trajectory。

## 5. 目前使用的 scripts

| Script | 實際用途 |
|---|---|
| `run_jj_v2_smoke.py` | 本次新增：固定 JJ model/config，建立新 run；三資料集依序執行 static registry check 與各七個 episodes |
| `run_interface_study.py` | 正式 target runner；載入資料與 condition、固定 seed 打散 question×condition schedule、執行 agent、寫 progress 和 summary |
| `calibrate_interface.py` | 檢查初始與 entity-visible 工具清單；`--live` 另做 provider probes，但不執行 retriever，因此不能提供 execution-success rate |
| `audit_v2_smoke.py` | 本次新增：檢查 artifacts、budget、available/selected tools、usage、實際 messages 與 visible spans 一致性，統計 tool uptake |
| `validate_v2_substrate.py` | 檢查 embedding model、dimension、三種 dense indexes 可載入 |
| `evaluate_semantics.py` | 用既有 episodes 做獨立 semantic judge，不重跑 target agent |
| `analyze_interface_study.py` | 從 progress 重算 condition aggregates、有限的 trajectory cases 和 semantic means |
| `build_interface_substrate.py` / `build_graphrag_substrate.py` | 建立 HotpotQA / Novel、Medical substrate |

本次每個資料集用來源檔第一題（`--limit 1`），同題跑七種配置；`seed=20260805` 決定執行順序。這不是 21 題抽樣，也不是評估研究效果。此 seed 目前控制 schedule，不代表已傳給 Ollama 作 decoding seed。

每個資料集的實際完整命令保存成 `<dataset>_command.json`。Target config 與 model digests 保存在 smoke 根目錄。

## 6. 記錄哪些東西，以及用途

| 記錄 | 位置 | 為什麼記錄 |
|---|---|---|
| 當輪實際 messages | episode.json → trajectory.messages | 確認模型真正看見什麼；檢查 prompt、來源與 gold leakage |
| available tools 和 schema digest | trajectory.tool_definitions / available_action_space / decision_schema_sha256 | 分清「工具沒提供」與「提供了但模型沒選」 |
| native call、raw response、arguments | provider_metadata.raw_tool_calls / raw_provider_output；decision / resolved_decision | 檢查模型選擇與 provider/parser 是否改變選擇 |
| entity E# 對 stable ID | context_reference_map、state、messages 名稱卡 | 確認 ref 來自已看見文字，可追溯到 source |
| Policy-input spans | trajectory.visible_source_spans | 判斷 evidence 是否真的送進模型，以及在哪一次 decision 前已取得 |
| 新回傳文字與 annotation audit | observation.results / metadata.projected_source_spans / entity_mention_audit | 將本輪檢索結果與下一輪 Policy input 區分 |
| validation、錯誤、duplicate | validation_status / validation_error / observation / provider_metadata | 區分輸出協定問題、reference/重複問題、retrieval backend 問題 |
| query、target、navigation source | decision / resolved_decision / observation.metadata | 分析 query 改寫、global/local route 選擇、工具轉換和有效使用比例 |
| provider input/output/total tokens | provider_metadata.provider_*_tokens | 量測真實 target token 成本；未知值在此層為 null |
| retrieval tokens / remaining budget | usage / state_before / state_after | 檢查是否符合共享 budget；這裡的 retrieval tokens 是估計值 |
| Schema / visible payload token estimate | telemetry、provider_metadata | 估計新增工具或名稱卡帶來多少額外 prompt 負擔 |
| ranking counters / wall time | provider_metadata、observation.metadata | 看 encoding/scoring工作量與 retrieval耗時；counter 是服務累積值，不能當獨立每步值直接相加 |
| answer、citation、terminal reason | episode.json | 區分正常 finish、預算終止、provider error 與 runtime error |
| per-question outcome | progress.jsonl | 做同題 paired comparison，也用來確認完成與 resume |
| source/config/skill/renderer/provider hashes | run_manifest.json；smoke_manifest.json | 避免不知情地混合版本；smoke manifest 額外記錄 source code hashes/model digests |

每個 episode 有六個檔案：`episode.json`、`conversation.json`、`target_system_prompt.txt`、`target_user_prompt.txt`、`skill.md`、`effective_config.json`。

`conversation.json` 是 action/feedback 摘要，不能取代完整 messages。沒有啟用 thinking，也沒有將 `assessment` 當成模型內部推理過程。

### 已輸出的指標與用途

| 指標 | 要回答的問題 | 目前注意事項 |
|---|---|---|
| exact / contain | 答案是否與 reference 相同／包含 reference？ | 此 runner 只統一大小寫與空白；不是完整官方 HotpotQA normalization，也不是 semantic correctness |
| support recall | Gold supporting sentences 中，有多少比例進入 Policy input？ | 依可對齊 gold mapping；無 mapping 為 not_evaluable |
| complete support | 一題所需的全部 gold support 是否都被看見？ | 題目層級是 yes/no，資料集層級取比例 |
| complete support cited | 完整證據是否也被答案引用？ | passage citation 的 evaluator 尚需修正，不作本輪結論 |
| first complete-support decision / prefix curve | 到第幾次 decision 已有完整證據？ | 目前以 input 所在輪次記錄；curve 是達標題數，需搭配有效題數才成比例 |
| first complete-support tokens | 取得完整證據的累積 token 成本？ | 現有欄位不是正確 prefix 累積值，須修正後才使用 |
| calls / invalid / retrieval attempts | 哪些配置花更多互動、哪些成本浪費在失敗？ | 嘗試數與成功執行數必須分開；不能只用 terminal error 計算失敗 |
| tool uptake / query sequences | 模型實際用了哪些工具，何時改寫 query 或切換 route？ | 保存原始序列；深度 query reformulation 分析尚未自動完成 |
| provider tokens / time | 代價來自 prompt、輸出、retrieval 還是模型等待？ | target 與 judge 分開，缺值不能當零 |
| semantic correctness / ROUGE-L / coverage / faithfulness / relevance / evidence recall | 字面答案以外的品質、相關性與證據支持如何？ | 獨立 evaluator，可回填；本次沒有呼叫 live semantic judge |

目前可產生 per-condition 描述統計，但不是完整的 paired significance analysis。Smoke 每個資料集只有一題，不能用於研究效果或統計顯著性的結論。

## 7. 這次 smoke 的判定與已知限制

2026-09-22 稽核發現 controller 沒有把 `BuiltPolicyContext.visible_source_spans` 接到每輪紀錄，反而退回 observation 新檢索結果。已修正所有正常、invalid、finalize 與 execution-error 的紀錄路徑，明確保留第一輪的空 spans；七個 condition 的正常/invalid 後續輪次回歸測試通過。修正只影響紀錄，沒有改 retrieval policy。

本次 21 episodes 已全部完成，artifact audit 未發現所檢查的檔案、capability、budget 或 Policy-input span 違規。本機回歸測試為 53 passed。一次自然執行不一定會選每個工具；沒有選 follow 不能被解讀為該 condition 的 follow 已通過 live execution。Static calibration 的 100% 也不能解讀成 LLM 工具操作 100%。

### 實際 smoke 結果

遠端 run：

```text
/home/jj/Large_Space_A/chiu/PW/Agentic-RAG-data/interface_study_v2/runs/live-smoke-v2-20260922T100603Z
```

| 資料集 | 完成 episodes | Policy calls（含 finish） | Protocol invalid | State invalid |
|---|---:|---:|---:|---:|
| HotpotQA | 7/7 | 51 | 0 | 15 |
| Novel | 7/7 | 22 | 0 | 0 |
| Medical | 7/7 | 14 | 0 | 0 |
| 合計 | 21/21 | 87 | 0 | 15 |

15 次 state invalid 全部是同一 action 重複執行被拒絕，不是 JSON 格式錯誤或 backend exception。HotpotQA C0 有 11 次重複相同 passage query，耗盡 15 次正常 decisions，再以第 16 次 finish-only 結束；C2 有 4 次重複已執行的 entity/query 組合。Native tool calling 解決了這次觀察到的格式問題，但没有避免模型陷入重複操作。

程式與 C0 第 6 次呼叫的實際 messages 都顯示：模型有收到歷史 query 和 `duplicate_action` 代碼，但 renderer 移除了人類可讀的錯誤說明，而且把累積來源全文作為最新 tool result 重送。這不是「完全沒有 feedback」，但目前 feedback 不夠直接，是待測的成因；這次單次軌跡不能證明它就是重複操作的唯一原因。

| 工具 | 模型選用次數（含被拒絕者） | 成功執行次數 |
|---|---:|---:|
| find_passages | 44 | 33 |
| find_sentences | 4 | 4 |
| follow_entity_to_passages | 15 | 11 |
| follow_entity_to_sentences | 3 | 3 |
| finish | 21 | 21 |

全部 87 次呼叫皆符合協定格式；72 次合法且成功，包括 21 次 finish。以所有呼叫為分母，成功率為 72/87 = 82.8%；以 retrieval 嘗試為分母則為 51/66 = 77.3%。若只看已通過 validation 的呼叫，backend 成功率為 100%。這三者回答不同問題，不能混用分母來宣稱 calibration 過關。

此輪確認了五種工具至少各有一次真實成功執行，但不代表每個 dataset × condition × tool 都已覆蓋。Medical 七個 episodes 都只有一次 passage search 後 finish；不能拿這一題比較 global/local 使用偏好。

目前不啟動完整實驗。下一步應先查明重複 action 的 feedback、history 呈現與可選 action 是否一致，並對所有配置使用相同修正；不能只替 C0/C2 加策略提示。任何改動需另開 run 重測。21 episodes 的 observed protocol-valid 100% 亦不等於已證明長期可靠度 ≥99%。

正式研究前還需處理或明確界定：

- `evaluate_episode` 的 `first_complete_support_tokens` 目前取該步 token，尚不是 prefix 累積 token。
- `complete_support_cited` 目前的 evaluator 主要從 sentence refs 判定，完整 passage citation 的 contained sentences 尚需正確納入。
- `candidate_retrieved` 目前是保留結果數，不等於 backend 全部候選；`text_projected/text_seen/evidence_eligible` 的分層仍需再核對。
- provider raw metadata 的 unknown usage 可為 null，但較舊 aggregate 路徑會把缺值當 0；正式成本分析應使用可用性旗標與有效分母。
- 完整 `condition × question_type`、paired CI/p-values 尚不由現有 `analyze_interface_study.py` 全部產生。
- semantic judge 不是本次 21 target episodes 的一部分；需另外驗證 judge 的 context 與錯誤處理。Target smoke 過關不代表 semantic evaluation 已完成。
- 跨 episode query cache、冷啟動/模型換載與固定 tool order 會影響時間或使用傾向，正式成本比較需控制並報告。
- 前次資料準備的 GraphRAG downloader 使用 `main` URL，manifest 又填固定 commit；不能只依該 commit 欄宣稱 source 已固定。須驗證實際 hash 或還原可核實的舊資料 lineage。Substrate 可載入不代表此項已通過。

上述不影響本次檢驗 native tool 的工程目的，但目前不能將「索引成功載入」或「21 episodes 結束」等同於「正式研究的所有量測都已正確」。
