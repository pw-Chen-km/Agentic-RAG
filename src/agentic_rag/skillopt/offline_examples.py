"""Write auditable, hand-built examples without running an Agent or an LLM.

These are synthetic contract checks, not benchmark results or learned skills.
All four representations are rendered from each identical immutable episode.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentic_rag.agent.action_space import AvailableActionSpaceBuilder
from agentic_rag.agent.models import (
    Assessment, ChunkMemoryItem, ChunkRef, ContextNodeReference, ContextReferenceMap,
    EntityMemoryItem, EpisodeResult, EpisodeState, ExpandAction, ExpansionKind,
    FinishAction, Observation, ObservationStatus, PolicyDecision, PolicyStateView,
    PolicyView, ReadAction, ResolvedDecision, ResolvedEvidence, ResolvedExpandAction,
    ResolvedFinishAction, ResolvedReadAction, SearchAction, SearchMethod,
    SearchTarget, SentenceMemoryItem, SentenceRef, StepRecord, TerminationReason,
    Usage, ValidationStatus,
)
from agentic_rag.skillopt.adapter import _native_reflection_conversation
from agentic_rag.skillopt.evidence_progress import EVIDENCE_PROGRESS_VERSION
from agentic_rag.skillopt.trajectory import (
    TRAJECTORY_REPRESENTATIONS, build_reflection_input, build_training_reference_text,
)


OFFLINE_EXAMPLES_VERSION = "skillopt-synthetic-progress-examples-v1"
_SKILL = "Synthetic shared skill: choose SEARCH, EXPAND, READ, or FINISH using visible evidence."
_PROTOCOL = "Synthetic shared protocol: emit one legal action and use visible references."


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _ExampleTrace:
    """Small explicit state recorder; deliberately performs no tool execution."""

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.state = EpisodeState.initial()
        self.memory: dict[str, Any] = {}
        self.steps: list[StepRecord] = []
        self.canary = f"RETRIEVED_ONLY_CANARY_{case_id}"

    def _references(self) -> ContextReferenceMap:
        return ContextReferenceMap(typed_refs={
            ref: ContextNodeReference(
                node_type=self.state.reference_registry.node_type_for_ref(ref), stable_id=stable_id,
                can_read=stable_id in self.state.visible_chunk_ids - self.state.read_chunk_ids,
                can_use_as_evidence=stable_id in self.state.eligible_sentence_ids | self.state.read_chunk_ids,
            ) for ref, stable_id in self.state.reference_registry.ref_to_stable_id.items()
        })

    def _record(self, action: Any, resolved: Any, update: Callable[[], list[dict[str, Any]]]) -> None:
        before = self.state.model_copy(deep=True)
        references = self._references()
        assessment = Assessment(missing_information=["Need the remaining requested information."])
        view = PolicyView(policy_state=PolicyStateView(
            step=before.step, policy_attempts=before.policy_attempts,
            semantic_memory=[item.model_copy(deep=True) for item in self.memory.values()],
            budget=f"steps={before.remaining_step_budget}; attempts={before.remaining_policy_attempt_budget}; tokens={before.remaining_retrieved_token_budget}",
        ))
        results = update()
        self.state.step += 1
        self.state.policy_attempts += 1
        self.state.remaining_step_budget -= 1
        self.state.remaining_policy_attempt_budget -= 1
        retrieved_tokens = 0 if action.type == "FINISH" else 10
        self.state.remaining_retrieved_token_budget -= retrieved_tokens
        self.state.last_assessment = assessment
        observation = Observation(
            action_id=f"synthetic-action-{self.state.step}", status=ObservationStatus.OK,
            action=resolved, results=results, retrieved_tokens=retrieved_tokens,
            novel_node_ids=sorted(
                (self.state.visible_sentence_ids | self.state.visible_chunk_ids | self.state.visible_entity_ids)
                - (before.visible_sentence_ids | before.visible_chunk_ids | before.visible_entity_ids)
            ),
        )
        self.state.newest_observation = observation
        self.steps.append(StepRecord(
            step=self.state.step, policy_attempt=self.state.policy_attempts,
            decision=PolicyDecision(assessment=assessment, action=action),
            resolved_decision=ResolvedDecision(assessment=assessment, action=resolved),
            validation_status=ValidationStatus.VALID, observation=observation,
            state_before=before, state_after=self.state.model_copy(deep=True),
            policy_view=view, context_reference_map=references,
            available_action_space=AvailableActionSpaceBuilder(tuple(ExpansionKind)).build(before, references),
            usage=Usage(retrieved_tokens=retrieved_tokens),
        ))

    def sentence(self, sentence_id: str, text: str) -> None:
        chunk_id = f"chunk-for-{sentence_id}"
        def update():
            chunk_ref = self.state.reference_registry.register(chunk_id, "CHUNK")
            sentence_ref = self.state.reference_registry.register(sentence_id, "SENTENCE")
            self.state.visible_chunk_ids.add(chunk_id)
            self.state.visible_sentence_ids.add(sentence_id)
            self.state.eligible_sentence_ids.add(sentence_id)
            self.memory[chunk_id] = ChunkMemoryItem(ref=chunk_ref, chunk_position=0,
                                                   has_been_read=False, previews=[text])
            self.memory[sentence_id] = SentenceMemoryItem(ref=sentence_ref, parent_chunk_ref=chunk_ref, text=text)
            return [{"sentence_id": sentence_id, "parent_chunk_id": chunk_id, "document_id": "synthetic-doc",
                     "text": text, "target": "SENTENCE", "evidence_eligible": True}]
        action = SearchAction(query=f"synthetic retrieval query {len(self.steps) + 1}",
                              method=SearchMethod.BM25, target=SearchTarget.SENTENCE)
        self._record(action, action, update)

    def entity(self, entity_id: str) -> None:
        def update():
            ref = self.state.reference_registry.register(entity_id, "ENTITY")
            self.state.visible_entity_ids.add(entity_id)
            self.memory[entity_id] = EntityMemoryItem(ref=ref, canonical_name="Bridge person")
            return [{"entity_id": entity_id, "canonical_name": "Bridge person", "target": "ENTITY"}]
        action = SearchAction(query="Bridge person", method=SearchMethod.LEXICAL, target=SearchTarget.ENTITY)
        self._record(action, action, update)

    def chunk(self, chunk_id: str, preview: str, *, parent: str | None = None) -> None:
        def update():
            ref = self.state.reference_registry.register(chunk_id, "CHUNK")
            self.state.visible_chunk_ids.add(chunk_id)
            self.memory[chunk_id] = ChunkMemoryItem(ref=ref, chunk_position=0, has_been_read=False, previews=[preview])
            return [{"chunk_id": chunk_id, "document_id": "synthetic-doc", "target": "CHUNK",
                     "navigation_only": True, "previews": [{"text": preview, "navigation_only": True}]}]
        if parent is None:
            action = SearchAction(query="synthetic chunk retrieval", method=SearchMethod.BM25, target=SearchTarget.CHUNK)
            resolved = action
        else:
            action = ExpandAction(kind=ExpansionKind.ENTITY_MENTIONED_IN_CHUNK,
                                  source_ref=self.state.reference_registry.ref_for(parent))
            resolved = ResolvedExpandAction(kind=action.kind, source_id=parent)
        self._record(action, resolved, update)

    def read(self, chunk_id: str, full_text: str) -> None:
        def update():
            self.state.read_chunk_ids.add(chunk_id)
            self.memory[chunk_id] = ChunkMemoryItem(ref=self.state.reference_registry.ref_for(chunk_id),
                                                   chunk_position=0, has_been_read=True, text=full_text)
            return [{"chunk_id": chunk_id, "document_id": "synthetic-doc", "text": full_text,
                     "sentences": [], "target": "CHUNK", "content_read": True}]
        action = ReadAction(chunk_ref=self.state.reference_registry.ref_for(chunk_id))
        self._record(action, ResolvedReadAction(chunk_id=chunk_id), update)

    def finish(self, question: str, answer: str) -> EpisodeResult:
        evidence = [ChunkRef(id=stable_id) for stable_id in sorted(self.state.read_chunk_ids)]
        evidence += [SentenceRef(id=stable_id) for stable_id in sorted(self.state.eligible_sentence_ids)]
        if not evidence:
            raise ValueError("synthetic FINISH requires exposed, eligible evidence")
        action = FinishAction(answer=answer, evidence_refs=[self.state.reference_registry.ref_for(ref.id) for ref in evidence])
        resolved = ResolvedFinishAction(answer=answer, evidence_refs=evidence)
        self._record(action, resolved, lambda: [])
        return EpisodeResult(
            episode_id=self.case_id, query=question, scope_id="synthetic:offline_examples",
            answer=answer, evidence_refs=evidence, termination_reason=TerminationReason.FINISH,
            trajectory=self.steps, final_state=self.state.model_copy(deep=True),
            resolved_evidence=[ResolvedEvidence(ref=ref, text=self.memory[ref.id].text,
                                                document_id="synthetic-doc") for ref in evidence],
            usage=Usage(retrieved_tokens=sum(step.usage.retrieved_tokens for step in self.steps)),
        )


def _lexical(*facts: str) -> dict[str, Any]:
    return {"kind": "reference_text", "method": "lexical_f1", "status": "ready",
            "facts": [{"fact_id": f"G{index}", "text": text, "source_indices": [index - 1]}
                      for index, text in enumerate(facts, 1)]}


def _cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add(trace: _ExampleTrace, question: str, answer: str, reference: dict[str, Any],
            title: str, explanation: str) -> None:
        cases.append({"case_id": trace.case_id, "title": title, "explanation": explanation,
                      "canary": trace.canary, "episode": trace.finish(question, answer),
                      "reference": reference})

    trace = _ExampleTrace("lexical_repeat")
    first = f"biopsy confirms cancer diagnosis.\n{trace.canary}"
    trace.sentence("sentence-a", first)
    trace.sentence("sentence-a", first)
    trace.sentence("sentence-b", "surgery removes lesion")
    add(trace, "Which evidence concerns diagnosis and lesion removal?", "Diagnosis and surgery.",
        _lexical("biopsy confirms diagnosis", "surgery removes lesion"), "相同結果不會一直加分",
        "第一條標準文字有3個詞，取回的對應句有4個詞，共同3個，所以F1=6/7。另一條尚未匹配，平均為3/7≈0.429。第二次同樣文字增加0；第三次完整找到第二條，平均變成13/14≈0.929。這是字詞接近程度，不是醫學正確性判定。")

    trace = _ExampleTrace("preview_then_read")
    text = f"Marie Curie won Nobel Prize.\n{trace.canary}"
    trace.chunk("chunk-a", text)
    trace.read("chunk-a", text)
    add(trace, "What did the named person win?", "Nobel Prize.", _lexical("Marie Curie won Nobel Prize"),
        "看到了，與可以引用，是兩件事",
        "預覽已含相同字詞，所以「看過」分數是1；但未讀Chunk尚不能引用，所以「可以引用」分數是0。READ後兩者都是1。READ的看過分數增加0，但引用分數增加1，不能說READ沒有價值。")

    trace = _ExampleTrace("expand_then_read")
    trace.entity("entity-bridge")
    trace.chunk("chunk-a", f"Unrelated navigation preview.\n{trace.canary}", parent="entity-bridge")
    trace.read("chunk-a", f"biopsy confirms diagnosis.\n{trace.canary}")
    add(trace, "Which observation confirms the diagnosis?", "Biopsy confirms diagnosis.", _lexical("biopsy confirms diagnosis"),
        "展開提供入口，讀取才看到相關文字",
        "第1步取得E1，第2步從E1展開取得C1，但預覽尚未增加字詞分數；第3步READ C1才增加1。紀錄保留E1→EXPAND→C1→READ，但不把READ的1分追加入EXPAND。這只是記錄直接取得時間，不是宣稱EXPAND對成功沒有間接價值。")

    trace = _ExampleTrace("word_order_limitation")
    trace.sentence("sentence-a", f"The king killed the duke.\n{trace.canary}")
    add(trace, "Who killed whom?", "The king killed the duke.", _lexical("The duke killed the king"),
        "同樣的詞，不代表同樣的意思",
        "標準文字是公爵殺國王，取回文字是國王殺公爵。兩句詞袋相同，F1仍是1，但角色完全相反。此例刻意保留，證明此指標不能充當真假判斷。")

    trace = _ExampleTrace("paragraph_exposure")
    first, second = "A" * 20 + "B" * 80, "C" * 100
    full_text = f"{first}\n{trace.canary}"
    reference = {"kind": "paragraph", "method": "paragraph_span", "status": "ready", "facts": [
        {"fact_id": "P1", "text": first, "source_indices": [0], "mappings": [
            {"unit_id": "chunk-a", "unit_text": full_text, "unit_start": 0, "unit_end": 100,
             "fact_start": 0, "fact_end": 100}]},
        {"fact_id": "P2", "text": second, "source_indices": [1], "mappings": [
            {"unit_id": "unseen-chunk-b", "unit_text": second, "unit_start": 0, "unit_end": 100,
             "fact_start": 0, "fact_end": 100}]},
    ]}
    trace.chunk("chunk-a", first[:20])
    trace.read("chunk-a", full_text)
    add(trace, "Show the exposure of two synthetic 100-character passages.", "Only the first passage was read.", reference,
        "段落只看到一部分，不能算整段取得",
        "兩個人工段落各有100個非空白字元，用重複字母只為了方便核算。先顯示P1的20個字元：平均(0.2+0)/2=0.1，部分看過1段、完整看過0段。讀完P1後平均(1+0)/2=0.5，完整看過1段。這不是答案完整度50%。")
    return cases


def _progress_rows(rendered: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory = rendered["trajectory"]
    return [step["reference_progress"] for step in trajectory.get("step_diagnostics", trajectory.get("abstract_steps", []))]


def _decision_contexts(rendered: dict[str, Any]) -> list[dict[str, Any]]:
    trajectory = rendered["trajectory"]
    steps = trajectory.get("raw_steps", trajectory.get("action_ledger", trajectory.get("abstract_steps", [])))
    return [step["decision_context"] for step in steps]


def _build_artifacts() -> tuple[dict[str, bytes], dict[str, Any]]:
    artifacts: dict[str, bytes] = {}
    summaries = []
    report = ["# 標準證據進度：離線示意案例", "",
              "**所有案例都是 SYNTHETIC：人工建構、沒有呼叫模型，不是實驗成績。**",
              "本報告沒有使用任何訓練、驗證或測試集的真實軌跡。它檢查計分與四組輸入是否遵守相同規則，不證明任何一組效果較好。", "",
              "## 如何讀表", "",
              "「看過」只計算實際保存的Policy輸入文字；「可以引用」只計算當時合法證據。每步的工具結果，要到下一次模型輸入才算呈現；FINISH沒有新檢索。字詞分數是各條標準文字的歷史最高重疊分數平均，段落分數則是原始非空白字元的累積呈現比例。", "",
              "原始紀錄、整理紀錄、整理加分數三組保留檢索文字；第四組不保留檢索文字，但仍保留四組共有的標準答案／標準證據。第三、四組的進度數字必須逐步相同。", ""]
    for number, case in enumerate(_cases(), 1):
        case_id, episode, reference = case["case_id"], case["episode"], case["reference"]
        item = {"id": case_id, "question": episode.query, "answer": episode.answer,
                "source": "synthetic_offline_examples", "question_type": "SYNTHETIC",
                "reference_evidence": reference}
        # The role-reversal example deliberately answers contrary to its GT.
        if case_id == "word_order_limitation":
            item["answer"] = "The duke killed the king."
        prefix = f"cases/{case_id}"
        artifacts[f"{prefix}/episode.json"] = _json_bytes(episode.model_dump(mode="json"))
        artifacts[f"{prefix}/reference.evaluator_only.json"] = _json_bytes(reference)
        outputs = {}
        contexts = []
        common_reference = build_training_reference_text(item)
        for arm in TRAJECTORY_REPRESENTATIONS:
            rendered, manifest = build_reflection_input(
                episode=episode, item=item, skill_content=_SKILL, target_system_prompt=_PROTOCOL,
                rollout_phase="train", rollout_split="synthetic_only", trajectory_representation=arm,
            )
            native = {"synthetic": True, "conversations": _native_reflection_conversation(rendered),
                      "training_reference": common_reference}
            native_bytes = _json_bytes(native)
            contains_canary = case["canary"].encode() in native_bytes
            if contains_canary != (arm != "progress_abstracted"):
                raise AssertionError(f"retrieval-text boundary failed for {case_id}/{arm}")
            if "mappings" in json.dumps(rendered["hidden_reference"]):
                raise AssertionError("evaluator source mappings leaked into common reference")
            outputs[arm] = rendered
            contexts.append(_decision_contexts(rendered))
            artifacts[f"{prefix}/{arm}/reflection_input.json"] = _json_bytes(rendered)
            artifacts[f"{prefix}/{arm}/optimizer_input.json"] = native_bytes
            artifacts[f"{prefix}/{arm}/manifest.json"] = _json_bytes(manifest)
        if any(context != contexts[0] for context in contexts):
            raise AssertionError(f"shared step background differs in {case_id}")
        if any(output["hidden_reference"] != outputs["raw"]["hidden_reference"] for output in outputs.values()):
            raise AssertionError(f"shared hidden reference differs in {case_id}")
        rows = _progress_rows(outputs["organized_support_labels"])
        if rows != _progress_rows(outputs["progress_abstracted"]):
            raise AssertionError(f"third/fourth progress differs in {case_id}")
        if any(row["status"] != "ready" for row in rows):
            raise AssertionError(f"synthetic reference could not be evaluated in {case_id}")
        if case["canary"] in json.dumps(rows):
            raise AssertionError("retrieval text leaked into progress rows")
        values = []
        for step, row in zip(episode.trajectory, rows, strict=True):
            values.append({"step": step.step, "action": step.decision.action.type,
                           "visible_score": row["visible"]["after"]["score"],
                           "visible_delta": row["visible"]["delta"],
                           "eligible_score": row["eligible"]["after"]["score"],
                           "eligible_delta": row["eligible"]["delta"]})
        summaries.append({"case_id": case_id, "synthetic": True, "title": case["title"],
                          "method": reference["method"], "steps": values,
                          "checks": {"third_fourth_progress_equal": True, "shared_context_equal": True,
                                     "shared_hidden_reference_equal": True, "fourth_retrieval_text_absent": True,
                                     "first_three_retrieval_text_present": True}})
        report.extend([f"## 案例{number}：{case['title']}（SYNTHETIC）", "", case["explanation"], "",
                       "| 步驟 | 動作 | 看過分數 | 本步增加 | 可引用分數 | 本步增加 |",
                       "|---|---|---:|---:|---:|---:|"])
        report.extend(f"| {row['step']} | {row['action']} | {row['visible_score']:.3f} | {row['visible_delta']:.3f} | {row['eligible_score']:.3f} | {row['eligible_delta']:.3f} |" for row in values)
        report.extend(["", f"逐步輸入與四組對照：`{prefix}/`。", ""])
        if case_id == "expand_then_read":
            use = outputs["progress_abstracted"]["trajectory"]["abstract_steps"][0]["later_expand_uses_of_acquired_information"][0]
            if use["reference_progress"]["visible_delta"] != 0 or use["later_reads_of_direct_children"][0]["reference_progress"]["visible_delta"] != 1:
                raise AssertionError("READ gain was not kept separate from EXPAND")
            summaries[-1]["bridge_attribution"] = use
    report.extend(["## 驗收範圍與限制", "",
                   "這些案例驗證：重複不加分、預覽與引用分開、未來READ不回填、直接EXPAND與後續READ分開、段落只按真正呈現範圍計分，以及第四組移除检索文字。它們不評估模型正確率，不涉及資料切分，也沒有調整任何保留測試題的設定。", "",
                   "工具失敗、缺少Policy輸入、來源座標錯誤與重複片段歧義，另由自動測試覆蓋；不可評估不是零分。字詞分數不能辨認否定、因果、角色反轉或同義改寫；段落呈現比例不等於必要事實已全部取得。", ""])
    artifacts["example_report.md"] = "\n".join(report).encode("utf-8")
    summary = {"schema_version": OFFLINE_EXAMPLES_VERSION, "progress_version": EVIDENCE_PROGRESS_VERSION,
               "synthetic": True, "model_calls": 0, "saved_real_trajectories_used": 0,
               "case_count": len(summaries), "arm_count": len(TRAJECTORY_REPRESENTATIONS), "cases": summaries,
               "source_code_sha256": {"offline_examples.py": _sha256(Path(__file__).read_bytes()),
                                       "evidence_progress.py": _sha256(Path(__file__).with_name("evidence_progress.py").read_bytes()),
                                       "trajectory.py": _sha256(Path(__file__).with_name("trajectory.py").read_bytes())},
               "artifact_sha256": {name: _sha256(content) for name, content in sorted(artifacts.items())}}
    artifacts["summary.json"] = _json_bytes(summary)
    return artifacts, summary


def write_offline_examples(output: Path) -> dict[str, Any]:
    """Write new example artifacts or verify byte-identical existing artifacts.

    Different existing files cause FileExistsError before any writes.  No old
    output is overwritten, deleted, or treated as a resumable experiment.
    """

    output = Path(output)
    artifacts, summary = _build_artifacts()
    for name, content in artifacts.items():
        destination = output / name
        if destination.exists() and (not destination.is_file() or destination.read_bytes() != content):
            raise FileExistsError(f"refusing to overwrite non-identical offline example: {destination}")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("xb") as stream:
                stream.write(content)
        except FileExistsError:
            if not destination.is_file() or destination.read_bytes() != content:
                raise FileExistsError(f"offline example changed during write: {destination}") from None
    return summary
