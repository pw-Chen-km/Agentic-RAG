from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


_FRESH_PROCESS_SCRIPT = r"""
import json
import sys
from pathlib import Path

from agentic_rag.agent.answer import FakeAnswerGenerator
from agentic_rag.agent.config import AgentConfig
from agentic_rag.agent.harness import AgentHarness
from agentic_rag.agent.models import (
    AssessmentStatus,
    EvidenceAssessment,
    FinishAction,
    PolicyDecision,
    SearchAction,
    SentenceRef,
)
from agentic_rag.agent.policy import ScriptedPolicy
from agentic_rag.agent.skill import SkillDocument
from agentic_rag.storage import Substrate

substrate = Substrate.open(Path(sys.argv[1]))
sentence_id = next(
    sentence.sentence_id
    for sentence in substrate.sentences
    if sentence.text == "Marie Curie was born in Warsaw."
)
search = PolicyDecision(
    assessment=EvidenceAssessment(
        status="INSUFFICIENT",
        missing_information=["The birthplace is not established yet."],
    ),
    action=SearchAction(
        query="Where was Marie Curie born?",
        method="BM25",
        target="SENTENCE",
    ),
)
finish = PolicyDecision(
    assessment=EvidenceAssessment(
        status=AssessmentStatus.SUFFICIENT,
        supported_facts=["Marie Curie was born in Warsaw."],
        selected_evidence_refs=[SentenceRef(id=sentence_id)],
    ),
    action=FinishAction(evidence_refs=[SentenceRef(id=sentence_id)]),
)
harness = AgentHarness(
    substrate=substrate,
    config=AgentConfig(),
    skill=SkillDocument.from_text("# Test skill\n\nUse the full question.\n"),
    policy=ScriptedPolicy([search, finish]),
    answer_generator=FakeAnswerGenerator("Warsaw"),
    output_root=Path(sys.argv[2]),
)
result = harness.controller.run_episode(
    "Where was Marie Curie born?",
    "q1",
    episode_id="fresh-process-reload",
)
payload = {
    "answer": result.answer,
    "termination_reason": result.termination_reason,
    "selected_evidence_refs": result.selected_evidence_refs,
    "trajectory": [
        {
            "step": record.step,
            "decision": record.decision,
            "validation_status": record.validation_status,
            "observation": record.observation,
            "state_after": record.state_after,
            "policy_view": record.policy_view,
        }
        for record in result.trajectory
    ],
}
print(
    "RESULT_JSON="
    + json.dumps(
        payload,
        default=lambda value: value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
)
"""


def _run_in_fresh_process(substrate: Path, output: Path) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _FRESH_PROCESS_SCRIPT,
            str(substrate),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result_line = next(
        line
        for line in reversed(completed.stdout.splitlines())
        if line.startswith("RESULT_JSON=")
    )
    return json.loads(result_line.removeprefix("RESULT_JSON="))


def test_fresh_process_reload_preserves_observations_ids_and_trajectory_shape(
    built_substrate: Path,
    tmp_path: Path,
) -> None:
    first = _run_in_fresh_process(
        built_substrate, tmp_path / "first-process"
    )
    second = _run_in_fresh_process(
        built_substrate, tmp_path / "second-process"
    )

    assert first == second
    assert first["answer"] == "Warsaw"
    assert first["termination_reason"] == "finish"
    assert [step["decision"]["action"]["type"] for step in first["trajectory"]] == [
        "SEARCH",
        "FINISH",
    ]
    sentence_results = first["trajectory"][0]["observation"]["results"]
    assert any(
        result.get("text") == "Marie Curie was born in Warsaw."
        and result.get("parent_chunk_id")
        and result.get("document_id")
        for result in sentence_results
    )
    assert first["trajectory"][0]["state_after"]["node_handles"]
    assert (
        first["trajectory"][1]["policy_view"]["context_mode"]
        == "compact_evidence"
    )
