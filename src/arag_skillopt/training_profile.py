"""Explicit prompt profiles and source-aware checks for training preflight."""

import json
import hashlib
from collections import Counter
from pathlib import Path

from .skill_rules import render_formatted_prompt, skill_sha256
from .support import digest
from .diagnostics import STATE_ACTION_PROTOCOL, FORMAT_REPAIR_VERSION

PROFILE = "diagnosis_skill_v4"
PROTOCOL = "diagnosis-to-edits-v4-single-case"
NEW_PROFILE = "diagnosis_skill_v6"
REPAIRED_PROFILE = "diagnosis_skill_v6r1"
PROFILES = {PROFILE: PROTOCOL, NEW_PROFILE: STATE_ACTION_PROTOCOL,
            REPAIRED_PROFILE: STATE_ACTION_PROTOCOL}
EXPECTED_TOOLS = {"semantic_search", "read_chunk", "entity_hop"}
CONTEXT_REVISION = {"version": "qwen-context64-v1", "from_context_length": 32768,
                    "to_context_length": 65536}


def validate_ablation_runtime(cfg, contract):
    """Allow only an explicit, recorded context repair; retain other guards."""
    revision = cfg.get("experiment", {}).get("runtime_context_revision")
    context = cfg["llm"]["context_length"]
    if revision is not None:
        required = {"transport": "ollama_exact", "context_length": 65536,
                    "input_token_limit": 60928, "context_safety_tokens": 512,
                    "max_tokens": 4096, "require_complete_output": True}
        if (revision != CONTEXT_REVISION or contract["context_length"] != 32768
                or any(cfg["llm"].get(k) != v for k, v in required.items())):
            raise ValueError("invalid explicit ablation context revision")
        context = revision["from_context_length"]
    expected = {"model": cfg["llm"]["model"],
                "model_expected_digest": cfg["llm"]["expected_digest"],
                "context_length": context, "seed": cfg["llm"]["seed"],
                "agent_token_budget": cfg["agent"]["token_budget"]}
    if any(contract[k] != v for k, v in expected.items()):
        raise ValueError("runtime model/seed/budget differs from completed ablation")


def prompt_bundle(root, cfg, workflow=False):
    """Return a self-contained common/stage bundle; keep legacy selection intact."""
    profile = cfg.get("optimizer_prompt_version", "legacy")
    if profile in PROFILES:
        folder = Path(root) / "prompts" / profile
        read = lambda name: (folder / (name + ".md")).read_text(encoding="utf-8")
        result = {"common": read("common")}
        for stage in ("success", "failure"):
            result[stage] = read(stage) + "\n" + read("reflection")
        result["merge"] = read("merge")
        if workflow:
            result["meta"] = read("meta") + "\n" + read("reflection")
            result["meta_merge"] = read("meta_merge") + "\n" + read("merge")
        return result
    if profile != "legacy":
        raise ValueError(f"unknown optimizer_prompt_version: {profile}")
    names = {k: f"phase1_{k}" for k in ("common", "success", "failure", "merge")}
    if workflow:
        names.update(meta="meta", meta_merge="meta_merge")
    if cfg.get("entity_hop", {}).get("enabled"):
        for k in ("success", "failure", "merge", "meta", "meta_merge"):
            if k in names:
                names[k] = "entityhop_" + k
    return {k: (Path(root) / "prompts" / (name + ".md")).read_text()
            for k, name in names.items()}


def agent_prompt(root, cfg):
    return (Path(root) / cfg.get("prompt_path", "src/arag/agent/prompts/default.txt")).read_text()


def validate_profile(cfg, phase_cfg):
    profile = cfg.get("optimizer_prompt_version", "legacy")
    if profile not in PROFILES:
        return
    if phase_cfg.analysis_protocol != PROFILES[profile] or phase_cfg.min_support_cases != 1:
        raise ValueError(f"{profile} requires its protocol and min_support_cases=1")
    if profile == REPAIRED_PROFILE and phase_cfg.diagnostic_format_repair != FORMAT_REPAIR_VERSION:
        raise ValueError("v6r1 requires the explicitly versioned diagnostic format repair")
    hop = cfg.get("entity_hop", {})
    if (cfg.get("keyword_search", {}).get("enabled") is not False
            or hop.get("enabled") is not True
            or hop.get("entity_source") != "question_or_visible_evidence"
            or hop.get("backend") != "direct_bm25"):
        raise ValueError("v4 profile requires No-Keyword + Question Entities Direct-BM25")
    if not cfg.get("prompt_path") or not cfg.get("skill_path"):
        raise ValueError("explicit Agent prompt and Initial Skill are required")
    if cfg["optimizer"].get("reasoning_effort") != "medium":
        raise ValueError("Optimizer thinking must be medium")
    if cfg["llm"].get("reasoning_effort") != "none" or cfg["judge"].get("think") is not False:
        raise ValueError("Agent and Judge must keep the ablation thinking settings")
    if not cfg.get("experiment", {}).get("reviewed_test_manifest"):
        raise ValueError("reviewed test manifest is required")
    if not cfg.get("data", {}).get("source_questions"):
        raise ValueError("original source questions are required")


def validate_runtime_guidance(cfg, schemas, original, skill):
    if cfg.get("optimizer_prompt_version", "legacy") not in PROFILES:
        return
    tools = {s["function"]["name"]: s["function"] for s in schemas}
    if set(tools) != EXPECTED_TOOLS:
        raise ValueError("runtime tool list differs from the training profile")
    prompt = render_formatted_prompt(original, skill)
    if "keyword_search" in prompt + json.dumps(schemas):
        raise ValueError("disabled tool leaked into runtime guidance")
    description = tools["entity_hop"]["description"]
    if "no prior search or read is required" not in description:
        raise ValueError("entity_hop schema still requires previous evidence")
    if "in the question or visible evidence" not in prompt:
        raise ValueError("Agent prompt does not allow question entities")
    reference = cfg.get("experiment", {}).get("ablation_reference")
    if reference:
        directory = Path(reference)
        contract = json.loads((directory / "contract.json").read_text())
        arm = "entityhop_direct_no_keyword_question_entities"
        with (directory / arm / "episodes.jsonl").open() as stream:
            first = json.loads(stream.readline())["raw"]["decision_audit"][0]["request"]
        if first["messages"][0]["content"] != prompt or first["tools"] != schemas:
            raise ValueError("initial Agent prompt/tools differ from completed Question Entities ablation")
        if contract["arm_inputs"][arm]["skill_sha256"] != skill_sha256(skill):
            raise ValueError("initial Skill differs from completed ablation")
        validate_ablation_runtime(cfg, contract)
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        semantic = Path(cfg["data"]["index_dir"])
        entries = [(str(p.relative_to(semantic)), sha(p))
                   for p in sorted(semantic.rglob("*")) if p.is_file()]
        semantic_sha = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        if (sha(cfg["data"]["chunks_file"]) != contract["chunks_file_sha256"]
                or sha(cfg["entity_hop"]["index_file"]) != contract["index_hashes"]["direct_bm25"]
                or semantic_sha != contract["index_hashes"]["semantic_index_dir"]):
            raise ValueError("corpus or retrieval index differs from completed ablation")


def rows(path):
    text = Path(path).read_text()
    return json.loads(text) if Path(path).suffix == ".json" else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]


def validate_split_sources(cfg):
    """Check IDs AND original row/source identity, before projecting runtime rows."""
    if cfg.get("optimizer_prompt_version", "legacy") not in PROFILES:
        return {}
    data, experiment = cfg["data"], cfg["experiment"]
    split_rows = {split: rows(data[split]) for split in ("train", "validation", "test")}
    expected = experiment["split_counts"]
    corpus_rows = rows(data["questions"])
    by_id = {r["id"]: r for r in corpus_rows}
    original_source = rows(data["source_questions"]) if data.get("source_questions") else None
    if len(by_id) != len(corpus_rows):
        raise ValueError("duplicate canonical question ID")
    all_rows = []
    for split, items in split_rows.items():
        if len(items) != expected[split]:
            raise ValueError(f"{split} count differs from the frozen split")
        for r in items:
            canonical = by_id.get(r["id"])
            if canonical is None or any(r.get(k) != canonical.get(k) for k in
                    ("id", "question", "answer", "question_type", "source_question_id")):
                raise ValueError("split row differs from canonical question record")
            if type(r.get("raw_source_row_index")) is not int or not r.get("source_question_id"):
                raise ValueError("original source row and source question ID are required")
            if r["raw_source_row_index"] != canonical.get("source_row_index"):
                raise ValueError("split source row differs from canonical source row")
            if original_source is not None:
                index = r["raw_source_row_index"]
                if not 0 <= index < len(original_source):
                    raise ValueError("original source row is out of range")
                source = original_source[index]
                kind = str(source.get("question_type", source.get("type", ""))).strip().lower().replace(" ", "_")
                if (source.get("question") != r["question"]
                        or source.get("answer") != r["answer"]
                        or source.get("id") != r["source_question_id"]
                        or kind != r["question_type"]):
                    raise ValueError("split question differs from its original source row")
        all_rows.extend(items)
    # Some source question IDs are reused in the upstream dataset. The row
    # coordinate, not that string alone, is the unique original identity.
    for field in ("id", "raw_source_row_index", "question"):
        values = [r[field] for r in all_rows]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate source identity across splits: {field}")
    if {r["id"] for r in all_rows} != set(by_id):
        raise ValueError("splits do not partition canonical questions")
    manifest = json.loads(Path(experiment["reviewed_test_manifest"]).read_text())
    records = manifest.get("question_records", [])
    if (manifest.get("source_split") != "test" or len(records) != 80
            or len(manifest.get("question_ids", [])) != 80
            or len(set(manifest.get("question_ids", []))) != 80
            or set(manifest["question_ids"]) != {r["id"] for r in records}
            or Counter(r["question_type"] for r in records) != {
                "fact_retrieval": 20, "complex_reasoning": 20,
                "contextual_summarize": 20, "creative_generation": 20}):
        raise ValueError("reviewed test manifest must contain the frozen 80 unique questions")
    test_by_id = {r["id"]: r for r in split_rows["test"]}
    for r in records:
        actual = test_by_id.get(r["id"])
        if actual is None:
            raise ValueError("reviewed test question is absent from test or leaked into training")
        for key in ("raw_source_row_index", "question_type"):
            if r.get(key) != actual.get(key):
                raise ValueError(f"reviewed question source mapping differs: {key}")
        if r.get("source_question_id") is not None and r["source_question_id"] != actual["source_question_id"]:
            raise ValueError("reviewed question source ID differs")
    preparation = experiment.get("preparation_manifest")
    if preparation:
        preparation_record = json.loads(Path(preparation).read_text())
        frozen = preparation_record["output_files_sha256"]
        for key in ("train", "validation", "test", "questions", "support_map"):
            path = Path(data[key])
            if hashlib.sha256(path.read_bytes()).hexdigest() != frozen[path.name]:
                raise ValueError(f"frozen preparation hash differs: {key}")
        if original_source is not None:
            source_hashes = [value for path, value in preparation_record["input_files_sha256"].items()
                             if Path(path).name == "source_questions.json"]
            if len(source_hashes) != 1 or hashlib.sha256(Path(data["source_questions"]).read_bytes()).hexdigest() != source_hashes[0]:
                raise ValueError("original source questions hash differs from preparation provenance")
    # The reference-text map must not manufacture exact supporting-fact recall.
    support = json.loads(Path(data["support_map"]).read_text())
    if any(q.get("evaluable") is True or q.get("exact_recall_available") is True
           for q in support.get("questions", {}).values()):
        raise ValueError("Novel exact evidence recall must remain unavailable")
    return {"split_counts": {k: len(v) for k, v in split_rows.items()},
            "reviewed_test_questions": len(records), "reviewed_test_overlap_train_validation": 0,
            "original_source_coordinates_verified": True,
            "evidence_recall": "N/A", "manifest_hash": digest(manifest)}
