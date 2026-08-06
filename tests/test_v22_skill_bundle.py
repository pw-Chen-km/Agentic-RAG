from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
import shutil

import pytest

from agentic_rag.agent.skill import ProgressiveSkillBundle


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "skills" / "agentic-rag-v2-2"


class _ActionType(StrEnum):
    SEARCH = "SEARCH"


def _copy_bundle(tmp_path: Path) -> Path:
    destination = tmp_path / "agentic-rag-v2-2"
    shutil.copytree(BUNDLE_ROOT, destination)
    return destination


def _document_paths(documents: object) -> list[str]:
    return [
        str(Path(document.source_path)).replace("\\", "/")
        for document in documents
    ]


def test_loads_fixed_bundle_and_accepts_enum_or_string_action_keys() -> None:
    bundle = ProgressiveSkillBundle.load(BUNDLE_ROOT / "SKILL.md")

    assert tuple(bundle.actions) == ("SEARCH", "EXPAND", "READ", "FINISH")
    assert bundle.actions["search"] is bundle.actions[_ActionType.SEARCH]
    assert bundle.action("EXPAND").action_type == "EXPAND"
    assert bundle.version == bundle.sha256
    assert len(bundle.sha256) == 64
    assert len(bundle.manifest()["documents"]) == 15


def test_stage_documents_enforce_progressive_disclosure() -> None:
    bundle = ProgressiveSkillBundle.load(BUNDLE_ROOT / "SKILL.md")

    draft = bundle.stage_documents("EXPAND")
    draft_paths = _document_paths(draft)
    assert [Path(path).name for path in draft_paths] == [
        "SKILL.md",
        "contract.md",
        "relations.md",
    ]
    assert all("/actions/expand/" in path for path in draft_paths)
    assert all(not path.endswith("/recovery.md") for path in draft_paths)
    assert "legal_action_options" not in "\n".join(
        document.content for document in draft
    )

    recovery = bundle.stage_documents("EXPAND", recovery=True)
    recovery_paths = _document_paths(recovery)
    assert recovery_paths[:-1] == draft_paths
    assert recovery_paths[-1].endswith(
        "/actions/expand/references/recovery.md"
    )
    assert "legal_action_options" in recovery[-1].content


def test_each_stage_discloses_only_the_selected_action() -> None:
    bundle = ProgressiveSkillBundle.load(BUNDLE_ROOT / "SKILL.md")

    for action_type in bundle.actions:
        action_directory = action_type.casefold()
        documents = bundle.stage_documents(action_type, recovery=True)
        paths = _document_paths(documents)
        assert all(
            f"/actions/{action_directory}/" in path for path in paths
        )
        assert bundle.root not in documents


def test_bundle_hash_is_canonical_and_location_independent(
    tmp_path: Path,
) -> None:
    original = ProgressiveSkillBundle.load(BUNDLE_ROOT / "SKILL.md")
    copied_root = _copy_bundle(tmp_path)
    copied = ProgressiveSkillBundle.load(copied_root / "SKILL.md")

    assert copied.sha256 == original.sha256
    manifest = copied.manifest()
    canonical_payload = {
        "schema_version": manifest["schema_version"],
        "documents": manifest["documents"],
    }
    encoded = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert copied.sha256 == hashlib.sha256(encoded).hexdigest()
    assert json.loads(json.dumps(manifest))["sha256"] == copied.sha256

    contract = copied_root / "actions" / "search" / "references" / "contract.md"
    contract.write_bytes(contract.read_bytes() + b"\n")
    changed = ProgressiveSkillBundle.load(copied_root / "SKILL.md")
    assert changed.sha256 != copied.sha256


def test_bundle_fails_fast_when_required_document_is_missing(
    tmp_path: Path,
) -> None:
    copied_root = _copy_bundle(tmp_path)
    missing = (
        copied_root
        / "actions"
        / "read"
        / "references"
        / "recovery.md"
    )
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="actions/read/references/recovery.md"):
        ProgressiveSkillBundle.load(copied_root / "SKILL.md")


@pytest.mark.parametrize(
    ("relative_path", "replacement", "message"),
    [
        ("actions/search/references/contract.md", b" \r\n", "empty"),
        (
            "actions/search/references/contract.md",
            b"\xff\xfe\x00",
            "not valid UTF-8",
        ),
    ],
)
def test_bundle_fails_fast_for_invalid_document_bytes(
    tmp_path: Path,
    relative_path: str,
    replacement: bytes,
    message: str,
) -> None:
    copied_root = _copy_bundle(tmp_path)
    copied_root.joinpath(*relative_path.split("/")).write_bytes(replacement)

    with pytest.raises(ValueError, match=message):
        ProgressiveSkillBundle.load(copied_root / "SKILL.md")


def test_bundle_requires_the_root_skill_path(tmp_path: Path) -> None:
    copied_root = _copy_bundle(tmp_path)

    with pytest.raises(ValueError, match="root SKILL.md"):
        ProgressiveSkillBundle.load(copied_root)


def test_expand_skill_exposes_only_the_four_v22_relations() -> None:
    bundle = ProgressiveSkillBundle.load(BUNDLE_ROOT / "SKILL.md")
    relations = bundle.action("EXPAND").references[0].content

    assert "ENTITY_MENTIONED_IN_SENTENCE" in relations
    assert "SENTENCE_MENTIONS_ENTITY" in relations
    assert "ENTITY_CO_OCCURS_ENTITY_SENTENCE" in relations
    assert "CHUNK_ADJACENT_CHUNK" in relations
    assert "ENTITY_MENTIONED_IN_CHUNK" not in relations
    assert "ENTITY_CO_OCCURS_ENTITY_CHUNK" not in relations
    assert "CHUNK_CONTAINS_SENTENCE" not in relations
    assert "CHUNK_MENTIONS_ENTITY" not in relations
