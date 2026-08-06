"""Plain Markdown skills and progressive V2-2 action-skill bundles."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self

import yaml

from pydantic import BaseModel, ConfigDict, Field


_ACTION_NAMES: tuple[str, ...] = ("SEARCH", "EXPAND", "READ", "FINISH")
_ACTION_LAYOUT: dict[str, tuple[str, ...]] = {
    "SEARCH": ("contract.md",),
    "EXPAND": ("contract.md", "relations.md"),
    "READ": ("contract.md",),
    "FINISH": ("contract.md", "evidence.md"),
}
_BUNDLE_SCHEMA_VERSION = 1

SkillStage = Literal["draft", "recovery"]


class SkillDocument(BaseModel):
    """Exact Markdown contents plus a content-addressed version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> Self:
        source = Path(path)
        raw = source.read_bytes()
        content = raw.decode("utf-8")
        if not content.strip():
            raise ValueError(f"skill Markdown is empty: {source}")
        return cls(
            content=content,
            sha256=hashlib.sha256(raw).hexdigest(),
            source_path=source.as_posix(),
        )

    @classmethod
    def from_text(cls, content: str, *, source_path: str | None = None) -> Self:
        if not content.strip():
            raise ValueError("skill Markdown must not be empty")
        raw = content.encode("utf-8")
        return cls(
            content=content,
            sha256=hashlib.sha256(raw).hexdigest(),
            source_path=source_path,
        )

    @property
    def version(self) -> str:
        return self.sha256

    def write_snapshot(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Preserve the exact bytes represented by ``sha256``. Text-mode writes
        # translate LF to CRLF on Windows and would corrupt that contract.
        destination.write_bytes(self.content.encode("utf-8"))
        return destination


@dataclass(frozen=True, slots=True)
class ActionSkillPack:
    """The documents belonging to one progressively disclosed action skill."""

    action_type: str
    skill: SkillDocument
    contract: SkillDocument
    recovery: SkillDocument
    references: tuple[SkillDocument, ...] = ()

    def __post_init__(self) -> None:
        normalized = _normalize_action_key(self.action_type)
        object.__setattr__(self, "action_type", normalized)

    def draft_documents(self) -> tuple[SkillDocument, ...]:
        """Return only documents allowed in the first action-parameter call."""

        return (self.skill, self.contract, *self.references)

    def recovery_documents(self) -> tuple[SkillDocument, ...]:
        """Return draft documents plus guidance revealed after rejection."""

        return (*self.draft_documents(), self.recovery)

    def documents_for_stage(
        self,
        stage: SkillStage,
    ) -> tuple[SkillDocument, ...]:
        """Return the exact disclosure set for an action decision stage."""

        if stage == "draft":
            return self.draft_documents()
        if stage == "recovery":
            return self.recovery_documents()
        raise ValueError(f"unsupported action skill stage: {stage}")

    @property
    def sha256(self) -> str:
        """Return a location-independent hash of this action pack."""

        documents = [
            {"role": "skill", "sha256": self.skill.sha256},
            {"role": "contract", "sha256": self.contract.sha256},
            *[
                {"role": f"reference:{index}", "sha256": document.sha256}
                for index, document in enumerate(self.references)
            ],
            {"role": "recovery", "sha256": self.recovery.sha256},
        ]
        return _canonical_sha256(
            {
                "action_type": self.action_type,
                "documents": documents,
            }
        )

    @property
    def version(self) -> str:
        return self.sha256


class ActionSkillRegistry(Mapping[str, ActionSkillPack]):
    """Read-only action lookup accepting strings and string-valued enums."""

    __slots__ = ("_packs",)

    def __init__(self, packs: Mapping[str, ActionSkillPack]) -> None:
        normalized = {
            _normalize_action_key(action): pack
            for action, pack in packs.items()
        }
        missing = set(_ACTION_NAMES).difference(normalized)
        extra = set(normalized).difference(_ACTION_NAMES)
        if missing or extra:
            raise ValueError(
                "action skill registry must contain exactly SEARCH, EXPAND, "
                f"READ, and FINISH (missing={sorted(missing)}, "
                f"extra={sorted(extra)})"
            )
        self._packs = MappingProxyType(normalized)

    def __getitem__(self, action: object) -> ActionSkillPack:
        return self._packs[_normalize_action_key(action)]

    def __iter__(self) -> Iterator[str]:
        return iter(_ACTION_NAMES)

    def __len__(self) -> int:
        return len(self._packs)


@dataclass(frozen=True, slots=True)
class ProgressiveSkillBundle:
    """Validated root skill plus four isolated V2-2 action skill packs."""

    root: SkillDocument
    actions: ActionSkillRegistry
    sha256: str
    source_root: str
    _documents: tuple[tuple[str, SkillDocument], ...]

    @classmethod
    def load(cls, root_skill_path: str | Path) -> Self:
        """Load the fixed V2-2 layout and fail before any model call."""

        requested_root = Path(root_skill_path)
        if requested_root.name != "SKILL.md":
            raise ValueError(
                "progressive skill bundle path must point to its root SKILL.md"
            )
        try:
            root_path = requested_root.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"missing progressive skill root: {requested_root}"
            ) from error
        if not root_path.is_file():
            raise ValueError(
                f"progressive skill root is not a file: {requested_root}"
            )

        bundle_root = root_path.parent.resolve(strict=True)
        document_items: list[tuple[str, SkillDocument]] = []

        root_document = _load_bundle_document(
            bundle_root,
            "SKILL.md",
            require_frontmatter=True,
        )
        document_items.append(("SKILL.md", root_document))

        packs: dict[str, ActionSkillPack] = {}
        for action_type in _ACTION_NAMES:
            directory = action_type.casefold()
            skill_relative_path = f"actions/{directory}/SKILL.md"
            contract_relative_path = (
                f"actions/{directory}/references/contract.md"
            )
            recovery_relative_path = (
                f"actions/{directory}/references/recovery.md"
            )
            skill = _load_bundle_document(
                bundle_root,
                skill_relative_path,
                require_frontmatter=True,
            )
            contract = _load_bundle_document(
                bundle_root,
                contract_relative_path,
            )
            supplemental: list[SkillDocument] = []
            for filename in _ACTION_LAYOUT[action_type][1:]:
                relative_path = (
                    f"actions/{directory}/references/{filename}"
                )
                reference = _load_bundle_document(bundle_root, relative_path)
                supplemental.append(reference)
                document_items.append((relative_path, reference))
            recovery = _load_bundle_document(
                bundle_root,
                recovery_relative_path,
            )

            document_items.extend(
                (
                    (skill_relative_path, skill),
                    (contract_relative_path, contract),
                    (recovery_relative_path, recovery),
                )
            )
            packs[action_type] = ActionSkillPack(
                action_type=action_type,
                skill=skill,
                contract=contract,
                references=tuple(supplemental),
                recovery=recovery,
            )

        ordered_documents = tuple(
            sorted(document_items, key=lambda item: item[0])
        )
        canonical_payload = {
            "schema_version": _BUNDLE_SCHEMA_VERSION,
            "documents": [
                {
                    "path": path,
                    "sha256": document.sha256,
                    "content": document.content,
                }
                for path, document in ordered_documents
            ],
        }
        return cls(
            root=root_document,
            actions=ActionSkillRegistry(packs),
            sha256=_canonical_sha256(canonical_payload),
            source_root=bundle_root.as_posix(),
            _documents=ordered_documents,
        )

    @property
    def version(self) -> str:
        return self.sha256

    def action(self, action: object) -> ActionSkillPack:
        """Return one pack using an ActionType/string-compatible key."""

        return self.actions[action]

    def stage_documents(
        self,
        action: object,
        *,
        recovery: bool = False,
    ) -> tuple[SkillDocument, ...]:
        """Return only the selected action's draft or recovery documents."""

        stage: SkillStage = "recovery" if recovery else "draft"
        return self.action(action).documents_for_stage(stage)

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-safe, auditable description of the loaded bundle."""

        relative_paths = {
            id(document): path for path, document in self._documents
        }

        def path_for(document: SkillDocument) -> str:
            return relative_paths[id(document)]

        actions: dict[str, Any] = {}
        for action_type in _ACTION_NAMES:
            pack = self.actions[action_type]
            actions[action_type] = {
                "sha256": pack.sha256,
                "draft_documents": [
                    path_for(document)
                    for document in pack.draft_documents()
                ],
                "recovery_documents": [
                    path_for(document)
                    for document in pack.recovery_documents()
                ],
            }
        return {
            "schema_version": _BUNDLE_SCHEMA_VERSION,
            "source_root": self.source_root,
            "sha256": self.sha256,
            "root_skill": "SKILL.md",
            "documents": [
                {
                    "path": path,
                    "sha256": document.sha256,
                    # Preserve the complete progressively disclosed bundle in
                    # the immutable episode artifact, not only the root skill.
                    "content": document.content,
                }
                for path, document in self._documents
            ],
            "actions": actions,
        }


def _normalize_action_key(action: object) -> str:
    raw = getattr(action, "value", action)
    if not isinstance(raw, str):
        raise KeyError(f"unsupported action skill key: {action!r}")
    normalized = raw.strip().upper()
    if normalized not in _ACTION_NAMES:
        raise KeyError(f"unsupported action skill key: {action!r}")
    return normalized


def _load_bundle_document(
    bundle_root: Path,
    relative_path: str,
    *,
    require_frontmatter: bool = False,
) -> SkillDocument:
    candidate = bundle_root.joinpath(*relative_path.split("/"))
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"missing progressive skill document: {relative_path}"
        ) from error
    try:
        resolved.relative_to(bundle_root)
    except ValueError as error:
        raise ValueError(
            f"progressive skill document escapes bundle root: {relative_path}"
        ) from error
    if not resolved.is_file():
        raise ValueError(
            f"progressive skill document is not a file: {relative_path}"
        )

    try:
        document = SkillDocument.load(resolved)
    except UnicodeDecodeError as error:
        raise ValueError(
            f"progressive skill document is not valid UTF-8: {relative_path}"
        ) from error
    except ValueError as error:
        raise ValueError(
            f"invalid progressive skill document {relative_path}: {error}"
        ) from error

    if require_frontmatter:
        _validate_skill_frontmatter(document.content, relative_path)
    return document


def _validate_skill_frontmatter(content: str, relative_path: str) -> None:
    lines = content.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(
            f"skill document lacks YAML frontmatter: {relative_path}"
        )
    try:
        closing_index = lines[1:].index("---") + 1
    except ValueError as error:
        raise ValueError(
            f"skill document has unterminated YAML frontmatter: {relative_path}"
        ) from error
    try:
        metadata = yaml.safe_load("\n".join(lines[1:closing_index]))
    except yaml.YAMLError as error:
        raise ValueError(
            f"skill document has invalid YAML frontmatter: {relative_path}"
        ) from error
    if not isinstance(metadata, dict):
        raise ValueError(
            f"skill document frontmatter must be a mapping: {relative_path}"
        )
    if set(metadata) != {"name", "description"}:
        raise ValueError(
            "skill document frontmatter must contain exactly name and "
            f"description: {relative_path}"
        )
    if not all(
        isinstance(metadata[field], str) and metadata[field].strip()
        for field in ("name", "description")
    ):
        raise ValueError(
            f"skill document name and description must be nonempty: {relative_path}"
        )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
