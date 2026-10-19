"""Build the corpus-only entity substrate used by EntityHop Linked.

The substrate is deliberately an offline artefact.  It never reads questions,
answers, or evidence annotations.  NER and the abbreviation setting are the
same ones used by the Novel substrate (``en_core_web_sm`` with the optional
scispaCy abbreviation detector).  Entity IDs in this file are implementation
details and are not exposed to the language model.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


VERSION = "entityhop-entity-link-v1"
_PAREN_RE = re.compile(
    r"(?P<long>[A-Z][A-Za-z0-9][A-Za-z0-9 .,'’\-]{2,100}?)\s*\(\s*(?P<short>[A-Z][A-Za-z0-9 .,'’\-]{1,30})\s*\)"
)
_ALSO_KNOWN_RE = re.compile(
    r"(?P<prefix>(?:also known as|aka|a\.k\.a\.)\s+)(?P<alias>[A-Z][A-Za-z0-9 .,'’\-]{1,80})"
)
_ACRONYM_RE = re.compile(
    r"(?P<long>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,})\s*\(\s*(?P<short>[A-Z]{2,12})\s*\)"
)


def normalize_entity_name(value: str) -> str:
    """Canonical representation matching the existing substrate processor."""

    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^\W+|\W+$", "", value, flags=re.UNICODE)
    return value.strip()


def _chunks(rows: Sequence[Mapping[str, Any]] | Sequence[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i, row in enumerate(rows):
        if isinstance(row, Mapping):
            out.append({"id": str(row.get("id", i)), "text": str(row.get("text", ""))})
            continue
        value = str(row)
        chunk_id, sep, text = value.partition(":")
        out.append({"id": chunk_id if sep else str(i), "text": text if sep else value})
    return out


def load_chunks(path: str | Path) -> list[dict[str, str]]:
    """Read either the list-of-strings or list-of-objects chunk format."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("chunks file must contain a JSON list")
    return _chunks(payload)


def source_chunks_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _explicit_alias_pairs(text: str) -> list[tuple[str, str]]:
    """Return conservative, surface-form aliases visible in one chunk.

    The patterns require a capitalized long form and therefore do not infer an
    entity from arbitrary words.  They are only used to connect NER mentions;
    pairs with no NER side are ignored by :func:`build_entity_index`.
    """

    pairs: list[tuple[str, str]] = []
    for regex in (_ACRONYM_RE, _PAREN_RE):
        pairs.extend((m.group("long").strip(), m.group("short").strip()) for m in regex.finditer(text))
    for match in _ALSO_KNOWN_RE.finditer(text):
        # A sentence may continue after the alias.  Keep only a short title
        # phrase rather than using an arbitrary sentence suffix.
        alias = match.group("alias").strip(" ,.;:")
        words = alias.split()
        if len(words) > 10:
            alias = " ".join(words[:10])
        before = re.split(r"[.!?;\n]", text[: match.start("prefix")])[-1].rstrip(" ,:")
        candidate = re.search(r"([A-Z][A-Za-z0-9 .,'’\-]{2,100})$", before)
        if candidate:
            pairs.append((candidate.group(1).strip(), alias))
    # Stable de-duplication while retaining source order.
    return list(dict.fromkeys((a, b) for a, b in pairs if normalize_entity_name(a) != normalize_entity_name(b)))


def _load_nlp(model_name: str, enable_abbreviations: bool):
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("spaCy is required to build the entity-linked substrate") from exc
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"spaCy model {model_name!r} is not installed") from exc
    if enable_abbreviations:
        try:
            import scispacy  # noqa: F401
            from scispacy.abbreviation import AbbreviationDetector  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("scispaCy is required when abbreviation detection is enabled") from exc
        if "abbreviation_detector" not in nlp.pipe_names:
            nlp.add_pipe("abbreviation_detector", last=True)
    return nlp


def build_entity_index(
    chunks: Sequence[Mapping[str, Any]] | Sequence[str],
    *,
    nlp: Any | None = None,
    model_name: str = "en_core_web_sm",
    enable_abbreviations: bool = True,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract and link entities using only ``chunks``.

    ``nlp`` is injectable solely for deterministic tests.  Production callers
    leave it unset, which strictly loads the configured spaCy model.
    """

    if nlp is None:
        nlp = _load_nlp(model_name, enable_abbreviations)
    rows = _chunks(chunks)
    dsu = _DisjointSet()
    # norm -> all observed surface forms, labels, and chunk ids
    surfaces: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, set[str]] = defaultdict(set)
    chunks_by_surface: dict[str, set[str]] = defaultdict(set)
    explicit_pairs: list[tuple[str, str, str]] = []

    for row in rows:
        doc = nlp(row["text"])
        mentions = getattr(doc, "ents", ())
        observed: set[str] = set()
        for mention in mentions:
            surface = str(getattr(mention, "text", "")).strip()
            norm = normalize_entity_name(surface)
            if not norm:
                continue
            observed.add(norm)
            dsu.add(norm)
            surfaces[norm].add(surface)
            chunks_by_surface[norm].add(row["id"])
            label = str(getattr(mention, "label_", "") or "")
            if label:
                labels[norm].add(label)
        for left, right in _explicit_alias_pairs(row["text"]):
            left_norm, right_norm = normalize_entity_name(left), normalize_entity_name(right)
            # Link aliases only when at least one side was independently
            # recognised by NER.  This avoids a regex-only entity dictionary.
            if left_norm in observed or right_norm in observed:
                dsu.add(left_norm)
                dsu.add(right_norm)
                dsu.union(left_norm, right_norm)
                surfaces[left_norm].add(left)
                surfaces[right_norm].add(right)
                chunks_by_surface[left_norm].add(row["id"])
                chunks_by_surface[right_norm].add(row["id"])
                explicit_pairs.append((left_norm, right_norm, row["id"]))
        # scispaCy's abbreviation detector is part of the Novel substrate
        # configuration.  It exposes links through ``doc._.abbreviations``;
        # retain those links even when the short form is not itself a NER span,
        # just as the existing substrate processor does.
        extensions = getattr(doc, "_", None)
        for abbreviation in getattr(extensions, "abbreviations", ()) if extensions is not None else ():
            short = str(getattr(abbreviation, "text", "")).strip()
            long_obj = getattr(getattr(abbreviation, "_", None), "long_form", None)
            long = str(getattr(long_obj, "text", "")).strip()
            left_norm, right_norm = normalize_entity_name(long), normalize_entity_name(short)
            if not left_norm or not right_norm or left_norm == right_norm:
                continue
            # A model-produced abbreviation is itself sufficient evidence to
            # create the corpus-only alias relation.
            dsu.add(left_norm)
            dsu.add(right_norm)
            dsu.union(left_norm, right_norm)
            surfaces[left_norm].add(long)
            surfaces[right_norm].add(short)
            chunks_by_surface[left_norm].add(row["id"])
            chunks_by_surface[right_norm].add(row["id"])
            explicit_pairs.append((left_norm, right_norm, row["id"]))

    groups: dict[str, set[str]] = defaultdict(set)
    for norm in dsu.parent:
        groups[dsu.find(norm)].add(norm)
    entities: list[dict[str, Any]] = []
    for group_norms in groups.values():
        if not any(norm in surfaces for norm in group_norms):
            continue
        # Prefer the longest observed form as canonical, then lexical order.
        canonical_norm = sorted(group_norms, key=lambda x: (-len(x), x))[0]
        canonical_surface = sorted(surfaces.get(canonical_norm, {canonical_norm}), key=lambda x: (-len(x), x.casefold(), x))[0]
        aliases: set[str] = set()
        chunk_ids: set[str] = set()
        entity_labels: set[str] = set()
        for norm in group_norms:
            aliases.update(surfaces.get(norm, set()))
            chunk_ids.update(chunks_by_surface.get(norm, set()))
            entity_labels.update(labels.get(norm, set()))
        aliases.add(canonical_surface)
        entities.append(
            {
                "entity_id": "",
                "canonical": canonical_surface,
                "aliases": sorted(aliases, key=lambda x: (normalize_entity_name(x), x)),
                "chunk_ids": sorted(chunk_ids),
                "entity_type": sorted(entity_labels)[0] if entity_labels else None,
            }
        )
    entities.sort(key=lambda row: (normalize_entity_name(row["canonical"]), row["canonical"]))
    for index, entity in enumerate(entities, start=1):
        entity["entity_id"] = f"e{index:06d}"

    manifest: dict[str, Any] = {
        "version": VERSION,
        "normalization": "NFKC + casefold + whitespace + surrounding punctuation",
        "source": {
            "chunk_count": len(rows),
            "source_chunks_sha256": source_chunks_sha256(source_path) if source_path else None,
            "source_path": str(source_path) if source_path else None,
            "questions_used": False,
            "answers_used": False,
            "gold_evidence_used": False,
        },
        "ner": {
            "model": model_name,
            "enable_abbreviations": bool(enable_abbreviations),
            "spacy_version": _package_version("spacy"),
            "scispacy_version": _package_version("scispacy") if enable_abbreviations else None,
        },
        "entities": entities,
    }
    # The content hash is useful for pairing an EntityHop run with exactly one
    # substrate without exposing any internal ID to the LLM.
    content = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    manifest["entity_index_sha256"] = hashlib.sha256(content).hexdigest()
    return manifest


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_entity_index(index: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

