"""Portable filesystem naming helpers.

Runtime identifiers are logical database keys and may contain characters such
as ``:`` that are legal in POSIX filenames but forbidden on Windows.  Artifact
directories therefore use a deterministic portable projection while the
original identifier remains unchanged inside JSON records.
"""

from __future__ import annotations

import hashlib
import re

_PORTABLE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_UNSAFE_RUN = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "CLOCK$",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
)
_MAX_COMPONENT_LENGTH = 120


def portable_path_component(identifier: str) -> str:
    """Return a deterministic filename component valid on Linux and Windows.

    Already-portable short identifiers are preserved for backward
    compatibility.  Other values receive a readable slug plus a case-sensitive
    SHA-256 suffix, preventing collisions on Windows' case-insensitive default
    filesystems.  The logical identifier itself is never changed.
    """

    _validate_logical_identifier(identifier)
    if _is_portable(identifier):
        return identifier

    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:12]
    slug = _UNSAFE_RUN.sub("_", identifier).strip(" ._-")
    if not slug or _reserved_windows_stem(slug):
        slug = "item"
    maximum_slug_length = _MAX_COMPONENT_LENGTH - len(digest) - 2
    slug = slug[:maximum_slug_length].rstrip(" .") or "item"
    return f"{slug}--{digest}"


def _is_portable(value: str) -> bool:
    return (
        len(value) <= _MAX_COMPONENT_LENGTH
        and _PORTABLE_COMPONENT.fullmatch(value) is not None
        and not value.endswith((".", " "))
        and not _reserved_windows_stem(value)
    )


def _reserved_windows_stem(value: str) -> bool:
    return value.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES


def _validate_logical_identifier(identifier: str) -> None:
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("identifier must be a non-empty string")
    if (
        identifier in {".", ".."}
        or "/" in identifier
        or "\\" in identifier
        or "\x00" in identifier
    ):
        raise ValueError("identifier must be one safe logical path component")


__all__ = ["portable_path_component"]
