"""Parse Skill blocks and fail closed when a stage crosses its boundary."""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass

SECTIONS = ("retrieval_policy", "recovery_policy", "answer_policy", "fixed_runtime_guidance", "fixed_answer_contract")
MARKERS = {s: (f"<!-- {s.upper()}_START -->", f"<!-- {s.upper()}_END -->") for s in SECTIONS}

@dataclass(frozen=True)
class SkillSections:
    text: str
    blocks: dict[str, str]
    hashes: dict[str, str]

    @classmethod
    def parse(cls, text: str) -> "SkillSections":
        blocks = {}
        for name, (start, end) in MARKERS.items():
            a, b = text.count(start), text.count(end)
            if a == 0 and b == 0: continue
            if a != 1 or b != 1 or text.index(start) >= text.index(end): raise ValueError(f"invalid markers for {name}")
            blocks[name] = text.split(start, 1)[1].split(end, 1)[0]
        return cls(text, blocks, {k: hashlib.sha256(v.encode()).hexdigest() for k, v in blocks.items()})

def validate_stage_patch(old: str, new: str, stage: str) -> None:
    """Require exactly the stage's editable blocks to change, otherwise fail."""
    allowed = {"retrieval": {"retrieval_policy", "recovery_policy"}, "meta": {"retrieval_policy", "recovery_policy"}, "answer": {"answer_policy"}}.get(stage)
    if allowed is None: raise ValueError(f"unknown stage: {stage}")
    before, after = SkillSections.parse(old), SkillSections.parse(new)
    if set(before.blocks) != set(after.blocks): raise ValueError("section structure changed")
    def protected(text: str) -> str:
        for name in allowed:
            start, end = MARKERS[name]
            text = re.sub(re.escape(start) + r".*?" + re.escape(end), start + end, text, flags=re.S)
        return text
    if protected(old) != protected(new):
        raise ValueError("patch changed text outside editable blocks")
    for name in before.blocks:
        if name not in allowed and before.hashes[name] != after.hashes[name]: raise ValueError(f"fixed section modified: {name}")
    if before.text == new: return
    if not any(before.hashes.get(n) != after.hashes.get(n) for n in allowed): raise ValueError("patch changed text outside editable blocks")
