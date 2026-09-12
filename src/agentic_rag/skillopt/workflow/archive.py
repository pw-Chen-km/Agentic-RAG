"""Content addressed archive for SkillOpt candidates and replay receipts."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Mapping

def skill_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

class CandidateArchive:
    def __init__(self, root: str | Path):
        self.root = Path(root); (self.root / 'skills' / 'by_hash').mkdir(parents=True, exist_ok=True)
        (self.root / 'candidates').mkdir(parents=True, exist_ok=True)
    def save_skill(self, text: str) -> str:
        digest = skill_hash(text); p = self.root/'skills'/'by_hash'/digest/'skill.md'; p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists(): p.write_text(text, encoding='utf-8')
        elif p.read_text(encoding='utf-8') != text: raise ValueError('skill hash collision or corrupted archive')
        return digest
    def register(self, candidate_id: str, record: Mapping[str, Any]) -> Path:
        if not candidate_id or '/' in candidate_id: raise ValueError('invalid candidate id')
        p=self.root/'candidates'/f'{candidate_id}.json'; payload=dict(record)
        if p.exists() and json.loads(p.read_text(encoding='utf-8')) != payload: raise FileExistsError(str(p))
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8'); return p
