#!/usr/bin/env python3
"""Compile A-RAG's official prompt with the formatted initial skill.

Usage:
  python scripts/prepare_prompt.py official.txt formatted.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arag_skillopt.skill_rules import load_rules, render_formatted_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_prompt")
    parser.add_argument("formatted_prompt")
    parser.add_argument("--rules", default="skills/arag_initial_rules.json")
    args = parser.parse_args()
    original = Path(args.official_prompt).read_text(encoding="utf-8")
    output = render_formatted_prompt(original, load_rules(args.rules))
    Path(args.formatted_prompt).write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
