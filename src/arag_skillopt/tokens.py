"""Explicit, reproducible text budgets; provider usage remains separate."""

import json
from functools import lru_cache

import tiktoken


@lru_cache(maxsize=4)
def encoding(name="o200k_base"):
    return tiktoken.get_encoding(name)


def count(value, name="o200k_base"):
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    )
    return len(encoding(name).encode(text))


def rule_text(rule):
    return "\n".join([rule["title"], rule["when"], *rule["action_sequence"]])
