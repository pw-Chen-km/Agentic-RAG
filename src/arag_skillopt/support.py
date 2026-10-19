"""Evaluator-only, exact normalized character-span HotpotQA alignment.

No embeddings, fuzzy matching, denominator reduction, or runtime GT injection.
Normalization preserves every non-whitespace character (NFKC/casefold).
Ambiguous normalized text shared by distinct source sentences is unscorable.
"""

import hashlib
import json
import re
import unicodedata

import regex


def normalize(text):
    return "".join(
        c for c in unicodedata.normalize("NFKC", text).casefold() if not c.isspace()
    )


def raw_offsets(text):
    """Map normalized characters back to original grapheme spans.

    Keep combining accents/Hangul sequences intact. Check equivalence to whole
    text NFKC before trusting the offsets; no guessed positions.
    """
    chars, starts, ends = [], [], []
    for match in regex.finditer(r"\X", text):
        for c in unicodedata.normalize("NFKC", match.group()).casefold():
            if not c.isspace():
                chars.append(c)
                starts.append(match.start())
                ends.append(match.end())
    return (starts, ends) if "".join(chars) == normalize(text) else None


def text_tokens(text):
    return re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFKC", text).casefold())


def contains_sentence(piece, sentence, snippet=False):
    # Word boundaries prevent 'York' matching 'Yorkshire'. Preserve internal
    # punctuation/numbers/negation; tolerate only tool-stripped end punctuation.
    needle = text_tokens(sentence)
    if snippet:
        while needle and needle[-1] in ".!?":
            needle.pop()
    haystack = text_tokens(piece)
    return bool(needle) and any(
        haystack[i : i + len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def chunk_records(chunks):
    return [
        {"id": str(c["id"]), "text": c["text"]}
        if isinstance(c, dict)
        else {"id": c.split(":", 1)[0], "text": c.split(":", 1)[1]}
        for c in chunks
    ]


def build_support_map(source, chunks):
    corpus = {c["id"]: c["text"] for c in chunk_records(chunks)}
    if len(corpus) != len(chunks):
        raise ValueError("duplicate A-RAG chunk ID")
    normalized_chunks = {cid: normalize(text) for cid, text in corpus.items()}
    offsets = {}
    sources = {}
    for q in source:
        for title, sentences in q["context"]:
            for pos, sentence in enumerate(sentences):
                text = normalize(sentence)
                sources.setdefault(text, set()).add((title, pos))
    questions = {}
    for q in source:
        qid = str(q.get("_id", q.get("id")))
        if qid in questions:
            raise ValueError(f"duplicate source question ID: {qid}")
        docs = dict(q["context"])
        facts = []
        for title, pos in dict.fromkeys((t, p) for t, p in q["supporting_facts"]):
            fact = {"title": title, "sentence_id": pos, "matches": []}
            if (
                title not in docs
                or not isinstance(pos, int)
                or pos < 0
                or pos >= len(docs[title])
            ):
                fact["status"] = "invalid_coordinate"
            else:
                fact["text"] = docs[title][pos]
                needle = normalize(fact["text"])
                if not needle:
                    fact["status"] = "empty_sentence"
                elif len(sources.get(needle, set())) != 1:
                    fact["status"] = "ambiguous_source_text"
                else:
                    for cid, haystack in normalized_chunks.items():
                        start = haystack.find(needle)
                        while start >= 0:
                            if cid not in offsets:
                                offsets[cid] = raw_offsets(corpus[cid])
                            positions = offsets[cid]
                            if positions is None:
                                break
                            fact["matches"].append(
                                {
                                    "chunk_id": cid,
                                    "normalized_start": start,
                                    "normalized_end": start + len(needle),
                                    "raw_chunk_start": positions[0][start],
                                    "raw_chunk_end": positions[1][
                                        start + len(needle) - 1
                                    ],
                                }
                            )
                            start = haystack.find(needle, start + 1)
                    fact["status"] = "mapped" if fact["matches"] else "unmapped"
            facts.append(fact)
        questions[qid] = {
            "question": q["question"],
            "facts": facts,
            "evaluable": bool(facts) and all(f["status"] == "mapped" for f in facts),
        }
    return {
        "version": "hotpotqa-arag-exact-v1",
        "normalization": "NFKC-casefold-drop-whitespace-v1",
        "source_sha256": digest(source),
        "chunks_sha256": digest(chunks),
        "questions": questions,
    }


class SupportTracker:
    def __init__(self, entry=None):
        self.entry = entry or {"facts": [], "evaluable": False}
        self.visible, self.read = set(), set()

    def snapshot(self):
        n = len(self.entry["facts"])
        available = self.entry["evaluable"]
        return {
            "evaluable": available,
            "gold_fact_count": n,
            "visible_hits": len(self.visible) if available else None,
            "read_hits": len(self.read) if available else None,
            "visible_recall": len(self.visible) / n if available else None,
            "read_recall": len(self.read) / n if available else None,
            "unavailable_reasons": sorted(
                {f["status"] for f in self.entry["facts"] if f["status"] != "mapped"}
            )
            if n
            else ["no_gold_mapping"],
        }

    def observe(self, units):
        hits = []
        for unit in units:
            if unit["kind"] not in {"snippet", "full"}:
                continue
            # Snippet omission markers are barriers; never join words across them.
            pieces = (
                unit["text"].split(" ... ")
                if unit["kind"] == "snippet"
                else [unit["text"]]
            )
            for index, fact in enumerate(self.entry["facts"]):
                if fact["status"] != "mapped" or not any(
                    m["chunk_id"] == unit["chunk_id"] for m in fact["matches"]
                ):
                    continue
                # A-RAG's sentence split removes terminal .!? from snippets.
                # Only that known delimiter loss is tolerated; no word deletion.
                if any(
                    contains_sentence(piece, fact["text"], unit["kind"] == "snippet")
                    for piece in pieces
                ):
                    hits.append(
                        {
                            "fact_index": index,
                            "chunk_id": unit["chunk_id"],
                            "kind": unit["kind"],
                            "gold_title": fact["title"],
                            "gold_sentence_id": fact["sentence_id"],
                            "gold_text": fact["text"],
                            "delivered_body": unit["text"],
                            "mapped_chunk_spans": [
                                m
                                for m in fact["matches"]
                                if m["chunk_id"] == unit["chunk_id"]
                            ],
                        }
                    )
                    self.visible.add(index)
                    if unit["kind"] == "full":
                        self.read.add(index)
        return hits
