"""Lossless body spans from the pinned A-RAG tool output format.

Offsets are into the actual tool_result; titles, claims or unseen corpus text
are never synthesized. Unknown formats remain verbatim and unscored.
"""

import re


def result_units(tool, text):
    if tool in {"keyword_search", "semantic_search", "entity_hop", "linear_hop"}:
        pattern = (
            r"(?m)^Chunk ID: ([^,\n]+), Matched keywords in chunk: "
            if tool == "keyword_search"
            else (
                r"(?m)^Chunk ID: ([^\n]+?) \(Similarity: [^\n]+\)\nMatched: "
                if tool == "semantic_search"
                else (
                    r"(?m)^Chunk ID: ([^\n]+?) \(LinearHop score: [^\n]+\)\n"
                    r"(?:Bridge: [^\n]+\n)?Matched: "
                    if tool == "linear_hop"
                    else r"(?m)^Chunk ID: ([^,\n]+), Matched entity in chunk: "
                )
            )
        )
        matches = list(re.finditer(pattern, text))
        units = []
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body_start = match.end()
            body_end = end
            while body_end > body_start and text[body_end - 1].isspace():
                body_end -= 1
            body = text[body_start:body_end]
            units.append(
                {
                    "chunk_id": match[1],
                    "kind": "snippet",
                    "text": body,
                    "start": body_start,
                    "end": body_end,
                    "available": True,
                }
            )
        if units:
            return units, None
        if text.startswith(("No results", "No results found")):
            return [], "empty"
        return [], "unparsed"
    if tool == "read_chunk":
        pattern = r"(?m)^\[Chunk ([^\]]+)\](?: - Not found)?"
        matches = list(re.finditer(pattern, text))
        units = []
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            tail = text[match.end() : end]
            if " - Not found" in match[0]:
                units.append(
                    {
                        "chunk_id": match[1],
                        "kind": "not_found",
                        "text": "Not found",
                        "available": False,
                    }
                )
            elif tail.lstrip().startswith("(This chunk has been read before)"):
                units.append(
                    {
                        "chunk_id": match[1],
                        "kind": "already_read",
                        "text": "This chunk has been read before",
                        "available": True,
                    }
                )
            else:
                opening = re.match(r"\n-{80}\n", tail)
                if not opening:
                    return [], "unparsed"
                start = match.end() + opening.end()
                closing = text.find("\n" + "=" * 80, start, end)
                if closing < 0:
                    return [], "unparsed"
                units.append(
                    {
                        "chunk_id": match[1],
                        "kind": "full",
                        "text": text[start:closing],
                        "start": start,
                        "end": closing,
                        "available": True,
                    }
                )
        return (units, None) if units else ([], "unparsed")
    return [], "unparsed"
