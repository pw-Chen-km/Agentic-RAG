"""Agent execution context for ARAG."""

import re
from typing import Any, Dict, List, Set
from dataclasses import dataclass, field


@dataclass
class RetrievalLog:
    """Log entry for a retrieval operation."""
    tool_name: str
    tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentContext:
    """Context manager for agent execution state."""
    
    def __init__(self, query: str | None = None):
        # The original question is runtime state, not a tool argument.  Query
        # conditioned graph tools can use it for ranking while the policy only
        # selects an entity to expand.
        self.query: str | None = query
        # Token statistics
        self.total_retrieved_tokens: int = 0
        self.retrieval_logs: List[RetrievalLog] = []
        
        # State management
        self.read_chunk_ids: Set[str] = set()
        self.search_history: List[Dict[str, Any]] = []
        # Compact runtime state.  Full messages/results remain in the audit;
        # these structures contain only deterministic decision/evidence cards
        # needed to rebuild the next policy request.
        self.action_history: List[Dict[str, Any]] = []
        self.evidence_cards: Dict[str, Dict[str, Any]] = {}
    
    def add_retrieval_log(
        self,
        tool_name: str,
        tokens: int,
        metadata: Dict[str, Any] = None
    ):
        """Add a retrieval log entry."""
        log = RetrievalLog(
            tool_name=tool_name,
            tokens=tokens,
            metadata=metadata or {}
        )
        self.retrieval_logs.append(log)
        self.total_retrieved_tokens += tokens
    
    def mark_chunk_as_read(self, chunk_id: str):
        """Mark chunk as read."""
        self.read_chunk_ids.add(str(chunk_id))
    
    def is_chunk_read(self, chunk_id: str) -> bool:
        """Check if chunk has been read."""
        return str(chunk_id) in self.read_chunk_ids
    
    # Aliases for backward compatibility
    def add_read_chunk(self, chunk_id: str, content: str = None):
        """Alias for mark_chunk_as_read."""
        self.mark_chunk_as_read(chunk_id)
    
    def has_read_chunk(self, chunk_id: str) -> bool:
        """Alias for is_chunk_read."""
        return self.is_chunk_read(chunk_id)
    
    def get_read_chunk(self, chunk_id: str):
        """Check if chunk was read (returns None, content not stored)."""
        return None if not self.is_chunk_read(chunk_id) else ""
    
    def reset(self):
        """Reset context for new query."""
        self.retrieval_logs = []
        self.read_chunk_ids = set()
        self.search_history = []
        self.total_retrieved_tokens = 0
        self.action_history = []
        self.evidence_cards = {}

    @staticmethod
    def _chunk_blocks(text: str):
        pattern = re.compile(r"(?:Chunk ID:\s*([^,\s\n]+)|\[Chunk\s+([^\]]+)\])")
        matches = list(pattern.finditer(str(text)))
        for index, match in enumerate(matches):
            chunk_id = str(match.group(1) or match.group(2)).strip()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(str(text))
            yield chunk_id, str(text)[match.start():end].strip()

    def record_turn(self, loop: int, assistant_message, tool_messages, tool_logs):
        """Record a policy turn in compact form, without retaining prompts."""
        assistant_message = assistant_message or {}
        tool_calls = (
            assistant_message.get("tool_calls", [])
            if isinstance(assistant_message, dict)
            else assistant_message
        )
        calls = []
        for call, message, log in zip(tool_calls or [], tool_messages or [], tool_logs or []):
            function = call.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    import json
                    arguments = json.loads(arguments)
                except (TypeError, ValueError):
                    arguments = {"unparsed_arguments": arguments}
            tool_name = function.get("name", "unknown_tool")
            metadata = dict(log or {})
            ids = metadata.get("chunk_ids") or metadata.get("new_chunks_read") or metadata.get("chunk_ids_requested") or []
            if ids:
                ids = [str(value) for value in ids]
            calls.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "chunk_ids": ids,
                "status": "error" if metadata.get("error") else metadata.get("status", "ok"),
            })
            result_text = str(message.get("content", ""))
            read_tool = tool_name == "read_chunk"
            for chunk_id, block in self._chunk_blocks(result_text):
                already = "already been read before" in block
                if already and chunk_id in self.evidence_cards:
                    continue
                kind = "full" if read_tool and not already else "snippet"
                if len(block) > 12000:
                    block = block[:12000] + "\n[card truncated; full text is in the audit artifact]"
                previous = self.evidence_cards.get(chunk_id)
                # A later full read supersedes a search snippet; otherwise keep
                # the first stable card and only refresh its provenance.
                if previous is None or (kind == "full" and previous.get("kind") != "full"):
                    self.evidence_cards[chunk_id] = {
                        "chunk_id": chunk_id,
                        "kind": kind,
                        "source_tool": tool_name,
                        "loop": loop,
                        "text": block,
                    }
                else:
                    previous["loop"] = loop
                    previous["source_tool"] = tool_name
            summary = []
            if ids:
                summary.append("chunks=" + ",".join(ids))
            if metadata.get("retrieved_tokens") is not None:
                summary.append(f"retrieved_tokens={metadata.get('retrieved_tokens', 0)}")
            calls[-1]["result_summary"] = ", ".join(summary) or calls[-1]["status"]
        note = assistant_message.get("content") if isinstance(assistant_message, dict) else None
        turn = {"loop": int(loop), "calls": calls}
        if note:
            turn["assistant_note"] = str(note)[:1000]
        self.action_history.append(turn)

    def runtime_snapshot(self, *, card_chars: int = 8000, exclude_loop: int | None = None, recent_turns: int = 1, max_cards: int | None = None):
        cards = []
        for card in self.evidence_cards.values():
            if exclude_loop is not None and card.get("loop") == exclude_loop:
                continue
            item = dict(card)
            item["text"] = str(item.get("text", ""))[:card_chars]
            cards.append(item)
        omitted = 0
        if max_cards is not None and len(cards) > max_cards:
            # Prefer complete reads, then recent evidence; tie-break by ID so
            # identical runs render byte-for-byte identically.
            ranked = sorted(
                cards,
                key=lambda item: (
                    item.get("kind") != "full",
                    -int(item.get("loop", 0)),
                    str(item.get("chunk_id", "")),
                ),
            )
            omitted = len(ranked) - int(max_cards)
            cards = ranked[: int(max_cards)]
        return {
            "action_history": self.action_history[-max(1, int(recent_turns)) * 20 :],
            "evidence_cards": cards,
            "omitted_evidence_cards": omitted,
            "read_chunk_ids": sorted(str(value) for value in self.read_chunk_ids),
            "total_retrieved_tokens": self.total_retrieved_tokens,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get context summary."""
        return {
            "total_retrieved_tokens": self.total_retrieved_tokens,
            "retrieval_logs": [
                {
                    "tool_name": log.tool_name,
                    "tokens": log.tokens,
                    "metadata": log.metadata
                }
                for log in self.retrieval_logs
            ],
            "chunks_read_count": len(self.read_chunk_ids),
            "chunks_read_ids": sorted(self.read_chunk_ids),
            "runtime_context": self.runtime_snapshot(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export context as dictionary."""
        return self.get_summary()
