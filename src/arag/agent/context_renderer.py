"""Deterministic compact context for the Agentic RAG policy loop.

The upstream A-RAG protocol still receives an ordinary OpenAI-style message
list.  This renderer changes only how that list is rebuilt between policy
turns: the stable system prompt, a compact state summary, and the most recent
assistant/tool exchange are sent.  Older exchanges remain in the audit and
trajectory artifacts, but are not resent verbatim on every request.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence


RUNTIME_CONTEXT_RENDERER_VERSION = "arag-agent-context-compact-v1"


class RuntimeContextRenderer:
    """Render a bounded, deterministic request from :class:`AgentContext`."""

    def __init__(self, *, card_chars: int = 8000, recent_turns: int = 1, max_evidence_cards: int = 32):
        self.card_chars = max(256, int(card_chars))
        self.recent_turns = max(1, int(recent_turns))
        self.max_evidence_cards = max(1, int(max_evidence_cards))

    @staticmethod
    def _format_args(arguments: Any) -> str:
        if isinstance(arguments, Mapping):
            pairs = [f"{key}={arguments[key]!r}" for key in sorted(arguments)]
            return ", ".join(pairs)
        return repr(arguments)

    def _state_text(self, query: str, context, *, exclude_loop: int | None) -> str:
        snapshot = context.runtime_snapshot(
            card_chars=self.card_chars,
            exclude_loop=exclude_loop,
            recent_turns=self.recent_turns,
            max_cards=self.max_evidence_cards,
        )
        lines = [
            f"[AGENT CONTEXT / {RUNTIME_CONTEXT_RENDERER_VERSION}]",
            f"Question: {query}",
            "Use only the observed retrieval evidence below. Search/hop results are snippets; read a useful chunk before answering.",
            "[OBSERVED ACTIONS]",
        ]
        actions = snapshot.get("action_history") or []
        if not actions:
            lines.append("none yet")
        for turn in actions:
            calls = turn.get("calls") or []
            if not calls:
                lines.append(f"Turn {turn.get('loop', '?')}: no tool call")
                continue
            rendered = []
            for call in calls:
                name = call.get("tool_name", "unknown_tool")
                rendered.append(f"{name}({self._format_args(call.get('arguments', {}))})")
            result = turn.get("result_summary") or ""
            line = f"Turn {turn.get('loop', '?')}: " + "; ".join(rendered) + (f" -> {result}" if result else "")
            if turn.get("assistant_note"):
                line += f"; note={turn['assistant_note']}"
            lines.append(line)

        lines += ["[UNIQUE EVIDENCE CARDS]"]
        cards = snapshot.get("evidence_cards") or []
        if not cards:
            lines.append("none yet")
        for card in cards:
            cid = card.get("chunk_id", "unknown")
            kind = card.get("kind", "snippet")
            source = card.get("source_tool", "retrieval")
            text = str(card.get("text", "")).strip()
            lines.append(f"Chunk {cid} [{kind}; source={source}]")
            if text:
                lines.append(text)
        omitted = snapshot.get("omitted_evidence_cards", 0)
        if omitted:
            lines.append(f"{omitted} older/low-priority evidence cards are available in the trajectory artifact but omitted from this request.")

        lines += [
            "[RETRIEVAL STATE]",
            f"Read chunk IDs: {', '.join(snapshot.get('read_chunk_ids') or []) or 'none'}",
            f"Retrieved tokens: {snapshot.get('total_retrieved_tokens', 0)}",
        ]
        return "\n".join(lines)

    def render(
        self,
        system_prompt: str,
        query: str,
        context,
        continuation: Sequence[Mapping[str, Any]] | None = None,
        *,
        recent_loop: int | None = None,
    ) -> list[dict[str, Any]]:
        """Build the next model request without replaying old tool results."""
        continuation = list(continuation or ())
        has_state = bool(context.action_history or context.evidence_cards)
        user_content = query if not has_state and not continuation else self._state_text(
            query, context, exclude_loop=recent_loop if continuation else None
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        messages.extend(copy.deepcopy(list(continuation)[-2 * self.recent_turns :]))
        return messages
