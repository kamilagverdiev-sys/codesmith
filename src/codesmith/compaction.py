"""Context compaction — summarize old messages to stay within token budget.

When a session's message history grows past the configured thresholds
(message count or estimated token count), this module replaces the oldest
portion of the conversation with a concise LLM-generated summary, keeping
recent messages intact so the agent retains fresh working context.

Strategy (Phase 1 — simple, good-enough):
  1. Keep system prompt (messages[0]) untouched.
  2. Keep the last ``keep_tail`` messages untouched (fresh context).
  3. Ask the LLM to summarize the middle "old" block.
  4. Replace the old block with a synthetic user+assistant pair carrying
     the summary.

Phase 2+ may add chunked deduplication, tool-result trimming, or
semantic compression — but this simple approach already prevents
Qwen 7B from crashing on sessions longer than ~40 turns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codesmith.config import CompactionConfig
    from codesmith.llm import LLMRouter
    from codesmith.session import Session

log = logging.getLogger(__name__)

# How many recent messages to keep verbatim (system prompt excluded).
_DEFAULT_KEEP_TAIL = 10

_SUMMARY_SYSTEM_PROMPT = """\
You are a conversation summarizer. You will receive a block of messages \
from an AI coding assistant conversation. Summarize the key facts, \
decisions, files touched, errors encountered, and current state of the \
task. Be concise — aim for 200-400 words. Use bullet points. \
Do NOT invent information. If tool results contain file contents, \
summarize what was done, not the full contents."""

_SUMMARY_USER_TEMPLATE = """\
Summarize the following conversation block so the assistant can \
continue the task without losing context:

{block}"""


def should_compact(session: Session, config: CompactionConfig) -> bool:
    """Return True when the session history is large enough to compact."""
    if not config.enabled:
        return False
    non_system = [m for m in session.messages if m.get("role") != "system"]
    if len(non_system) < _DEFAULT_KEEP_TAIL + 4:
        # Not enough messages to compact meaningfully — we need at least
        # a few messages beyond the tail we keep.
        return False
    msg_count = len(non_system)
    token_est = session.token_estimate()
    return (
        msg_count >= config.trigger_at_messages
        or token_est >= config.trigger_at_tokens
    )


def _format_message_for_summary(msg: dict[str, Any]) -> str:
    """Format a single message into readable text for the summary prompt."""
    role = msg.get("role", "unknown")
    content = msg.get("content") or ""
    if isinstance(content, str) and len(content) > 800:
        content = content[:800] + "…(truncated)"

    tool_calls = msg.get("tool_calls")
    if tool_calls and isinstance(tool_calls, list):
        tc_names = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            tc_names.append(fn.get("name", "?"))
        suffix = f" [called: {', '.join(tc_names)}]"
    else:
        suffix = ""

    tool_call_id = msg.get("tool_call_id")
    if tool_call_id:
        role = f"tool({tool_call_id[:8]})"

    return f"[{role}]{suffix} {content}"


async def compact(
    session: Session,
    llm: LLMRouter,
    config: CompactionConfig,
    *,
    keep_tail: int = _DEFAULT_KEEP_TAIL,
) -> int:
    """Summarize old messages in-place, return count of messages removed.

    Returns 0 if compaction was skipped (not enough messages, or LLM
    failure). Never raises — logs warnings on error and leaves the
    session unchanged.
    """
    messages = session.messages

    # Find where system prompt ends.
    start = 0
    if messages and messages[0].get("role") == "system":
        start = 1

    non_system = messages[start:]
    if len(non_system) <= keep_tail + 2:
        return 0

    # Split: old block (to summarize) + tail (to keep verbatim).
    split_idx = len(non_system) - keep_tail
    old_block = non_system[:split_idx]
    # tail = non_system[split_idx:]  — stays in place

    # Format old block for the summary prompt.
    block_text = "\n\n".join(
        _format_message_for_summary(m) for m in old_block
    )
    # Cap block text to avoid blowing up the summary call itself.
    if len(block_text) > 12000:
        block_text = block_text[:12000] + "\n\n…(conversation truncated for summary)"

    summary_messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": _SUMMARY_USER_TEMPLATE.format(block=block_text)},
    ]

    try:
        response = await llm.chat(
            messages=summary_messages,  # type: ignore[arg-type]
            tools=None,
        )
        summary_text = (response.content or "").strip()
    except Exception as e:  # noqa: BLE001
        log.warning("compaction LLM call failed, skipping: %s: %s", type(e).__name__, e)
        return 0

    if not summary_text:
        log.warning("compaction produced empty summary, skipping")
        return 0

    # Build replacement messages.
    summary_user: dict[str, Any] = {
        "role": "user",
        "content": (
            "[Conversation summary — earlier messages were compacted "
            f"({len(old_block)} messages removed)]\n\n{summary_text}"
        ),
    }
    summary_assistant: dict[str, Any] = {
        "role": "assistant",
        "content": "Understood. I have the context from the summary above and will continue from here.",
    }

    # Rebuild messages in-place: system + summary pair + tail.
    system_part = messages[:start]  # 0 or 1 system messages
    tail_part = messages[start + split_idx:]  # the kept tail
    session.messages[:] = system_part + [summary_user, summary_assistant] + tail_part  # type: ignore[assignment]

    removed = len(old_block)
    log.info(
        "compaction done: removed %d messages, summary %d chars, "
        "session now has %d messages (~%d tokens)",
        removed,
        len(summary_text),
        len(session.messages),
        session.token_estimate(),
    )
    return removed
