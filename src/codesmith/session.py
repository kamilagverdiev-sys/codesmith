"""Session state — holds message history and workspace for one conversation.

Phase 1: in-memory only.
Phase 2: will add SQLite persistence (see ROADMAP section 2.1).
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from litellm.types.completion import ChatCompletionMessageParam as Message


@dataclass
class Session:
    """One conversation session.

    Holds the full message history in LiteLLM-compatible format and an
    isolated workspace directory on the host filesystem. The workspace is
    what gets mounted into the sandbox container.
    """

    session_id: str
    messages: list[Message]
    workspace_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(cls, workspace_root: Path, session_id: str | None = None) -> Session:
        """Create a new session with a fresh isolated workspace."""
        sid = session_id or str(uuid.uuid4())
        workspace_root = workspace_root.expanduser().resolve()
        ws = workspace_root / sid
        ws.mkdir(parents=True, exist_ok=True)
        return cls(session_id=sid, messages=[], workspace_dir=ws)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)  # type: ignore[arg-type]

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Append a tool result message linked to a specific tool_call_id."""
        self.messages.append(
            {  # type: ignore[typeddict-item]
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    def add_system(self, content: str) -> None:
        """System prompt. Should be the first message if present."""
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = {"role": "system", "content": content}
        else:
            self.messages.insert(0, {"role": "system", "content": content})

    def token_estimate(self) -> int:
        """Rough token count for compaction triggers. Real count needs tokenizer."""
        total = 0
        for m in self.messages:
            content = m.get("content") or ""
            if isinstance(content, str):
                total += len(content) // 4  # ~4 chars per token, good enough
        return total

    def destroy(self) -> None:
        """Remove workspace from disk. Call when session is really done."""
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
