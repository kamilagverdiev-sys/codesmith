"""Self-repair loop.

The Agent already retries within a single run: LLM writes code → sandbox
returns stderr → LLM sees it in tool_result and tries again. That's the
first layer of self-repair and it's free.

This module adds a SECOND layer: if the agent gives up without solving
the task (either hit its iteration limit, or wrote a plain answer
without running any code when the task clearly needed it), we restart
it with an explicit nudge pointing at the failure.

Don't use SelfRepairLoop for simple chat — use it only when you have a
task that should end in "code that runs and produces the right answer".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from codesmith.agent import Agent, AgentRun
from codesmith.session import Session
from codesmith.tools.base import ToolResult

log = logging.getLogger(__name__)


@dataclass
class RepairResult:
    ok: bool                 # did we get a successful execute_python in the end?
    attempts: int
    final_text: str
    runs: list[AgentRun]
    total_tokens: int


class SelfRepairLoop:
    """Run an Agent with external retry on failure.

    Args:
        agent: the Agent to wrap.
        max_attempts: how many independent agent runs to try.
        require_success_exec: if True, the loop only considers the task
            solved if the agent ran at least one execute_python and the
            LAST execute_python returned exit_code 0. Set False for tasks
            that don't need execution (e.g. "explain this code").
    """

    def __init__(
        self,
        agent: Agent,
        max_attempts: int = 5,
        require_success_exec: bool = True,
    ) -> None:
        self.agent = agent
        self.max_attempts = max_attempts
        self.require_success_exec = require_success_exec

    async def solve(self, session: Session, task: str) -> RepairResult:
        session.add_user(task)
        runs: list[AgentRun] = []
        total_tokens = 0

        for attempt in range(1, self.max_attempts + 1):
            log.info("self-repair attempt %d/%d", attempt, self.max_attempts)
            run = await self.agent.run(session)
            runs.append(run)
            total_tokens += run.total_tokens

            ok = self._check_success(run)
            if ok:
                return RepairResult(
                    ok=True,
                    attempts=attempt,
                    final_text=run.final_text,
                    runs=runs,
                    total_tokens=total_tokens,
                )

            if attempt < self.max_attempts:
                nudge = self._build_nudge(run)
                log.info("nudging agent with: %s", nudge[:100])
                session.add_user(nudge)

        # all attempts exhausted
        return RepairResult(
            ok=False,
            attempts=self.max_attempts,
            final_text=runs[-1].final_text if runs else "",
            runs=runs,
            total_tokens=total_tokens,
        )

    # ------------------------------------------------------------
    # internals
    # ------------------------------------------------------------

    def _check_success(self, run: AgentRun) -> bool:
        """Did this run solve the task?"""
        if not self.require_success_exec:
            return not run.hit_limit

        # Walk backwards through steps looking for the last execute_python.
        last_exec: ToolResult | None = None
        for step in reversed(run.steps):
            for _tc_id, result in step.tool_results:
                # Any execute_python with exit_code in metadata is our
                # success signal. The registry gives us the ToolResult
                # directly so we can check metadata.
                if "exit_code" in result.metadata:
                    last_exec = result
                    break
            if last_exec is not None:
                break

        if last_exec is None:
            # Never ran code — fail if we require it.
            return False

        return last_exec.ok and last_exec.metadata.get("exit_code") == 0

    def _build_nudge(self, run: AgentRun) -> str:
        """Craft a message telling the agent what went wrong."""
        if run.hit_limit:
            return (
                "You stopped without finishing the task (hit the iteration "
                "limit). Review what you've done so far, simplify your "
                "approach, and try again. Run the code with execute_python "
                "to verify it works."
            )

        # Find the last failing tool result, if any
        last_error: str | None = None
        for step in reversed(run.steps):
            for _tc_id, result in step.tool_results:
                if not result.ok and result.error:
                    last_error = result.error
                    break
            if last_error:
                break

        if last_error:
            return (
                f"Your previous attempt failed. The last error was:\n"
                f"{last_error}\n\n"
                f"Diagnose the root cause and fix it. Run the code with "
                f"execute_python to verify."
            )

        return (
            "Your previous answer didn't include a verified execution. "
            "Please write the code, run it with execute_python, and show "
            "the real output."
        )
