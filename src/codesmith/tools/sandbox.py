"""Docker sandbox — the single most security-sensitive file in the project.

Read docs/ROADMAP.md section 1.2 before changing anything here. The 10
security rules listed there are NOT negotiable. Any change that relaxes
isolation without understanding the threat model is a bug, even if it
"makes tests pass".

Threat model: the LLM is adversarial-by-accident. It may:
  - generate infinite loops / fork bombs
  - try to open network sockets
  - try to read /etc/passwd, /home/*, ~/.ssh, etc.
  - consume all memory
  - write malware payloads into files
  - be prompt-injected via web_search results to do any of the above

Our answer:
  1. --network none          → no sockets to anywhere
  2. read_only=True          → no root-fs writes (only /work and /tmp tmpfs)
  3. only /work is mounted   → LLM can't see the host file system
  4. mem_limit               → can't OOM the host
  5. nano_cpus               → can't pin a CPU
  6. pids_limit              → no fork bomb
  7. cap_drop=ALL            → no Linux capabilities
  8. user=nobody             → not root inside container
  9. wall-clock timeout      → hard kill after N seconds
 10. no --privileged, ever   → not even for "debugging"
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codesmith.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from codesmith.config import SandboxConfig
    from codesmith.session import Session

log = logging.getLogger(__name__)

# Try to import docker lazily so unit tests without docker still pass
try:
    import docker  # type: ignore
    _DOCKER_AVAILABLE = True
except ImportError:
    _DOCKER_AVAILABLE = False


SCRIPT_FILENAME = "_codesmith_run.py"


class SandboxTool(BaseTool):
    """`execute_python` — run a Python snippet in an isolated container.

    The session's workspace directory is mounted at /work inside the
    container. Any files the LLM has written via the filesystem tool
    will be visible there. Any files the script produces stay in the
    workspace after execution and can be read back.
    """

    name = "execute_python"
    description = (
        "Execute a Python script in a sandboxed environment. "
        "The script runs with no network access, limited memory, and a timeout. "
        "It can read and write files in the current working directory (/work). "
        "Returns stdout, stderr, and the exit code. "
        "Use this to TEST your code, not just to think about it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute. Will be saved to /work/_codesmith_run.py and run with `python`.",
            },
        },
        "required": ["code"],
    }

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        if not _DOCKER_AVAILABLE:
            raise RuntimeError(
                "docker SDK not installed. Run: pip install docker"
            )
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def execute(self, session: Session, **kwargs: Any) -> ToolResult:
        code = kwargs.get("code", "")
        if not code.strip():
            return ToolResult(
                ok=False, content="", error="empty code argument"
            )

        # Run the blocking docker stuff in a worker thread — we're async.
        try:
            return await asyncio.to_thread(self._run_sync, code, session.workspace_dir)
        except Exception as e:
            log.exception("sandbox failed")
            return ToolResult(
                ok=False,
                content="",
                error=f"sandbox error: {type(e).__name__}: {e}",
            )

    def _run_sync(self, code: str, workspace: Path) -> ToolResult:
        """Synchronous part, intended to run in asyncio.to_thread."""
        client = self._get_client()

        # 1. Write the script into the workspace. It will be visible as
        #    /work/_codesmith_run.py inside the container.
        workspace.mkdir(parents=True, exist_ok=True)
        # Make workspace writable by container user `nobody` (uid 65534).
        # We intentionally widen perms — the workspace is already isolated
        # per-session on the host, so this is not a meaningful weakening.
        with suppress(PermissionError):
            os.chmod(workspace, 0o777)

        script_path = workspace / SCRIPT_FILENAME
        script_path.write_text(code, encoding="utf-8")
        with suppress(PermissionError):
            os.chmod(script_path, 0o666)

        # 2. Check image exists
        try:
            client.images.get(self.config.image)
        except docker.errors.ImageNotFound:
            return ToolResult(
                ok=False,
                content="",
                error=(
                    f"sandbox image '{self.config.image}' not found. "
                    f"Build it: docker build -t {self.config.image} "
                    f"-f docker/sandbox.Dockerfile docker/"
                ),
            )

        # 3. Run the container with ALL the safety flags.
        container = client.containers.run(
            image=self.config.image,
            command=["python", f"{self.config.workspace_mount}/{SCRIPT_FILENAME}"],
            volumes={
                str(workspace.resolve()): {
                    "bind": self.config.workspace_mount,
                    "mode": "rw",
                },
            },
            working_dir=self.config.workspace_mount,
            network_mode=self.config.network,          # rule 1: "none"
            mem_limit=f"{self.config.memory_mb}m",     # rule 4
            nano_cpus=int(self.config.cpus * 1e9),     # rule 5
            pids_limit=self.config.pids_limit,         # rule 6
            cap_drop=["ALL"],                          # rule 7
            security_opt=["no-new-privileges:true"],
            read_only=True,                            # rule 2
            user="nobody",                             # rule 8
            tmpfs={"/tmp": "size=64m,uid=65534"},      # writable /tmp
            detach=True,
            stdout=True,
            stderr=True,
            remove=False,
        )

        timed_out = False
        exit_code = -1
        try:
            # 4. Wait with a wall-clock timeout. If exceeded, kill.
            try:
                result = container.wait(timeout=self.config.timeout_seconds)
                exit_code = int(result.get("StatusCode", -1))
            except Exception:
                # docker-py raises requests.exceptions.ReadTimeout or
                # requests.exceptions.ConnectionError on timeout. We treat
                # any wait failure as timeout to be safe.
                timed_out = True
                with suppress(Exception):
                    container.kill()

            stdout_bytes = container.logs(stdout=True, stderr=False) or b""
            stderr_bytes = container.logs(stdout=False, stderr=True) or b""
        finally:
            try:
                container.remove(force=True)
            except Exception:
                log.warning("failed to remove container %s", container.id)

        stdout = _safe_decode(stdout_bytes)
        stderr = _safe_decode(stderr_bytes)

        if timed_out:
            return ToolResult(
                ok=False,
                content=_format_output(stdout, stderr, exit_code, timed_out=True),
                error=f"timeout after {self.config.timeout_seconds}s",
                metadata={
                    "exit_code": -1,
                    "timed_out": True,
                    "stdout": stdout,
                    "stderr": stderr,
                },
            )

        ok = exit_code == 0
        return ToolResult(
            ok=ok,
            content=_format_output(stdout, stderr, exit_code),
            error=None if ok else f"non-zero exit: {exit_code}",
            metadata={
                "exit_code": exit_code,
                "timed_out": False,
                "stdout": stdout,
                "stderr": stderr,
            },
        )


# ============================================================
# helpers
# ============================================================


def _safe_decode(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="replace")
    except Exception:
        return repr(b)


def _format_output(stdout: str, stderr: str, exit_code: int, timed_out: bool = False) -> str:
    """Make one readable block for the LLM."""
    parts: list[str] = []
    if timed_out:
        parts.append("STATUS: TIMEOUT (killed by sandbox)")
    else:
        parts.append(f"EXIT CODE: {exit_code}")
    if stdout:
        parts.append(f"STDOUT:\n{stdout.rstrip()}")
    if stderr:
        parts.append(f"STDERR:\n{stderr.rstrip()}")
    if not stdout and not stderr and not timed_out:
        parts.append("(no output)")
    return "\n\n".join(parts)
