from __future__ import annotations

from pathlib import Path

import pytest

from codesmith.config import SandboxConfig
from codesmith.session import Session
from codesmith.tools.sandbox import SandboxTool

docker = pytest.importorskip("docker")


def _docker_client():
    client = docker.from_env()
    try:
        client.ping()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Docker daemon unavailable: {exc}")
    return client


def _require_sandbox_image() -> None:
    client = _docker_client()
    try:
        client.images.get("codesmith-sandbox:latest")
    except docker.errors.ImageNotFound as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"sandbox image not built: {exc}")


@pytest.fixture(autouse=True)
def _check_sandbox_image() -> None:
    _require_sandbox_image()


@pytest.fixture
def session(tmp_path: Path) -> Session:
    session = Session.create(tmp_path)
    try:
        yield session
    finally:
        session.destroy()


@pytest.mark.asyncio
async def test_sandbox_executes_python(session: Session) -> None:
    tool = SandboxTool(SandboxConfig(timeout_seconds=5))
    result = await tool.execute(session, code="print('hello from sandbox')")

    assert result.ok
    assert result.metadata["exit_code"] == 0
    assert "hello from sandbox" in result.metadata["stdout"]


@pytest.mark.asyncio
async def test_sandbox_blocks_network_access(session: Session) -> None:
    tool = SandboxTool(SandboxConfig(timeout_seconds=5))
    result = await tool.execute(
        session,
        code=(
            "import urllib.request\n"
            "urllib.request.urlopen('https://example.com', timeout=2).read()\n"
        ),
    )

    assert not result.ok
    assert result.metadata["exit_code"] != 0 or result.metadata["timed_out"]


@pytest.mark.asyncio
async def test_sandbox_enforces_timeout(session: Session) -> None:
    tool = SandboxTool(SandboxConfig(timeout_seconds=1))
    result = await tool.execute(session, code="while True:\n    pass\n")

    assert not result.ok
    assert result.metadata["timed_out"] is True
