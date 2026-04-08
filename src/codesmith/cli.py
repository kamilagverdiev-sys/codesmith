"""Command-line interface for Codesmith.

Commands:
    codesmith chat  "message"     — one-shot chat (no self-repair loop)
    codesmith solve "task"        — agentic task with self-repair
    codesmith repl                — interactive multi-turn chat session
    codesmith web                 — start the Web UI (FastAPI + browser)
    codesmith info                — print config + live health summary
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Annotated

# Force UTF-8 on Windows so Rich box-drawing and middle-dot don't choke on
# cp1251 console codepage. Safe no-op on POSIX. Must run before Rich Console().
if sys.platform == "win32":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from codesmith.agent import Agent
from codesmith.config import Config, load_config
from codesmith.llm import LLMRouter
from codesmith.loops.self_repair import SelfRepairLoop
from codesmith.session import Session
from codesmith.tools.filesystem import default_filesystem_tools
from codesmith.tools.registry import ToolRegistry
from codesmith.tools.sandbox import SandboxTool

app = typer.Typer(
    name="codesmith",
    help="Hybrid AI coder agent with sandboxed execution.",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def _build_agent(config_path: Path) -> tuple[Agent, Session]:
    config = load_config(config_path)
    _setup_logging(config.logging.level)

    router = LLMRouter(primary=config.llm, fallbacks=config.llm_fallbacks)

    registry = ToolRegistry()
    # Phase 1 tools
    registry.register(SandboxTool(config.sandbox))
    if config.tools.filesystem.enabled:
        for t in default_filesystem_tools():
            registry.register(t)

    agent = Agent(config=config, llm=router, registry=registry)
    session = Session.create(workspace_root=config.tools.filesystem.workspace_root)
    return agent, session


@app.command()
def chat(
    message: Annotated[str, typer.Argument(help="Your message to the agent")],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
) -> None:
    """One-shot chat. The agent may use tools; no outer retry loop."""
    agent, session = _build_agent(config)

    console.print(Rule("[bold cyan]Codesmith chat"))
    console.print(f"[dim]session={session.session_id[:8]}  "
                  f"workspace={session.workspace_dir}[/dim]\n")

    session.add_user(message)
    run = asyncio.run(agent.run(session))

    console.print(Panel(run.final_text or "(no text)", title="Response", border_style="green"))
    _print_stats(steps=len(run.steps), tokens=run.total_tokens, hit_limit=run.hit_limit)


@app.command()
def solve(
    task: Annotated[str, typer.Argument(help="Task for the agent to solve with code")],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
    max_attempts: Annotated[int, typer.Option("--attempts")] = 5,
) -> None:
    """Agentic task with the self-repair loop.

    Use this when the task should end in "code that runs and produces
    the right result". The loop retries on failure with error context.
    """
    agent, session = _build_agent(config)
    loop = SelfRepairLoop(agent=agent, max_attempts=max_attempts, require_success_exec=True)

    console.print(Rule("[bold cyan]Codesmith solve"))
    console.print(f"[dim]session={session.session_id[:8]}  "
                  f"workspace={session.workspace_dir}[/dim]\n")

    result = asyncio.run(loop.solve(session, task))

    style = "green" if result.ok else "red"
    title = "Solved" if result.ok else "Failed"
    console.print(Panel(result.final_text or "(no text)", title=title, border_style=style))
    _print_stats(
        steps=sum(len(r.steps) for r in result.runs),
        tokens=result.total_tokens,
        hit_limit=any(r.hit_limit for r in result.runs),
        attempts=result.attempts,
    )


@app.command()
def info(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
) -> None:
    """Show effective configuration + live health checks."""
    cfg = load_config(config)
    console.print(Rule("[bold cyan]Codesmith info"))

    # ---- Config summary ----
    cfg_table = Table.grid(padding=(0, 2))
    cfg_table.add_column(style="dim", justify="right")
    cfg_table.add_column()
    cfg_table.add_row("primary llm", f"[bold]{cfg.llm.provider}/{cfg.llm.model}[/bold]")
    if cfg.llm.api_base:
        cfg_table.add_row("api_base", cfg.llm.api_base)
    if cfg.llm.provider.lower() == "ollama":
        cfg_table.add_row("api key", "[green]not required[/green]")
    else:
        cfg_table.add_row(
            "api key",
            "[green]set[/green]" if cfg.llm.api_key else "[red]missing — check .env[/red]",
        )
    cfg_table.add_row(
        "fallbacks",
        ", ".join(f"{f.provider}/{f.model}" for f in cfg.llm_fallbacks) or "[dim]none[/dim]",
    )
    cfg_table.add_row("sandbox image", cfg.sandbox.image)
    cfg_table.add_row(
        "sandbox limits",
        f"{cfg.sandbox.memory_mb}MB · {cfg.sandbox.cpus} cpu · "
        f"{cfg.sandbox.timeout_seconds}s · network={cfg.sandbox.network}",
    )
    cfg_table.add_row("memory", "enabled" if cfg.memory.enabled else "[dim]disabled[/dim]")
    cfg_table.add_row(
        "web search",
        "enabled" if cfg.tools.web_search.enabled else "[dim]disabled[/dim]",
    )
    console.print(Panel(cfg_table, title="config", border_style="cyan"))

    # ---- Live health checks ----
    health = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    health.add_column("component", style="dim")
    health.add_column("status")
    health.add_column("detail", style="dim")

    for name, ok, detail in _health_checks(cfg):
        mark = "[green]ok[/green]" if ok else "[red]FAIL[/red]"
        health.add_row(name, mark, detail)
    console.print(Panel(health, title="health", border_style="cyan"))


@app.command()
def repl(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
) -> None:
    """Interactive multi-turn chat with one persistent session.

    Type your message and press Enter to send. Special commands:
        /reset    — drop the current session and start fresh
        /info     — show short status line
        /quit     — exit
    """
    agent, session = _build_agent(config)

    console.print(Rule("[bold cyan]Codesmith REPL"))
    console.print(
        f"[dim]session={session.session_id[:8]}  "
        f"workspace={session.workspace_dir}[/dim]"
    )
    console.print(
        "[dim]commands: /reset · /info · /quit · empty line repeats prompt[/dim]\n"
    )

    while True:
        try:
            line = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return

        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            console.print("[dim]bye[/dim]")
            return
        if line == "/reset":
            session.destroy()
            agent, session = _build_agent(config)
            console.print(f"[dim]new session={session.session_id[:8]}[/dim]\n")
            continue
        if line == "/info":
            console.print(
                f"[dim]session={session.session_id[:8]}  "
                f"messages={len(session.messages)}  "
                f"~tokens={session.token_estimate()}[/dim]\n"
            )
            continue

        session.add_user(line)
        try:
            run = asyncio.run(agent.run(session))
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]{type(e).__name__}: {e}[/red]\n")
            continue

        console.print()
        console.print(
            Panel(
                run.final_text or "[dim](no text)[/dim]",
                title="codesmith",
                border_style="magenta",
            )
        )
        _print_stats(steps=len(run.steps), tokens=run.total_tokens, hit_limit=run.hit_limit)
        console.print()


@app.command()
def web(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
    host: Annotated[str, typer.Option("--host", "-h")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p")] = 8000,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Start the Web UI (FastAPI on uvicorn). Opens the browser by default.

    The browser tab points at http://HOST:PORT/ which serves the
    single-page chat UI built into the codesmith.api package.
    """
    # Validate config eagerly so we fail fast with a friendly message,
    # rather than letting uvicorn crash inside its own lifespan.
    cfg_path = config.resolve()
    load_config(cfg_path)

    # Pass the absolute config path to the API server via env var.
    os.environ["CODESMITH_CONFIG"] = str(cfg_path)

    url = f"http://{host}:{port}/"
    console.print(Rule("[bold cyan]Codesmith Web UI"))
    console.print(f"[dim]config={cfg_path}[/dim]")
    console.print(f"[bold]→ {url}[/bold]\n")

    if open_browser and not reload:
        # Slight delay so uvicorn binds before the browser hits it.
        def _open() -> None:
            time.sleep(0.8)
            with contextlib.suppress(Exception):
                webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    try:
        import uvicorn
    except ImportError as e:
        raise typer.Exit(
            "uvicorn is not installed. Run scripts/bootstrap.ps1 first."
        ) from e

    uvicorn.run(
        "codesmith.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def _health_checks(cfg: Config) -> list[tuple[str, bool, str]]:
    """Run quick liveness checks. Each tuple is (name, ok, detail)."""
    results: list[tuple[str, bool, str]] = []

    # Docker CLI + daemon + sandbox image
    docker_path = shutil.which("docker")
    if docker_path is None:
        results.append(("docker", False, "docker CLI not in PATH"))
    else:
        try:
            import docker as docker_lib

            client = docker_lib.from_env()
            client.ping()
            try:
                img = client.images.get(cfg.sandbox.image)
                size_mb = (img.attrs.get("Size") or 0) // (1024 * 1024)
                results.append(("docker", True, "daemon ok"))
                results.append(
                    ("sandbox image", True, f"{cfg.sandbox.image}  ({size_mb} MB)")
                )
            except Exception:  # noqa: BLE001
                results.append(("docker", True, "daemon ok"))
                results.append(
                    (
                        "sandbox image",
                        False,
                        f"{cfg.sandbox.image} missing — run scripts/build-sandbox.ps1",
                    )
                )
        except Exception as e:  # noqa: BLE001
            results.append(("docker", False, f"daemon down: {type(e).__name__}"))

    # LLM provider liveness
    if cfg.llm.provider.lower() == "ollama":
        results.append(_check_ollama(cfg))
    else:
        ok = bool(cfg.llm.api_key)
        results.append(
            (
                "llm key",
                ok,
                "set" if ok else f"missing {cfg.llm.provider.upper()}_API_KEY in .env",
            )
        )

    return results


def _check_ollama(cfg: Config) -> tuple[str, bool, str]:
    base = cfg.llm.api_base or "http://localhost:11434"
    try:
        import urllib.request

        with urllib.request.urlopen(base + "/api/tags", timeout=2) as resp:
            import json as _json

            data = _json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            wanted = cfg.llm.model
            if wanted in models:
                return ("ollama", True, f"{base}  · model {wanted} present")
            return (
                "ollama",
                False,
                f"{base} reachable but model {wanted} not pulled  "
                f"(have: {', '.join(models) or 'none'})",
            )
    except Exception as e:  # noqa: BLE001
        return ("ollama", False, f"{base} unreachable: {type(e).__name__}")


def _print_stats(steps: int, tokens: int, hit_limit: bool, attempts: int | None = None) -> None:
    parts = [f"steps={steps}", f"tokens={tokens}"]
    if attempts is not None:
        parts.append(f"attempts={attempts}")
    if hit_limit:
        parts.append("[yellow]hit iteration limit[/yellow]")
    console.print("[dim]" + "  ".join(parts) + "[/dim]")


if __name__ == "__main__":
    app()
