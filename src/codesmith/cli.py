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
from typing import Annotated, Any

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
from codesmith.config import Config, LLMConfig, load_config
from codesmith.llm import LLMRouter
from codesmith.loops.self_repair import SelfRepairLoop
from codesmith.model_selection import (
    ResolvedModelProfile,
    UnknownModelProfileError,
    default_model_profile_key,
    list_installed_ollama_models,
    list_model_profiles,
    pull_target_for_selection,
    resolve_model_profile,
)
from codesmith.personas import ARCHITECT, CODER, Persona, list_personas
from codesmith.run_logger import RunLogger, build_run_record
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


def _build_agent(
    config_path: Path,
    model_profile: str | None = None,
    persona: Persona | None = None,
) -> tuple[Agent, Session, Config, ResolvedModelProfile]:
    """Build a ready-to-run Agent + fresh Session.

    When `persona` is provided, its system prompt replaces the default,
    and — if the persona is declared no_tools — the tool registry is
    left empty so the LLM cannot even see tool schemas. This is how the
    CLI `plan` command runs ARCHITECT in a tool-less sandbox.
    """
    config = load_config(config_path)
    _setup_logging(config.logging.level)

    try:
        resolved = resolve_model_profile(
            config,
            model_profile,
            list_installed_ollama_models(config),
        )
    except UnknownModelProfileError as e:
        raise typer.Exit(str(e)) from e
    router = LLMRouter(primary=resolved.primary, fallbacks=resolved.fallbacks)

    registry = ToolRegistry()
    if persona is None or not persona.no_tools:
        registry.register(SandboxTool(config.sandbox))
        if config.tools.filesystem.enabled:
            for t in default_filesystem_tools():
                registry.register(t)

    agent_kwargs: dict[str, Any] = {
        "config": config,
        "llm": router,
        "registry": registry,
    }
    if persona is not None:
        agent_kwargs["system_prompt"] = persona.system_prompt
    agent = Agent(**agent_kwargs)

    session = Session.create(workspace_root=config.tools.filesystem.workspace_root)
    session.metadata["model_profile"] = resolved.key
    session.metadata["resolved_model"] = f"{resolved.primary.provider}/{resolved.primary.model}"
    if persona is not None:
        session.metadata["persona"] = persona.key
    return agent, session, config, resolved


@app.command()
def chat(
    message: Annotated[str, typer.Argument(help="Your message to the agent")],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
    profile: Annotated[str | None, typer.Option("--profile")] = None,
) -> None:
    """One-shot chat. The agent may use tools; no outer retry loop."""
    agent, session, _cfg, resolved = _build_agent(config, profile)

    console.print(Rule("[bold cyan]Codesmith chat"))
    console.print(
        f"[dim]session={session.session_id[:8]}  "
        f"workspace={session.workspace_dir}  "
        f"model={resolved.primary.provider}/{resolved.primary.model}[/dim]\n"
    )

    session.add_user(message)
    started = time.monotonic()
    error: str | None = None
    run = None
    try:
        run = asyncio.run(agent.run(session))
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        console.print(f"[red]{error}[/red]")
    finally:
        _log_run(
            resolved=resolved,
            session=session,
            run=run,
            message=message,
            started=started,
            error=error,
            caller="cli.chat",
        )

    if run is None:
        raise typer.Exit(1)

    console.print(Panel(run.final_text or "(no text)", title="Response", border_style="green"))
    _print_stats(steps=len(run.steps), tokens=run.total_tokens, hit_limit=run.hit_limit)


@app.command()
def solve(
    task: Annotated[str, typer.Argument(help="Task for the agent to solve with code")],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
    max_attempts: Annotated[int, typer.Option("--attempts")] = 5,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
) -> None:
    """Agentic task with the self-repair loop.

    Use this when the task should end in "code that runs and produces
    the right result". The loop retries on failure with error context.
    """
    agent, session, _cfg, resolved = _build_agent(config, profile)
    loop = SelfRepairLoop(agent=agent, max_attempts=max_attempts, require_success_exec=True)

    console.print(Rule("[bold cyan]Codesmith solve"))
    console.print(
        f"[dim]session={session.session_id[:8]}  "
        f"workspace={session.workspace_dir}  "
        f"model={resolved.primary.provider}/{resolved.primary.model}[/dim]\n"
    )

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
    profile: Annotated[str | None, typer.Option("--profile")] = None,
) -> None:
    """Show effective configuration + live health checks."""
    cfg = load_config(config)
    installed = list_installed_ollama_models(cfg)
    resolved = resolve_model_profile(cfg, profile, installed)
    console.print(Rule("[bold cyan]Codesmith info"))

    # ---- Config summary ----
    cfg_table = Table.grid(padding=(0, 2))
    cfg_table.add_column(style="dim", justify="right")
    cfg_table.add_column()
    cfg_table.add_row("profile", f"[bold]{resolved.key}[/bold]")
    cfg_table.add_row("primary llm", f"[bold]{resolved.primary.provider}/{resolved.primary.model}[/bold]")
    if resolved.primary.api_base:
        cfg_table.add_row("api_base", resolved.primary.api_base)
    if resolved.primary.provider.lower() == "ollama":
        cfg_table.add_row("api key", "[green]not required[/green]")
    else:
        cfg_table.add_row(
            "api key",
            "[green]set[/green]" if resolved.primary.api_key else "[red]missing - check .env[/red]",
        )
    cfg_table.add_row(
        "fallbacks",
        ", ".join(f"{f.provider}/{f.model}" for f in resolved.fallbacks) or "[dim]none[/dim]",
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

    for name, ok, detail in _health_checks(cfg, resolved):
        mark = "[green]ok[/green]" if ok else "[red]FAIL[/red]"
        health.add_row(name, mark, detail)
    console.print(Panel(health, title="health", border_style="cyan"))

    profiles = list_model_profiles(cfg, installed)
    profile_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    profile_table.add_column("profile", style="cyan")
    profile_table.add_column("primary")
    profile_table.add_column("status")
    profile_table.add_column("detail", style="dim")
    for item in profiles:
        if item.available and "fallback ready" in item.availability:
            status = "[yellow]partial[/yellow]"
        elif item.available:
            status = "[green]ready[/green]"
        else:
            status = "[red]missing[/red]"
        label = f"[bold]{item.key}[/bold]" if item.key == resolved.key else item.key
        profile_table.add_row(
            label,
            f"{item.primary_provider}/{item.primary_model}",
            status,
            item.availability,
        )
    console.print(Panel(profile_table, title="profiles", border_style="cyan"))


@app.command()
def models(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
) -> None:
    """List model profiles and discovered local Ollama model tags."""
    cfg = load_config(config)
    installed = list_installed_ollama_models(cfg)
    current = default_model_profile_key(cfg)

    console.print(Rule("[bold cyan]Codesmith models"))

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("profile", style="cyan")
    table.add_column("primary")
    table.add_column("status")
    table.add_column("detail", style="dim")
    for item in list_model_profiles(cfg, installed):
        if item.available and "fallback ready" in item.availability:
            status = "[yellow]partial[/yellow]"
        elif item.available:
            status = "[green]ready[/green]"
        else:
            status = "[red]missing[/red]"
        label = f"[bold]{item.key}[/bold]" if item.key == current else item.key
        table.add_row(
            label,
            f"{item.primary_provider}/{item.primary_model}",
            status,
            item.availability,
        )
    console.print(Panel(table, title="profiles", border_style="cyan"))

    local = ", ".join(installed) if installed else "[dim]none discovered[/dim]"
    console.print(Panel(local, title="installed ollama models", border_style="cyan"))


@app.command()
def plan(
    task: Annotated[str, typer.Argument(help="Task for the architect to plan")],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
    profile: Annotated[str | None, typer.Option("--profile")] = None,
) -> None:
    """Architect mode: produce a numbered, executable plan without touching files.

    The ARCHITECT persona has no tools — it only writes the plan. Use
    `codesmith solve` (with the self-repair loop) or `codesmith chat`
    afterwards to actually execute the plan step by step.
    """
    agent, session, _cfg, resolved = _build_agent(config, profile, persona=ARCHITECT)
    agent.max_iterations = 3  # plans are short, no need to burn budget

    console.print(Rule("[bold cyan]Codesmith architect"))
    console.print(
        f"[dim]session={session.session_id[:8]}  "
        f"model={resolved.primary.provider}/{resolved.primary.model}  "
        f"persona=architect[/dim]\n"
    )

    session.add_user(task)
    started = time.monotonic()
    run = None
    error: str | None = None
    try:
        run = asyncio.run(agent.run(session))
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        console.print(f"[red]{error}[/red]")
    finally:
        _log_run(
            resolved=resolved,
            session=session,
            run=run,
            message=task,
            started=started,
            error=error,
            caller="cli.plan",
        )

    if run is None:
        raise typer.Exit(1)

    console.print(
        Panel(
            run.final_text or "[dim](no plan)[/dim]",
            title="plan",
            border_style="cyan",
        )
    )
    _print_stats(steps=len(run.steps), tokens=run.total_tokens, hit_limit=run.hit_limit)


@app.command()
def execute(
    plan_file: Annotated[
        Path | None,
        typer.Option("--plan", help="Path to a file containing the plan text."),
    ] = None,
    plan_text: Annotated[
        str | None,
        typer.Argument(help="Plan text to execute (alternative to --plan file)."),
    ] = None,
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
    profile: Annotated[str | None, typer.Option("--profile")] = None,
) -> None:
    """Execute a plan using the CODER persona.

    Provide the plan as a positional argument or via --plan FILE. If
    neither is given, reads from stdin. The CODER persona follows the
    plan step by step using tools.
    """
    text: str | None = plan_text
    if text is None and plan_file is not None:
        text = plan_file.read_text(encoding="utf-8")
    if text is None:
        # Read from stdin.
        console.print("[dim]reading plan from stdin (paste plan, then Ctrl+D)...[/dim]")
        text = sys.stdin.read()
    if not text or not text.strip():
        raise typer.Exit("No plan text provided.")

    agent, session, _cfg, resolved = _build_agent(config, profile, persona=CODER)

    console.print(Rule("[bold cyan]Codesmith execute"))
    console.print(
        f"[dim]session={session.session_id[:8]}  "
        f"model={resolved.primary.provider}/{resolved.primary.model}  "
        f"persona=coder[/dim]\n"
    )

    user_message = (
        "Execute this plan step by step. Do not deviate. Use tools.\n\n" + text
    )
    session.add_user(user_message)
    started = time.monotonic()
    run = None
    error: str | None = None
    try:
        run = asyncio.run(agent.run(session))
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        console.print(f"[red]{error}[/red]")
    finally:
        _log_run(
            resolved=resolved,
            session=session,
            run=run,
            message=user_message[:200],
            started=started,
            error=error,
            caller="cli.execute",
        )

    if run is None:
        raise typer.Exit(1)

    console.print(
        Panel(
            run.final_text or "[dim](no output)[/dim]",
            title="coder result",
            border_style="green",
        )
    )
    _print_stats(steps=len(run.steps), tokens=run.total_tokens, hit_limit=run.hit_limit)


@app.command()
def personas(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
) -> None:
    """List the available agent personas."""
    load_config(config)  # validate config early so misconfiguration fails fast
    console.print(Rule("[bold cyan]Codesmith personas"))
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("key", style="cyan")
    table.add_column("label")
    table.add_column("tools")
    table.add_column("description", style="dim")
    for p in list_personas():
        tools = "[yellow]no tools[/yellow]" if p.no_tools else "[green]tools on[/green]"
        table.add_row(p.key, p.label, tools, p.description)
    console.print(Panel(table, title="personas", border_style="cyan"))


@app.command()
def repl(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
    profile: Annotated[str | None, typer.Option("--profile")] = None,
) -> None:
    """Interactive multi-turn chat with one persistent session.

    Type your message and press Enter to send. Special commands:
        /reset    — drop the current session and start fresh
        /info     — show short status line
        /quit     — exit
    """
    agent, session, _cfg, resolved = _build_agent(config, profile)

    console.print(Rule("[bold cyan]Codesmith REPL"))
    console.print(
        f"[dim]session={session.session_id[:8]}  "
        f"workspace={session.workspace_dir}  "
        f"model={resolved.primary.provider}/{resolved.primary.model}[/dim]"
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
            agent, session, _cfg, resolved = _build_agent(config, profile)
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
        started = time.monotonic()
        error: str | None = None
        run = None
        try:
            run = asyncio.run(agent.run(session))
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            console.print(f"[red]{error}[/red]\n")
        finally:
            _log_run(
                resolved=resolved,
                session=session,
                run=run,
                message=line,
                started=started,
                error=error,
                caller="cli.repl",
            )

        if run is None:
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


@app.command(name="pull-model")
def pull_model(
    model: Annotated[
        str | None,
        typer.Argument(help="Ollama model tag, e.g. qwen2.5-coder:7b. Defaults to the one in config.yaml."),
    ] = None,
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("config.yaml"),
    profile: Annotated[str | None, typer.Option("--profile")] = None,
) -> None:
    """Pull an Ollama model. Convenience wrapper around `ollama pull`.

    Useful when `codesmith info` reports the model as missing — for example
    after Ollama Desktop wiped its model store, or on a fresh machine.
    """
    cfg = load_config(config)
    installed = list_installed_ollama_models(cfg)
    if model:
        target = model
    else:
        try:
            target = pull_target_for_selection(cfg, profile, installed)
        except UnknownModelProfileError as e:
            raise typer.Exit(str(e)) from e
    ollama_exe = shutil.which("ollama")
    if ollama_exe is None:
        # Try the default winget install location.
        candidate = Path.home() / "AppData/Local/Programs/Ollama/ollama.exe"
        if candidate.exists():
            ollama_exe = str(candidate)
    if ollama_exe is None:
        raise typer.Exit(
            "ollama executable not found in PATH. "
            "Install via: winget install --id Ollama.Ollama"
        )

    console.print(Rule("[bold cyan]ollama pull"))
    console.print(f"[dim]model={target}[/dim]\n")

    import subprocess  # local import: only used here

    rc = subprocess.call([ollama_exe, "pull", target])
    if rc != 0:
        raise typer.Exit(rc)
    console.print(f"\n[green]ok[/green] - {target} ready")


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


def _health_checks(
    cfg: Config,
    resolved: ResolvedModelProfile | None = None,
) -> list[tuple[str, bool, str]]:
    """Run quick liveness checks. Each tuple is (name, ok, detail)."""
    results: list[tuple[str, bool, str]] = []

    # Docker CLI + daemon + sandbox image
    docker_path = shutil.which("docker")
    if docker_path is None:
        fallback = Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe")
        if fallback.exists():
            docker_path = str(fallback)
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
    llm_cfg = resolved.primary if resolved is not None else cfg.llm
    if llm_cfg.provider.lower() == "ollama":
        results.append(_check_ollama(llm_cfg))
    else:
        ok = bool(llm_cfg.api_key)
        results.append(
            (
                "llm key",
                ok,
                "set" if ok else f"missing {llm_cfg.provider.upper()}_API_KEY in .env",
            )
        )

    return results


def _check_ollama(llm: LLMConfig) -> tuple[str, bool, str]:
    base = llm.api_base or "http://127.0.0.1:11434"
    try:
        import urllib.request

        with urllib.request.urlopen(base + "/api/tags", timeout=2) as resp:
            import json as _json

            data = _json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            wanted = llm.model
            if wanted in models:
                return ("ollama", True, f"{base}  · model {wanted} present")
            have = ", ".join(models) or "none"
            return (
                "ollama",
                False,
                f"{base} reachable but model '{wanted}' not pulled  "
                f"(have: {have}) — fix: codesmith pull-model {wanted}",
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


_RUN_LOGGER = RunLogger()


def _log_run(
    *,
    resolved: ResolvedModelProfile,
    session: Session,
    run: Any,
    message: str,
    started: float,
    error: str | None,
    caller: str,
) -> None:
    """Best-effort append to the JSONL run log — never raises on failure."""
    duration_ms = int((time.monotonic() - started) * 1000)
    with contextlib.suppress(Exception):
        _RUN_LOGGER.record(
            build_run_record(
                session_id=session.session_id,
                profile_key=resolved.key,
                provider=resolved.primary.provider,
                model=resolved.primary.model,
                user_message=message,
                steps=len(run.steps) if run else 0,
                tokens=getattr(run, "total_tokens", 0) if run else 0,
                hit_limit=bool(getattr(run, "hit_limit", False)) if run else False,
                duration_ms=duration_ms,
                forced_final=bool(getattr(run, "forced_final", False)) if run else False,
                error=error,
                extra={"caller": caller},
            )
        )


@app.command()
def logs(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Show the last N agent runs recorded in ~/.codesmith/logs/runs.jsonl."""
    records = _RUN_LOGGER.tail(limit)
    console.print(Rule("[bold cyan]Codesmith run logs"))
    console.print(f"[dim]file={_RUN_LOGGER.path}  shown={len(records)}[/dim]\n")

    if not records:
        console.print("[dim](no runs logged yet)[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("ts", style="dim")
    table.add_column("profile", style="cyan")
    table.add_column("model")
    table.add_column("steps", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("ms", justify="right")
    table.add_column("flags")
    table.add_column("msg", style="dim")
    for rec in records:
        flags_parts = []
        if rec.error:
            flags_parts.append("[red]error[/red]")
        if rec.hit_limit:
            flags_parts.append("[yellow]hit_limit[/yellow]")
        if rec.forced_final:
            flags_parts.append("[yellow]forced[/yellow]")
        if not flags_parts:
            flags_parts.append("[green]ok[/green]")
        table.add_row(
            rec.ts[11:19] if len(rec.ts) >= 19 else rec.ts,
            rec.profile,
            f"{rec.provider}/{rec.model}",
            str(rec.steps),
            str(rec.tokens),
            str(rec.duration_ms),
            " ".join(flags_parts),
            (rec.user_message or "").replace("\n", " ")[:60],
        )
    console.print(Panel(table, title="recent runs", border_style="cyan"))


if __name__ == "__main__":
    app()
