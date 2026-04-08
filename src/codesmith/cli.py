"""Command-line interface for Codesmith.

Commands:
    codesmith chat  "message"     — one-shot chat (no self-repair loop)
    codesmith solve "task"        — agentic task with self-repair
    codesmith info                — print config summary
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule

from codesmith.agent import Agent
from codesmith.config import load_config
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
    """Show effective configuration (without secrets)."""
    cfg = load_config(config)
    console.print(Rule("[bold cyan]Codesmith config"))
    console.print(f"Primary LLM:   [bold]{cfg.llm.provider}/{cfg.llm.model}[/bold]")
    has_key = "yes" if cfg.llm.api_key else "[red]NO — check .env[/red]"
    console.print(f"API key:       {has_key}")
    console.print(f"Fallbacks:     {len(cfg.llm_fallbacks)}")
    for f in cfg.llm_fallbacks:
        console.print(f"  - {f.provider}/{f.model}")
    console.print(f"Sandbox image: {cfg.sandbox.image}")
    console.print(f"Sandbox limit: {cfg.sandbox.memory_mb}MB / "
                  f"{cfg.sandbox.cpus} cpu / {cfg.sandbox.timeout_seconds}s")
    console.print(f"Memory:        {'enabled' if cfg.memory.enabled else 'disabled'}")
    console.print(f"Web search:    {'enabled' if cfg.tools.web_search.enabled else 'disabled'}")


def _print_stats(steps: int, tokens: int, hit_limit: bool, attempts: int | None = None) -> None:
    parts = [f"steps={steps}", f"tokens={tokens}"]
    if attempts is not None:
        parts.append(f"attempts={attempts}")
    if hit_limit:
        parts.append("[yellow]hit iteration limit[/yellow]")
    console.print("[dim]" + "  ".join(parts) + "[/dim]")


if __name__ == "__main__":
    app()
