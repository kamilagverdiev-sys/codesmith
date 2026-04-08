"""Codesmith — hybrid AI coder agent.

Import submodules directly rather than from the package root:

    from codesmith.agent import Agent
    from codesmith.session import Session
    from codesmith.config import load_config
    from codesmith.loops.self_repair import SelfRepairLoop
    from codesmith.tools.base import BaseTool, ToolResult, tool
    from codesmith.tools.registry import ToolRegistry

Root-level re-exports are intentionally omitted so that importing a
pure-stdlib submodule (e.g. ``codesmith.tools.filesystem``) doesn't
drag in litellm / docker / fastapi.
"""

__version__ = "0.1.0"
