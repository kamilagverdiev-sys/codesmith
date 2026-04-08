"""Pytest configuration.

Lets tests run against the source tree without requiring a full
`pip install -e .`. Useful during early development when the package
isn't installed yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src/ to sys.path so `import codesmith.tools.filesystem` works
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
