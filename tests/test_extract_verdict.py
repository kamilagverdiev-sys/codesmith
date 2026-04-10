"""Unit tests for _extract_verdict — the regex-based review verdict parser."""

from __future__ import annotations

import pytest

from codesmith.api.main import _extract_verdict


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Standard verdicts at start of line
        ("APPROVE\n\nThis looks great.", "APPROVE"),
        ("REQUEST CHANGES\n\nPlease fix the typo.", "REQUEST CHANGES"),
        ("APPROVE WITH NITS\n\nMinor style issues.", "APPROVE WITH NITS"),
        # Case-insensitive
        ("approve\nLooks fine.", "APPROVE"),
        ("Approve with Nits\nSmall things.", "APPROVE WITH NITS"),
        ("request changes\nFix it.", "REQUEST CHANGES"),
        # Verdict embedded in markdown bold
        ("**APPROVE**\n\nShip it!", "APPROVE"),
        ("**REQUEST CHANGES**\n\nNeeds work.", "REQUEST CHANGES"),
        # Verdict after preamble text
        ("After reviewing the code, I conclude:\n\nAPPROVE", "APPROVE"),
        ("My verdict: REQUEST CHANGES", "REQUEST CHANGES"),
        # Verdict with extra whitespace
        ("APPROVE  WITH  NITS\nMinor.", "APPROVE WITH NITS"),
        # Longest match wins: "APPROVE WITH NITS" over "APPROVE"
        ("APPROVE WITH NITS", "APPROVE WITH NITS"),
        # Unknown / missing verdict
        ("Great code, no issues found.", "UNKNOWN"),
        ("", "UNKNOWN"),
        ("LGTM", "UNKNOWN"),
        # Verdict in the middle of the text
        ("I think this deserves APPROVE because...", "APPROVE"),
        # Multi-line with verdict deep in the text
        (
            "## Code Review\n\nOverall well-structured.\n\n"
            "REQUEST CHANGES\n\n- Fix the SQL injection on line 42",
            "REQUEST CHANGES",
        ),
        # Verdict with emoji prefix (LLM quirk)
        ("✅ APPROVE\n\nAll good!", "APPROVE"),
        ("⚠️ APPROVE WITH NITS\n\nMinor issues.", "APPROVE WITH NITS"),
    ],
)
def test_extract_verdict(text: str, expected: str) -> None:
    assert _extract_verdict(text) == expected
