"""Contracts the specialists must leave in their notes, and the validator reads."""

import re
from typing import Literal

NO_RELIABLE_SOURCES = "NO_RELIABLE_SOURCES"

_SENTIMENT = re.compile(
    r"JUDGMENT:\s*sentiment=(positive|negative|mixed|neutral)\s+score=(-?\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)
_VALUE = re.compile(r"JUDGMENT:\s*value=(-?\d+(?:\.\d+)?)%?\s*$", re.IGNORECASE)
_INSUFFICIENT = re.compile(r"JUDGMENT:\s*insufficient\s*$", re.IGNORECASE)

JudgmentKind = Literal["sentiment", "value", "insufficient"]
ResearchStatus = Literal["sourced", "gap", "invalid"]


def _clean_line(line: str) -> str:
    return line.strip().strip("`").strip("*").strip()


def has_no_reliable_sources(notes: str) -> bool:
    """True when a line is exactly the no-source marker."""
    return any(_clean_line(line).upper() == NO_RELIABLE_SOURCES for line in notes.splitlines())


def has_source_url(notes: str) -> bool:
    """True when the notes cite an http or https URL."""
    return "http://" in notes or "https://" in notes


def research_status(notes: str) -> ResearchStatus:
    """Classify research notes. A URL outweighs a no-source marker on another line."""
    text = notes.strip()
    if not text:
        return "invalid"
    sourced = has_source_url(text)
    gap = has_no_reliable_sources(text)
    if sourced:
        return "sourced"
    if gap:
        return "gap"
    return "invalid"


def judgment_kind(notes: str) -> JudgmentKind | None:
    """Return the last well-formed JUDGMENT line, if there is one."""
    found: JudgmentKind | None = None
    for line in notes.splitlines():
        cleaned = _clean_line(line)
        if _SENTIMENT.match(cleaned):
            found = "sentiment"
        elif _VALUE.match(cleaned):
            found = "value"
        elif _INSUFFICIENT.match(cleaned):
            found = "insufficient"
    return found
