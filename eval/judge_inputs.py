"""Shared fallback rules for deterministic scoring, prompts, and estimates."""

from __future__ import annotations


def resolve_candidate(entry: dict, override: str | None = None) -> str:
    reference = entry.get("reference_answer", "")
    candidate = override if override is not None else entry.get("candidate_answer", reference)
    return candidate if isinstance(candidate, str) and candidate.strip() else reference


def resolve_sources(entry: dict) -> list[dict]:
    sources = entry.get("source_chunks", [])
    if not isinstance(sources, list):
        return []
    return [chunk for chunk in sources if isinstance(chunk, dict)]
