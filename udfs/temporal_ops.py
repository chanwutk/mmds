"""Deterministic temporal helpers for windowed video queries."""

from __future__ import annotations

from typing import Any


def rebase_clip_events(row: dict[str, Any]) -> dict[str, Any]:
    """Convert clip-relative event times to source-video time."""
    clip_start = row["clip"]["start"]
    events = [
        {
            **event,
            "start": clip_start + event["start"],
            "end": clip_start + event["end"],
        }
        for event in row["clip_events"]
    ]
    return {"events": events}


def collect_sorted_events(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect source-time events from several rows and sort without merging."""
    events = [
        dict(event)
        for row in rows
        for event in row["events"]
    ]
    events.sort(key=lambda event: event["start"])
    return {"events": events}


def reconcile_events(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility wrapper for the former, less explicit UDF name."""
    return collect_sorted_events(rows)
