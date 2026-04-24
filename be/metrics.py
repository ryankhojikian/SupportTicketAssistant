"""Thread-safe in-memory metrics store.

Accumulates aggregate stats across all /analyze calls so the
GET /metrics endpoint can return them without re-parsing log files.
A ring-buffer of recent queries is also kept for the live log view.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any


_lock = threading.Lock()

_SYSTEMS = ["ML (Random Forest)", "llm_zero_shot", "llm_non_rag", "llm_rag"]
_RECENT_MAX = 100  # keep last N queries in memory


def _blank_system() -> dict[str, Any]:
    return {
        "calls": 0,
        "errors": 0,
        "total_latency_ms": 0.0,
        "total_cost_usd": 0.0,
        "label_counts": {"Urgent": 0, "Normal": 0},
    }


_store: dict[str, Any] = {
    "total_queries": 0,
    "had_errors_count": 0,
    "systems": {s: _blank_system() for s in _SYSTEMS},
    "recent_queries": deque(maxlen=_RECENT_MAX),
}


def record_query_result(
    query: str,
    total_latency_ms: float,
    had_errors: bool,
    system_results: dict[str, dict[str, Any]],
) -> None:
    """Call once per /analyze after all four systems have completed."""
    ts = datetime.now(timezone.utc).isoformat()
    labels: dict[str, str | None] = {}

    with _lock:
        _store["total_queries"] += 1
        if had_errors:
            _store["had_errors_count"] += 1

        for sys_key, result in system_results.items():
            # Normalise key: analysis.py uses e.g. "llm_zero_shot" but the
            # display name for ML is "ML (Random Forest)".
            display = sys_key if sys_key in _store["systems"] else sys_key
            if display not in _store["systems"]:
                _store["systems"][display] = _blank_system()

            s = _store["systems"][display]
            s["calls"] += 1
            s["total_latency_ms"] += result.get("latency_ms", 0.0)
            s["total_cost_usd"] += result.get("cost_usd", 0.0)

            if result.get("error"):
                s["errors"] += 1
            else:
                label = result.get("label")
                if label in ("Urgent", "Normal"):
                    s["label_counts"][label] += 1

            labels[display] = result.get("label")

        _store["recent_queries"].appendleft(
            {
                "ts": ts,
                "query_snippet": query[:120],
                "total_latency_ms": round(total_latency_ms, 1),
                "had_errors": had_errors,
                "labels": labels,
            }
        )


def get_metrics() -> dict[str, Any]:
    """Return a JSON-serialisable snapshot of all metrics."""
    with _lock:
        systems_out: dict[str, Any] = {}
        for name, s in _store["systems"].items():
            calls = s["calls"] or 1  # avoid /0
            systems_out[name] = {
                "calls": s["calls"],
                "errors": s["errors"],
                "avg_latency_ms": round(s["total_latency_ms"] / calls, 2),
                "total_latency_ms": round(s["total_latency_ms"], 2),
                "total_cost_usd": round(s["total_cost_usd"], 6),
                "label_counts": dict(s["label_counts"]),
            }
        return {
            "total_queries": _store["total_queries"],
            "had_errors_count": _store["had_errors_count"],
            "systems": systems_out,
            "recent_queries": list(_store["recent_queries"]),
        }
