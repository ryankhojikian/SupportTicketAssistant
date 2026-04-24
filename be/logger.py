"""Structured JSON-lines logger for every analysis call.

Writes to /app/logs/backend.log (Docker volume) with a fallback to
<repo-root>/logs/backend.log for local development.  Every log record is a
single JSON object on its own line so it can be streamed, grepped, or parsed
by any log aggregator.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Resolve log directory
# ---------------------------------------------------------------------------

def _resolve_log_dir() -> Path:
    """Return /app/logs inside Docker or <repo-root>/logs locally."""
    docker_path = Path("/app/logs")
    if docker_path.parent.exists() and os.access(str(docker_path.parent), os.W_OK):
        return docker_path
    # Fall back to repo-root/logs
    return Path(__file__).resolve().parent.parent / "logs"


LOG_DIR = _resolve_log_dir()
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "backend.log"


# ---------------------------------------------------------------------------
# JSON-lines file handler
# ---------------------------------------------------------------------------

class _JsonLinesHandler(logging.FileHandler):
    """Writes each LogRecord as a compact JSON line."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "msg": record.getMessage(),
            }
            # Merge any extra dict passed as the `extra` kwarg
            if hasattr(record, "data"):
                payload.update(record.data)  # type: ignore[attr-defined]
            line = json.dumps(payload, default=str, ensure_ascii=False)
            self.stream.write(line + "\n")
            self.flush()
        except Exception:  # noqa: BLE001
            self.handleError(record)


# ---------------------------------------------------------------------------
# Build the logger
# ---------------------------------------------------------------------------

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("support_assistant")
    if logger.handlers:
        return logger  # already initialised (e.g. during hot-reload)

    logger.setLevel(logging.DEBUG)

    # JSON lines → file
    fh = _JsonLinesHandler(str(LOG_FILE), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Human-readable → stderr (visible in `docker compose logs`)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(sh)

    logger.propagate = False
    return logger


logger = _build_logger()


# ---------------------------------------------------------------------------
# Public helpers — thin wrappers so callers never touch `logging` directly
# ---------------------------------------------------------------------------

def log_query_start(query: str) -> None:
    logger.info("query_start: %s", query[:120], extra={"data": {"event": "query_start", "query": query}})


def log_rag_retrieval(query: str, documents: list[str], distances: list[float]) -> None:
    hits = [
        {"rank": i + 1, "similarity": round(1 - d, 4), "distance": round(d, 4), "snippet": doc[:120]}
        for i, (doc, d) in enumerate(zip(documents, distances))
    ]
    logger.info(
        "rag_retrieval: %d results for query '%s…'",
        len(hits),
        query[:60],
        extra={"data": {"event": "rag_retrieval", "query": query, "hits": hits}},
    )


def log_system_result(
    system: str,
    *,
    label: str | None,
    confidence: float | None,
    latency_ms: float,
    cost_usd: float,
    error: str | None,
    raw_output: str | None = None,
) -> None:
    level = logging.ERROR if error else logging.INFO
    logger.log(
        level,
        "system_result [%s]: label=%s latency=%.1fms cost=$%.6f error=%s",
        system,
        label,
        latency_ms,
        cost_usd,
        error,
        extra={
            "data": {
                "event": "system_result",
                "system": system,
                "label": label,
                "confidence": confidence,
                "latency_ms": round(latency_ms, 3),
                "cost_usd": round(cost_usd, 6),
                "raw_output": (raw_output or "")[:300],
                "error": error,
            }
        },
    )


def log_query_complete(query: str, total_latency_ms: float, had_errors: bool) -> None:
    logger.info(
        "query_complete: total_latency=%.1fms had_errors=%s",
        total_latency_ms,
        had_errors,
        extra={
            "data": {
                "event": "query_complete",
                "query": query,
                "total_latency_ms": round(total_latency_ms, 3),
                "had_errors": had_errors,
            }
        },
    )


def log_error(context: str, error: str) -> None:
    logger.error(
        "error in %s: %s",
        context,
        error[:300],
        extra={"data": {"event": "error", "context": context, "error": error}},
    )
