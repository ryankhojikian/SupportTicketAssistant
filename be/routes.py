"""HTTP routes."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from .analysis import analyze_support_tweet
from .config import MODEL_DIR
from .logger import LOG_FILE, logger
from .metrics import get_metrics
from .schemas import AnalyzeResponse, TweetIn
from .state import ml_state

router = APIRouter()


@router.get("/health")
def health():
    ml_ok = ml_state.get("rf") is not None
    return {"status": "ok", "ml_loaded": ml_ok, "model_dir": str(MODEL_DIR)}


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(body: TweetIn):
    try:
        return analyze_support_tweet(body.tweet)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/metrics")
def metrics_endpoint() -> dict[str, Any]:
    """Aggregate run-time statistics across all /analyze calls.

    Returns per-system call counts, error counts, average latency, cumulative
    cost, and label distribution, plus a ring-buffer of the last 100 queries.
    """
    return get_metrics()


@router.get("/logs")
def logs_endpoint(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    """Return the last *limit* structured log lines from backend.log.

    Each line is a JSON object written by logger.py.  Lines that cannot be
    parsed (plain-text fallback, truncated lines, etc.) are returned as-is
    under the key ``raw``.
    """
    if not LOG_FILE.exists():
        logger.warning("logs endpoint called but log file does not exist yet")
        return {"entries": [], "log_file": str(LOG_FILE)}

    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Cannot read log file: {exc}") from exc

    # Most recent lines first, truncated to limit
    tail = lines[-limit:][::-1]

    entries: list[dict[str, Any]] = []
    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"raw": line})

    return {"entries": entries, "total_lines": len(lines), "log_file": str(LOG_FILE)}
