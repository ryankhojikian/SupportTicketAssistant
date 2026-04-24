"""Paths, env, and static model settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

def _parse_urgent_patterns_csv(raw: str) -> list[str]:
    """Comma-separated phrases from URGENT_PATTERNS in .env (phrases must not contain commas)."""
    return [p.strip() for p in raw.split(",") if p.strip()]


_urgent_raw = (os.getenv("URGENT_PATTERNS") or "").strip()
URGENT_PATTERNS = _parse_urgent_patterns_csv(_urgent_raw)
if not URGENT_PATTERNS:
    raise RuntimeError(
        "Set URGENT_PATTERNS in .env to a comma-separated list of keyword phrases "
        "(see .env.example). No built-in default is used."
    )

MODEL_DIR = ROOT / "training" / "model"
MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3-flash-preview")

COST_PER_1K_INPUT_TOKENS = float(os.getenv("GEMINI_COST_INPUT_PER_1K", "0.0001"))
COST_PER_1K_OUTPUT_TOKENS = float(os.getenv("GEMINI_COST_OUTPUT_PER_1K", "0.0002"))


def get_gemini_api_key() -> str | None:
    return (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GEMENI_API_KEY")
    )
