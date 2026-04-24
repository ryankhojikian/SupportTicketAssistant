"""Runtime ML artifact cache (filled on app startup)."""

from __future__ import annotations

from typing import Any

import joblib

from .config import MODEL_DIR

ml_state: dict[str, Any] = {}


def init_ml_state() -> None:
    ml_state.clear()
    for name, fname in (
        ("rf", "random_forest_model.joblib"),
        ("tfidf", "tfidf_vectorizer.joblib"),
        ("scaler", "scaler.joblib"),
    ):
        path = MODEL_DIR / fname
        if path.is_file():
            try:
                ml_state[name] = joblib.load(path)
            except Exception as e:  # pragma: no cover
                ml_state[name] = None
                ml_state[f"{name}_error"] = str(e)
        else:
            ml_state[name] = None
