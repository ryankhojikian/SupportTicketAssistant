"""
ml-init entrypoint for Docker.

On first container start:
  1. Train the Random Forest model if training/model/*.joblib are missing.
  2. Populate ChromaDB with support tickets if the collection is empty.

On subsequent starts it is a no-op (artifacts and vector store live on named
volumes and survive container recreation).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "training" / "model"
MODEL_ARTIFACTS = [
    MODEL_DIR / "random_forest_model.joblib",
    MODEL_DIR / "tfidf_vectorizer.joblib",
    MODEL_DIR / "scaler.joblib",
]

CSV_CANDIDATES = [
    ROOT / "be" / "dataset_extracted" / "twcs" / "twcs.csv",
    ROOT / "data" / "sample_amazonhelp.csv",
]

URGENT_PATTERNS = [
    "refund", "cancel", "broken", "not working", "stolen", "charged", "delivery",
    "wait", "Worst", "money", "asap", "now", "today", "tommorow", "very", "urgent",
    "missing", "quickly", "unacceptable", "dissapointed", "never arrived", "scam",
    "illegal", "waiting", "tonight", "stole", "attorney", "late", "wrong", "stuck",
    "fucking", "shit",
]


def log(msg: str) -> None:
    print(f"[ml-init] {msg}", flush=True)


def train_if_needed() -> None:
    missing = [p for p in MODEL_ARTIFACTS if not p.is_file()]
    if not missing:
        log("Model artifacts already present, skipping training.")
        return
    log(f"Missing artifacts: {[p.name for p in missing]}. Running training/train.py ...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, str(ROOT / "training" / "train.py")], cwd=str(ROOT))
    log("Training complete.")


def _clean(text: str) -> str:
    text = re.sub(r"@\w+", "", str(text))
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    return " ".join(text.split())


def _ensure_priority(df):
    from textblob import TextBlob  # imported lazily so training runs first

    if "priority" in df.columns:
        return df
    df = df.copy()
    df["text"] = df["text"].apply(_clean)
    df["sent"] = df["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["caps"] = df["text"].apply(
        lambda x: 1 if any(w.isupper() for w in str(x).split() if len(w) > 3) else 0
    )
    df["excl"] = df["text"].str.count("!")
    df["ques"] = df["text"].str.count(r"\?")
    df["is_long"] = (df["text"].str.len() > 20).astype(int)
    df["kw"] = df["text"].apply(
        lambda x: sum(1 for w in URGENT_PATTERNS if w.lower() in str(x).lower())
    )

    def _label(r):
        score = r["kw"] * 2
        score += 1 if (r["excl"] > 1 or r["ques"] > 1) else 0
        score += 1 if r["caps"] > 1 else 0
        score += 1 if r["is_long"] == 1 else 0
        score += 1 if r["sent"] < -0.3 else 0
        return 1 if score >= 3 else 0

    df["priority"] = df.apply(_label, axis=1)
    return df


def index_if_empty() -> None:
    sys.path.insert(0, str(ROOT))
    from rag.rag import collection, index_tickets_from_csv

    if collection.count() > 0:
        log(f"ChromaDB already has {collection.count()} documents, skipping indexing.")
        return

    import pandas as pd

    csv_path = next((p for p in CSV_CANDIDATES if p.is_file()), None)
    if csv_path is None:
        log("No source CSV found, skipping vector-store indexing.")
        return

    log(f"Indexing ChromaDB from {csv_path} ...")
    df = pd.read_csv(csv_path)
    df = df[df['text'].notna() & (df['text'].astype(str).str.strip() != '')].reset_index(drop=True)
    if "author_id" not in df.columns:
        df["author_id"] = "unknown"
    df = _ensure_priority(df)

    required = ["tweet_id", "text", "priority", "author_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log(f"CSV is missing required columns {missing}, skipping indexing.")
        return

    tmp_path = ROOT / "training" / "_indexable.csv"
    df[required].to_csv(tmp_path, index=False)
    try:
        index_tickets_from_csv(str(tmp_path))
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
    log(f"Indexing complete — collection now has {collection.count()} docs.")


def main() -> None:
    os.chdir(ROOT)
    train_if_needed()
    index_if_empty()
    log("Done.")


if __name__ == "__main__":
    main()
