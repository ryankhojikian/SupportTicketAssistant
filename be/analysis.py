"""Four-way support-tweet analysis: ML + three LLM paths."""

from __future__ import annotations

import re
import time
from typing import Any

import google.generativeai as genai
import numpy as np
from textblob import TextBlob

from rag.rag import retrieve_support_context

from .config import (
    COST_PER_1K_INPUT_TOKENS,
    COST_PER_1K_OUTPUT_TOKENS,
    MODEL_ID,
    URGENT_PATTERNS,
    get_gemini_api_key,
)
from .logger import (
    log_error,
    log_query_complete,
    log_query_start,
    log_rag_retrieval,
    log_system_result,
)
from .metrics import record_query_result
from .prompts import build_llm_prompts
from .state import ml_state


def clean_text(text: str) -> str:
    """Same cleaning as training/train.py for consistent TF-IDF."""
    text = re.sub(r"@\w+", "", str(text))
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    return " ".join(text.split())


def calculate_llm_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        (prompt_tokens / 1000) * COST_PER_1K_INPUT_TOKENS
        + (completion_tokens / 1000) * COST_PER_1K_OUTPUT_TOKENS
    )


def extract_priority_from_text(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"PRIORITY:\s*(Urgent|Normal)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    m2 = re.search(r"\b(Urgent|Normal)\b", text.strip(), re.IGNORECASE)
    if m2:
        return m2.group(1).capitalize()
    return None


def extract_confidence_from_text(text: str) -> float | None:
    """Parse a CONFIDENCE: X% line from an LLM response into a 0–1 float."""
    if not text:
        return None
    m = re.search(r"CONFIDENCE:\s*(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1)) / 100.0
    return round(min(max(val, 0.0), 1.0), 4)


def build_ml_features(query_raw: str) -> tuple[np.ndarray | None, str | None]:
    rf = ml_state.get("rf")
    tfidf = ml_state.get("tfidf")
    scaler = ml_state.get("scaler")
    if rf is None or tfidf is None or scaler is None:
        return None, "ML artifacts not loaded (run training/train.py to create training/model/*.joblib)."

    query = clean_text(query_raw)
    sentiment = TextBlob(query).sentiment.polarity
    has_caps = 1 if any(w.isupper() for w in query.split() if len(w) > 3) else 0
    excl = query.count("!")
    word_count = len(query.split())
    keywords = sum(1 for word in URGENT_PATTERNS if word.lower() in query.lower())
    negativity = -sentiment
    X_text_q = tfidf.transform([query]).toarray()
    X_meta_q = scaler.transform([[negativity, word_count, has_caps, excl, keywords]])
    return np.hstack((X_text_q, X_meta_q)), None


def run_ml_prediction(X: np.ndarray) -> dict[str, Any]:
    rf = ml_state["rf"]
    pred = int(rf.predict(X)[0])
    proba = rf.predict_proba(X)[0]
    conf = float(np.max(proba))
    label = "Urgent" if pred == 1 else "Normal"
    return {"label": label, "confidence": conf, "class_index": pred}


def _response_text(response) -> str:
    try:
        return (response.text or "").strip()
    except Exception:
        parts: list[str] = []
        for c in getattr(response, "candidates", None) or []:
            content = getattr(c, "content", None)
            for p in getattr(content, "parts", None) or []:
                t = getattr(p, "text", None)
                if t:
                    parts.append(t)
        return "\n".join(parts).strip()


def _usage_cost(response) -> tuple[int, int, float]:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return 0, 0, 0.0
    pt = int(getattr(meta, "prompt_token_count", 0) or 0)
    ct = int(
        getattr(meta, "candidates_token_count", None)
        or getattr(meta, "output_token_count", None)
        or 0
    )
    return pt, ct, calculate_llm_cost(pt, ct)


def _serialize_chroma(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {"documents": [], "metadatas": [], "distances": [], "ids": []}
    out: dict[str, Any] = {}
    for key in ("documents", "metadatas", "distances", "ids"):
        v = raw.get(key)
        if v and isinstance(v, list) and len(v) > 0:
            row = v[0]
            if key == "distances" and row is not None:
                out[key] = [float(x) for x in row]
            elif key == "metadatas" and row is not None:
                out[key] = [
                    {k: (int(val) if k == "priority" and val is not None else val) for k, val in (m or {}).items()}
                    for m in row
                ]
            else:
                out[key] = list(row) if row is not None else []
        else:
            out[key] = []
    return out


def _finalize_response(tweet: str, results: dict[str, Any]) -> dict[str, Any]:
    ml_test_acc: float | None = (ml_state.get("metrics") or {}).get("ml_test_accuracy")
    rows = [
        {
            "system": "ML (Random Forest)",
            "priority": results["ml"].get("label"),
            "confidence": results["ml"].get("confidence"),
            "latency_ms": results["ml"]["latency_ms"],
            "cost_usd": results["ml"]["cost_usd"],
            "test_accuracy": ml_test_acc,
            "error": results["ml"].get("error"),
        },
        {
            "system": "LLM zero-shot",
            "priority": results["llm_zero_shot"].get("label"),
            "confidence": results["llm_zero_shot"].get("confidence"),
            "latency_ms": results["llm_zero_shot"]["latency_ms"],
            "cost_usd": results["llm_zero_shot"]["cost_usd"],
            "test_accuracy": None,
            "error": results["llm_zero_shot"].get("error"),
        },
        {
            "system": "LLM non-RAG",
            "priority": results["llm_non_rag"].get("label"),
            "confidence": results["llm_non_rag"].get("confidence"),
            "latency_ms": results["llm_non_rag"]["latency_ms"],
            "cost_usd": results["llm_non_rag"]["cost_usd"],
            "test_accuracy": None,
            "error": results["llm_non_rag"].get("error"),
        },
        {
            "system": "LLM RAG",
            "priority": results["llm_rag"].get("label"),
            "confidence": results["llm_rag"].get("confidence"),
            "latency_ms": results["llm_rag"]["latency_ms"],
            "cost_usd": results["llm_rag"]["cost_usd"],
            "test_accuracy": None,
            "error": results["llm_rag"].get("error"),
        },
    ]
    return {
        "tweet": tweet,
        "rag_retrieval": results.get("rag_retrieval", {}),
        "methods": {
            "ml": results["ml"],
            "llm_zero_shot": results["llm_zero_shot"],
            "llm_non_rag": results["llm_non_rag"],
            "llm_rag": results["llm_rag"],
        },
        "summary_table": rows,
    }


def analyze_support_tweet(tweet: str) -> dict[str, Any]:
    """
    Four-way comparison: ML classifier, LLM zero-shot, LLM non-RAG, LLM RAG
    (see prompts.build_llm_prompts).
    Every step is persisted to /app/logs/backend.log as structured JSON lines.
    """
    tweet = tweet.strip()
    results: dict[str, Any] = {}
    wall_start = time.perf_counter()

    log_query_start(tweet)

    # ------------------------------------------------------------------
    # ML prediction
    # ------------------------------------------------------------------
    start_ml = time.perf_counter()
    X_q, ml_err = build_ml_features(tweet)
    ml_latency = (time.perf_counter() - start_ml) * 1000
    if X_q is not None:
        pr = run_ml_prediction(X_q)
        results["ml"] = {
            "label": pr["label"],
            "confidence": pr["confidence"],
            "raw_output": None,
            "latency_ms": round(ml_latency, 3),
            "cost_usd": 0.0,
            "error": None,
        }
    else:
        results["ml"] = {
            "label": None,
            "confidence": None,
            "raw_output": None,
            "latency_ms": round(ml_latency, 3),
            "cost_usd": 0.0,
            "error": ml_err,
        }
        if ml_err:
            log_error("ML prediction", ml_err)

    log_system_result(
        "ML (Random Forest)",
        label=results["ml"]["label"],
        confidence=results["ml"]["confidence"],
        latency_ms=results["ml"]["latency_ms"],
        cost_usd=0.0,
        error=results["ml"]["error"],
    )

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------
    try:
        rag_context, chroma_raw = retrieve_support_context(tweet, n_results=3)
        results["rag_retrieval"] = _serialize_chroma(chroma_raw)

        docs: list[str] = results["rag_retrieval"].get("documents", [])
        dists: list[float] = results["rag_retrieval"].get("distances", [])
        log_rag_retrieval(tweet, docs, dists)
    except Exception as exc:
        rag_err = str(exc)[:200]
        log_error("RAG retrieval", rag_err)
        rag_context = ""
        results["rag_retrieval"] = {"documents": [], "metadatas": [], "distances": [], "ids": []}

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------
    api_key = get_gemini_api_key()
    if not api_key:
        err = "No API key: set GOOGLE_API_KEY or GEMINI_API_KEY in .env"
        log_error("LLM setup", err)
        for key in ("llm_zero_shot", "llm_non_rag", "llm_rag"):
            results[key] = {
                "label": None,
                "confidence": None,
                "raw_output": None,
                "latency_ms": 0.0,
                "cost_usd": 0.0,
                "error": err,
            }
            log_system_result(
                key,
                label=None,
                confidence=None,
                latency_ms=0.0,
                cost_usd=0.0,
                error=err,
            )
        total_ms = (time.perf_counter() - wall_start) * 1000
        log_query_complete(tweet, total_ms, had_errors=True)
        record_query_result(
            tweet,
            total_ms,
            True,
            {
                "ML (Random Forest)": results["ml"],
                "llm_zero_shot": results["llm_zero_shot"],
                "llm_non_rag": results["llm_non_rag"],
                "llm_rag": results["llm_rag"],
            },
        )
        return _finalize_response(tweet, results)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_ID)
    prompts = build_llm_prompts(tweet, rag_context)

    for key, prompt in prompts.items():
        out_text = ""
        cost = 0.0
        t0 = time.perf_counter()
        err = None
        try:
            response = model.generate_content(prompt)
            out_text = _response_text(response)
            _, _, cost = _usage_cost(response)
        except Exception as e:
            err = str(e)[:200]
        latency_ms = (time.perf_counter() - t0) * 1000
        label = extract_priority_from_text(out_text) if not err else None
        llm_conf = extract_confidence_from_text(out_text) if not err else None
        results[key] = {
            "label": label,
            "confidence": llm_conf,
            "raw_output": out_text if not err else None,
            "latency_ms": round(latency_ms, 3),
            "cost_usd": round(cost, 6),
            "error": err,
        }
        log_system_result(
            key,
            label=label,
            confidence=llm_conf,
            latency_ms=latency_ms,
            cost_usd=cost,
            error=err,
            raw_output=out_text,
        )

    had_errors = any(results[k].get("error") for k in ("ml", "llm_zero_shot", "llm_non_rag", "llm_rag"))
    total_ms = (time.perf_counter() - wall_start) * 1000
    log_query_complete(tweet, total_ms, had_errors=had_errors)

    # Record into in-memory metrics store (keys must match _SYSTEMS in metrics.py)
    system_results_for_metrics = {
        "ML (Random Forest)": results["ml"],
        "llm_zero_shot": results["llm_zero_shot"],
        "llm_non_rag": results["llm_non_rag"],
        "llm_rag": results["llm_rag"],
    }
    record_query_result(tweet, total_ms, had_errors, system_results_for_metrics)

    return _finalize_response(tweet, results)
