"""LLM prompt strings for urgency triage (classification + rationale, not customer replies)."""

from __future__ import annotations

_TRIAGE_RULES = (
    "You are an internal triage analyst. Your job is only to judge whether the ticket is "
    "Urgent or Normal for routing—do not write a reply to the customer, apologize, or "
    "offer solutions.\n"
    "On the first line, output exactly one word: Urgent or Normal.\n"
    "Then skip a line and briefly explain why (2–4 short sentences, internal reasoning only)."
)

_PRIORITY_TAIL = (
    "On the second-to-last line, write exactly one of: PRIORITY: Urgent or PRIORITY: Normal "
    "(must match the first line).\n"
    "On the very last line, write: CONFIDENCE: X% where X is your certainty as an integer (e.g. CONFIDENCE: 82%)."
)


def build_llm_prompts(tweet: str, rag_context: str) -> dict[str, str]:
    """Keys match analysis pipeline: llm_zero_shot, llm_non_rag, llm_rag."""
    ctx = rag_context or "(no context retrieved)"
    return {
        "llm_zero_shot": (
            "Is this customer support tweet Urgent or Normal?\n"
            "Reply with exactly two lines:\n"
            "Line 1: Urgent or Normal\n"
            "Line 2: CONFIDENCE: X% (your certainty as an integer, e.g. CONFIDENCE: 82%)\n\n"
            f"Tweet: {tweet}"
        ),
        "llm_non_rag": (
            f"{_TRIAGE_RULES}\n"
            f"{_PRIORITY_TAIL}\n\n"
            f"Tweet:\n{tweet}"
        ),
        "llm_rag": (
            "Use the following similar past tickets as context for your urgency judgment only.\n\n"
            f"Context:\n{ctx}\n\n"
            f"{_TRIAGE_RULES}\n"
            "Reference the context when it helps justify the classification.\n"
            f"{_PRIORITY_TAIL}\n\n"
            f"Tweet:\n{tweet}"
        ),
    }
