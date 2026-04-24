"""Request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TweetIn(BaseModel):
    tweet: str = Field(..., min_length=1, description="User support tweet / ticket text")


class AnalyzeResponse(BaseModel):
    tweet: str
    rag_retrieval: dict[str, Any]
    methods: dict[str, Any]
    summary_table: list[dict[str, Any]]
