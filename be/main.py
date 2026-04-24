"""
FastAPI entrypoint: wires lifespan (ML load) and API routes.
Business logic lives in analysis.py; prompts in prompts.py; HTTP in routes.py.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .logger import LOG_DIR, LOG_FILE, logger
from .routes import router as api_router
from .state import init_ml_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Backend starting — log file: %s", LOG_FILE)
    init_ml_state()
    yield


app = FastAPI(title="Support tweet decision assistant", lifespan=lifespan)
app.include_router(api_router)

_cors = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from be.analysis import analyze_support_tweet
    from be.state import init_ml_state

    init_ml_state()
    sample = "My package is 3 days late and I need a refund immediately!!"
    print(json.dumps(analyze_support_tweet(sample), indent=2, default=str))
