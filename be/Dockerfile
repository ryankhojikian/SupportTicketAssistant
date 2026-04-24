# Backend: FastAPI service (RAG + LLM + ML inference)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (build tools for numpy/sklearn wheels on slim)
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements-be.txt ./
RUN pip install -r requirements-be.txt

# TextBlob needs its corpora
RUN python -m textblob.download_corpora

# Pre-download the ChromaDB ONNX embedding model so it is baked into the
# image layer and never re-fetched at request time.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; ef = DefaultEmbeddingFunction(); ef(['warmup'])"

# App code. Model artifacts and chroma_db come in via volumes from the ml service.
COPY be ./be
COPY rag ./rag

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "be.main:app", "--host", "0.0.0.0", "--port", "8000"]
