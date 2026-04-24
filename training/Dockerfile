# ml-init: trains the classifier and seeds the vector DB on first run.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-ml.txt ./
RUN pip install -r requirements-ml.txt

RUN python -m textblob.download_corpora

# Ship the bits needed to (re)produce artifacts. Model outputs and chroma_db
# are written to bind/named volumes mounted from docker-compose.
COPY training ./training
COPY rag ./rag
COPY data ./data
COPY be ./be

CMD ["python", "training/docker_init.py"]
