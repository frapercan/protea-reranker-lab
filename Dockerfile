# Runtime image for the protea-reranker-lab offline LightGBM lab.
#
# The lab pulls a frozen dataset (train.parquet / eval.parquet /
# manifest.json) from PROTEA's artifact store, trains a LightGBM
# booster, and writes runs/<run_id>/. This image ships the lab's
# scripts ready to run; PROTEA-side training endpoints are NOT exposed
# (the lab is the only training surface).

FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

FROM python:3.12-slim

# libgomp1 is needed at runtime by lightgbm.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY scripts/ ./scripts/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Library image: importable as protea_reranker_lab. Override CMD to
# launch a training run:
#   docker run protea-reranker-lab python scripts/run.py --spec ...
CMD ["python", "-c", "import protea_reranker_lab; print('protea-reranker-lab', protea_reranker_lab.__name__, 'ready')"]
