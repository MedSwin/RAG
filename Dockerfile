# Build the clinician UI once and serve the generated bundle from FastAPI.
FROM node:22-alpine AS web-builder
WORKDIR /web
COPY web/package.json ./
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

# Supported Python runtime. The application and CI target Python 3.12; using an
# older interpreter here would make Docker a different execution contract from
# the tested local/operator path.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8100 \
    MONGODB_DB=medswin \
    MONGODB_DATABASE=medswin

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .
COPY --from=web-builder /web/dist /app/web/dist

RUN mkdir -p /app/models /app/data /app/logs /app/storage

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=5 \
    CMD curl --fail --silent http://127.0.0.1:8100/health >/dev/null || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]