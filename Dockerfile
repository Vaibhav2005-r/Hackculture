# One image: builds the React app, then serves it (and the API + OCR worker) from FastAPI.
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/backend
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
COPY backend/requirements.txt ./
# Inference only: PyTorch is NOT installed, so the CRNN engine is simply unavailable and PP-OCR reads.
RUN pip install -r requirements.txt
COPY backend/ ./
COPY --from=web /web/dist /app/frontend/dist
COPY start.sh /app/start.sh
# Bake the offline OCR models into the image so the first request does not download them.
RUN python -m scripts.warm_models
RUN mkdir -p data && chmod +x /app/start.sh
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/api/v1/health' % os.environ.get('PORT','8000')).status==200 else 1)"
CMD ["/app/start.sh"]
