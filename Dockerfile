# Single image: builds the React app, then serves it (and the API + OCR worker) from FastAPI.
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 tesseract-ocr tesseract-ocr-san tesseract-ocr-hin tesseract-ocr-guj \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt
# Optional inference for the trained CRNN (CPU build): pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY backend/ ./
COPY --from=web /web/dist /app/frontend/dist
RUN mkdir -p data && useradd -m lipi && chown -R lipi /app
USER lipi
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
