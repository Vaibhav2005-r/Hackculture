#!/bin/sh
# Container / hosted entry point: seed demo data on first start, then serve the API + website.
set -e
cd /app/backend
if [ ! -f data/lipisetu.db ]; then
  echo "First start: seeding demo data (runs real OCR on the sample folios)..."
  python -m scripts.seed
fi
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips "*"
