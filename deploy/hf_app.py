"""Entry point for a Hugging Face *Gradio-SDK* Space.

The Space runs `python app.py` and expects a web server on port 7860.  LipiSetu is a FastAPI app, so we simply
start it here (Gradio itself is not used).  Demo data is seeded on the first start of each container.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
os.chdir(BACKEND)
sys.path.insert(0, BACKEND)

if not os.path.exists(os.path.join("data", "lipisetu.db")):
    print("First start: seeding demo data (real OCR on the sample folios)...", flush=True)
    subprocess.run([sys.executable, "-m", "scripts.seed"], check=True)

import uvicorn  # noqa: E402

uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), proxy_headers=True, forwarded_allow_ips="*")
