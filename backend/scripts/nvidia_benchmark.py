"""Benchmark NVIDIA-hosted vision-language models as line transcribers against local engines.

Truth sets: the Navkar Mantra folio (9 lines, text known) and held-out real printed Sanskrit lines.
Reads NVIDIA_API_KEY from the environment / backend/.env.  Prints CER and seconds per line.

    python -m scripts.nvidia_benchmark --image navkar.jpg --models meta/llama-3.2-90b-vision-instruct ...
"""

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import httpx
import numpy as np
from rapidfuzz.distance import Levenshtein

from app.config import BACKEND_DIR

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TIMEOUT = 45.0
PROMPT = (
    "This image is one line of handwritten or printed Devanagari text from a historical Sanskrit/Prakrit manuscript. "
    "Transcribe exactly what is written, in Devanagari, character by character. Do not translate, explain, "
    "modernise spelling, or add anything. Output only the transcription of this one line."
)
NAVKAR = ["णमो अरिहंताणं", "णमो सिद्धाणं", "णमो आयरियाणं", "णमो उवज्झायाणं", "णमो लोए सव्वसाहूणं", "एसो पंच णमोक्कारो",
          "सव्वपावप्पणासणो", "मंगलाणं च सव्वेसिं", "पढमं हवइ मंगलं"]


def load_key():
    if "NVIDIA_API_KEY" not in os.environ:
        for line in (BACKEND_DIR / ".env").read_text().splitlines():
            if line.startswith("NVIDIA_API_KEY="):
                os.environ["NVIDIA_API_KEY"] = line.split("=", 1)[1].strip()
    return os.environ["NVIDIA_API_KEY"]


def png_b64(img) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def ask(model, img, key, prompt=PROMPT, max_tokens=300):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(img)}"}}]}]}
    t = time.time()
    for attempt in range(3):
        try:
            r = httpx.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=TIMEOUT)
        except httpx.HTTPError as exc:
            return f"[{type(exc).__name__}]", time.time() - t
        if r.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        break
    dt = time.time() - t
    if r.status_code != 200:
        return f"[HTTP {r.status_code}] {r.text[:120]}", dt
    msg = r.json()["choices"][0]["message"]
    return (msg.get("content") or "").strip(), dt


def clean(s: str) -> str:
    s = s.strip().strip("`").strip()
    for ch in "\"'“”‘’।॥|*":
        s = s.replace(ch, "")
    return " ".join(s.split())


def cer(pred, ref):
    ref = " ".join(ref.split())
    return Levenshtein.distance(pred, ref) / max(len(ref), 1)


def navkar_crops(path):
    from app.ocr.pipeline import crop_lines
    from app.ocr.preprocess import preprocess
    from app.ocr.segment import segment_lines

    pre = preprocess(cv2.imread(path))
    boxes = segment_lines(pre.binary)
    crops = crop_lines(pre.color, boxes)
    return crops[1:10] if len(crops) >= 10 else crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    key = load_key()
    crops = navkar_crops(args.image)
    print(len(crops), "line crops from", args.image)
    report = {}
    for model in args.models:
        with ThreadPoolExecutor(3) as pool:
            res = list(pool.map(lambda c: ask(model, c, key), crops))
        preds = [clean(r[0]) for r in res]
        cers = [cer(p, t) for p, t in zip(preds, NAVKAR)]
        report[model] = {"mean_cer": float(np.mean(cers)), "sec_per_line": float(np.mean([r[1] for r in res])), "preds": preds}
        print(f"\n{model}: CER {100 * np.mean(cers):.1f}%  {np.mean([r[1] for r in res]):.1f}s/line")
        for p, t, c in zip(preds, NAVKAR, cers):
            print(f"   {100 * c:5.0f}%  {p}   |  {t}")
    if args.out:
        json.dump(report, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
