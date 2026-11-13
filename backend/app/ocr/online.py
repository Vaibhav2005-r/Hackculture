"""Optional online reader: one vision-language-model call per page, aligned onto our own line boxes.

Local engines stay the default and always run first, so everything works offline.  When online mode is
active a second pass asks a hosted VLM (NVIDIA NIM, ``google/gemma-4-31b-it`` by default - it read a
real calligraphic Prakrit folio at about 5% CER where the local engines were far off) to transcribe
the page, then places each returned line onto the matching local line box.  Unverified lines are
replaced; verified (human-confirmed) lines are never touched.
"""

import base64
import logging
import re
import time
import unicodedata

import cv2
import httpx
import numpy as np
from rapidfuzz.distance import Levenshtein

from app.config import get_settings

log = logging.getLogger(__name__)

PROMPT = (
    "This is a page of a historical manuscript written in {script}. Transcribe every line of the main text "
    "exactly as written, in the original script, one output line per written line, top to bottom. "
    "Do not translate, explain, number the lines, or modernise spelling. Output only the transcription."
)


def _letters(line: str) -> int:
    """Indic letters in a line (the OM sign and danda marks are not letters)."""
    return sum(1 for ch in line if unicodedata.category(ch) == "Lo" and ch != "\u0950")


class OnlineError(RuntimeError):
    pass


def _encode(image: np.ndarray, max_side: int = 1600) -> str:
    h, w = image.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise OnlineError("could not encode the page image")
    return base64.b64encode(buf.tobytes()).decode()


def clean_lines(text: str) -> list[str]:
    """Model output -> transcription lines (no fences, numbering or ornament-only lines)."""
    lines = []
    for raw in text.replace("```", "\n").splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*])\s+", "", raw).strip()
        if _letters(line) >= 3:  # drops commentary and "॥ ॐ ॥" style ornaments
            lines.append(" ".join(line.split()))
    return lines


def read_page(image: np.ndarray, script: str = "Devanagari") -> list[str]:
    """Transcribe a page with the configured hosted VLM.  Raises OnlineError on any failure."""
    s = get_settings()
    if not s.nvidia_api_key or s.offline_mode:
        raise OnlineError("online reading is not available (no key, or OFFLINE_MODE is set)")
    body = {
        "model": s.nvidia_model,
        "temperature": 0,
        "max_tokens": 1200,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(script=script)},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _encode(image)}},
        ]}],
    }
    last: Exception | None = None
    for attempt, wait in enumerate((0, 5, 15)):
        if wait:
            time.sleep(wait)
        try:
            r = httpx.post(f"{s.nvidia_base_url}/chat/completions", json=body, timeout=s.nvidia_timeout_s,
                           headers={"Authorization": f"Bearer {s.nvidia_api_key}"})
        except httpx.HTTPError as exc:  # timeouts and connection errors: retry
            last = exc
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            last = OnlineError(f"HTTP {r.status_code}")
            continue
        if r.status_code != 200:
            raise OnlineError(f"online service answered HTTP {r.status_code}: {r.text[:160]}")
        content = (r.json()["choices"][0]["message"].get("content") or "").strip()
        lines = clean_lines(content)
        if not lines:
            raise OnlineError("online reader returned no text lines")
        return lines
    raise OnlineError(f"online reader unavailable: {type(last).__name__ if last else 'unknown'}")


def _dist(a: str, b: str) -> float:
    a, b = a.replace(" ", ""), b.replace(" ", "")
    return Levenshtein.distance(a, b) / max(len(a), len(b), 1)


def align(local: list[str], online: list[str], skip: float = 0.75) -> list[tuple[int, int]]:
    """Monotone alignment of online lines onto local lines: [(local_index, online_index), ...].

    Costs are edit distance between the local reading and the online line (garbage local text still
    carries order), plus a small positional prior.  Either side may skip lines (ornaments, lines the
    local segmenter missed or invented).
    """
    n, m = len(local), len(online)
    if not n or not m:
        return []
    inf = 1e9
    dp = np.full((n + 1, m + 1), inf)
    back: dict[tuple[int, int], tuple[int, int]] = {}
    dp[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i, j] >= inf:
                continue
            for di, dj, cost in ((1, 1, None), (1, 0, skip), (0, 1, skip)):
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                c = cost if cost is not None else min(1.0, _dist(local[i], online[j])) + 0.1 * abs(i / n - j / m)
                if dp[i, j] + c < dp[ni, nj]:
                    dp[ni, nj] = dp[i, j] + c
                    back[(ni, nj)] = (i, j)
    pairs, state = [], (n, m)
    while state != (0, 0):
        prev = back[state]
        if state[0] - prev[0] == 1 and state[1] - prev[1] == 1:
            pairs.append((prev[0], prev[1]))
        state = prev
    return pairs[::-1]
