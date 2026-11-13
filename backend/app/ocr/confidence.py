"""Honest per-line confidence.

A recogniser's own softmax confidence is high on lines it gets wrong (a bold scribal hand it has never
seen still scores 70-80%).  What does track errors is *instability*: re-reading the same crop with
thinner strokes or at a smaller scale gives the same text when the line is truly read and different
text when the recogniser is guessing.  Each line is scored by a small logistic model over
[engine confidence, agreement with the thin re-read, agreement with the small re-read, line length],
fitted by ``scripts/fit_confidence.py`` on held-out lines with known text, giving P(line is read with
<= 30% character error).  Without the fitted file the raw confidence is kept.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from app.config import BACKEND_DIR

log = logging.getLogger(__name__)

PATH = BACKEND_DIR / "data" / "confidence.json"
_model: dict | None = None
_tried = False


def agreement(a: str, b: str) -> float:
    """1 - normalised edit distance between two readings of the same line (0..1)."""
    if not a and not b:
        return 1.0
    return max(0.0, 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b), 1))


def variants(crops: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Two re-renderings of each crop: thinner strokes, and a smaller scale (bold hands read differently)."""
    import cv2

    thin, small = [], []
    for c in crops:
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
        thin.append(cv2.cvtColor(cv2.dilate(g, np.ones((3, 3), np.uint8)), cv2.COLOR_GRAY2BGR))  # max filter: dark ink thins
        small.append(cv2.resize(c, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA))
    return thin, small


def features(conf: float, a_thin: float, a_small: float, text: str) -> np.ndarray:
    return np.array([1.0, conf, a_thin, a_small, math.log1p(len(text))], dtype=np.float64)


def _load() -> dict | None:
    global _model, _tried
    if not _tried:
        _tried = True
        try:
            _model = json.loads(PATH.read_text()) if PATH.exists() else None
        except Exception as exc:  # corrupt file: fall back to raw confidences
            log.warning("could not load %s: %s", PATH, exc)
    return _model


def reload() -> None:
    global _tried
    _tried = False


def predict(conf: float, a_thin: float, a_small: float, text: str) -> float:
    m = _load()
    if m is None:
        return conf
    z = float(np.dot(np.array(m["w"]), features(conf, a_thin, a_small, text)))
    return 1.0 / (1.0 + math.exp(-z))


def rescore(crops: list[np.ndarray], texts: list[str], confs: list[float], script: str, engine=None) -> list[float]:
    """Calibrated confidence per line.

    The signal is *stability*: PP-OCR is run again on a thinner and on a smaller rendering of the same
    crop.  A line it truly reads comes out the same every time; a line it is guessing changes.  This
    needs no second engine, so it holds on manuscripts where the CRNN (trained on print) is the weaker
    reader and disagreement with it would only punish correct PP-OCR output.
    """
    from app.ocr.engines import get_engine

    eng = engine or get_engine("ppocr")
    if _load() is None or eng is None or not eng.supports(script) or not crops:
        return confs
    try:
        thin, small = variants(crops)
        r_thin, r_small = eng.recognize(thin, script), eng.recognize(small, script)
    except Exception as exc:
        log.warning("confidence re-scoring skipped: %s", exc)
        return confs
    return [
        round(predict(conf, agreement(text, x.text), agreement(text, y.text), text), 4)
        for text, conf, x, y in zip(texts, confs, r_thin, r_small)
    ]
