"""Ensemble engine: runs all available local engines in parallel, then votes.

Strategy
--------
1.  Collect outputs from every available engine that supports the requested script.
    API/cloud engines (Claude) are only included when *explicitly* requested via
    ``engine="ensemble+cloud"`` to avoid unnecessary costs.
2.  For each line position we have N candidate strings with confidence scores.
    We use a weighted ROVER (Recogniser Output Voting Error Reduction) variant:

    a.  Align candidates pairwise with Levenshtein alignment.
    b.  For each character position, pick the character from the candidate whose
        engine has the highest calibrated confidence *and* whose character was
        voted by the majority of engines (or the single-best if no majority).
    c.  Break ties by engine priority order.

3.  The optional CNN confidence calibrator (``calibrator.py``) re-weights each
    engine's raw confidence before voting.  It is trained on the correction log
    (human corrections = ground truth) and can be retrained at any time without
    touching the rest of the pipeline.

4.  The ensemble confidence for a line is the confidence of the winning engine
    *after* calibration, boosted slightly when the winning output agrees with
    at least one other engine.

If fewer than 2 engines are available the ensemble falls back to the single
engine transparently — no special-casing needed in the caller.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from app.ocr.engines.base import EngineUnavailable, LineResult, OCREngine

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pairwise character alignment (Smith-Waterman inspired, global on short strings)
# ---------------------------------------------------------------------------


_GAP = "\0"


def _project_onto(pivot: str, other: str) -> list[str]:
    """Align ``other`` to ``pivot`` and return one entry per pivot character.

    Each entry is the character of ``other`` aligned to that pivot position, or ``_GAP`` when
    ``other`` has a deletion there.  Characters ``other`` inserts relative to the pivot have no
    column and are dropped, so the result is always exactly ``len(pivot)`` long.
    """
    la, lb = len(pivot), len(other)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (0 if pivot[i - 1] == other[j - 1] else 1),
            )
    out = [_GAP] * la
    i, j = la, lb
    while i > 0 and j > 0:
        if dp[i][j] == dp[i - 1][j - 1] + (0 if pivot[i - 1] == other[j - 1] else 1):
            out[i - 1] = other[j - 1]
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1  # deletion in other: stays _GAP
        else:
            j -= 1  # insertion in other: dropped
    return out


# ---------------------------------------------------------------------------
# ROVER-style voting on a list of (text, confidence) pairs
# ---------------------------------------------------------------------------


def rover_vote(candidates: list[tuple[str, float]], weights: list[float] | None = None) -> tuple[str, float]:
    """Weighted ROVER vote over N (text, confidence) pairs.  Returns (best_text, fused_confidence).

    Every non-empty candidate is aligned to the strongest one (the pivot); each pivot position then
    takes the character (or deletion) with the most total weight.  Fused confidence is the pivot's
    confidence with a small bonus per agreeing engine.
    """
    if not candidates:
        return "", 0.0
    if weights is None:
        weights = [1.0] * len(candidates)

    live = [(c, w) for c, w in zip(candidates, weights) if c[0]]
    if not live:
        return candidates[0]
    if len(live) == 1:
        return live[0][0]

    total_w = sum(w for _, w in live) or 1.0
    live.sort(key=lambda cw: -cw[1])
    cands = [c for c, _ in live]
    ws = [w / total_w for _, w in live]

    pivot_text = cands[0][0]
    rows = [(list(pivot_text), ws[0])] + [(_project_onto(pivot_text, c[0]), w) for c, w in zip(cands[1:], ws[1:])]

    result_chars: list[str] = []
    for col in range(len(pivot_text)):
        char_weights: dict[str, float] = {}
        for chars, w in rows:
            char_weights[chars[col]] = char_weights.get(chars[col], 0.0) + w
        # max() keeps the first maximum, i.e. the pivot's character wins ties
        best = max(char_weights, key=lambda ch: (char_weights[ch], ch == pivot_text[col]))
        if best != _GAP:
            result_chars.append(best)

    agreements = sum(1 for c, _ in zip(cands[1:], ws[1:]) if _similarity(pivot_text, c[0]) >= 0.5)
    fused_conf = min(1.0, cands[0][1] * (1.0 + 0.08 * agreements))
    return "".join(result_chars), fused_conf


def _similarity(a: str, b: str) -> float:
    """Fraction of ``a`` that is matched in ``b`` by the alignment."""
    if not a:
        return 0.0
    return sum(1 for ch in _project_onto(a, b) if ch != _GAP) / len(a)


# ---------------------------------------------------------------------------
# EnsembleEngine
# ---------------------------------------------------------------------------


class EnsembleEngine(OCREngine):
    """Runs all available engines in parallel, then ROVER-votes per line.

    This engine is *not* registered in the automatic "auto" fallback list —
    it is registered as ``"ensemble"`` and must be requested explicitly
    (or used by the pipeline when ``engine="ensemble"``).

    Parameters
    ----------
    include_cloud : bool
        Whether to include cloud engines (Claude).  Default False to keep
        the cost and latency predictable.
    max_workers : int
        Thread-pool size for parallel engine calls.
    """

    name = "ensemble"
    label = "Ensemble (ROVER vote over all local engines)"
    kind = "local"
    architecture = "ROVER weighted voting across PP-OCRv5 + Tesseract 5 LSTM (+ optional CNN calibrator)"

    def __init__(self, include_cloud: bool = False, max_workers: int = 4) -> None:
        self._include_cloud = include_cloud
        self._max_workers = max_workers
        self._calibrator_lock = threading.Lock()
        self._calibrator: object | None = None  # loaded lazily
        self._calibrator_tried = False

    # ------------------------------------------------------------------
    # OCREngine interface
    # ------------------------------------------------------------------

    def available(self) -> bool:
        return len(self._active_engines()) >= 1

    def engine_count(self, script: str) -> int:
        """How many member engines can read ``script`` right now."""
        return sum(1 for e in self._active_engines() if e.supports(script))

    def supported_scripts(self) -> set[str]:
        scripts: set[str] = set()
        for eng in self._active_engines():
            scripts |= eng.supported_scripts()
        return scripts

    def recognize(
        self, line_images: list[np.ndarray], script: str, include_cloud: bool | None = None
    ) -> list[LineResult]:
        """``include_cloud`` overrides the engine default for this call only (thread-safe)."""
        engines = [e for e in self._active_engines(include_cloud) if e.supports(script)]
        if not engines:
            raise EngineUnavailable(f"No engine in ensemble supports script '{script}'")

        # Run the engines one after another.  They are fast (milliseconds per line) and mixing
        # ONNX Runtime and PyTorch inference across threads of one process corrupted the heap on
        # macOS (SIGTRAP in malloc), so we never run native inference libraries concurrently here.
        all_results: list[list[LineResult]] = []
        all_names: list[str] = [e.name for e in engines]
        for idx, e in enumerate(engines):
            try:
                all_results.append(e.recognize(line_images, script))
            except Exception as exc:
                log.warning("ensemble: engine %s failed: %s", all_names[idx], exc)
                all_results.append([LineResult("", 0.0)] * len(line_images))

        # Drop engines that completely failed (all empty / zero confidence)
        valid = [(eng, res) for eng, res in zip(engines, all_results) if res and any(r.text for r in res)]
        if not valid:
            raise EngineUnavailable("All engines in ensemble failed")
        if len(valid) == 1:
            return valid[0][1]

        # Per-line, per-engine weights: the calibrator's P(engine is right on this image) when a
        # fitted calibrator exists, otherwise the engine's own raw confidence.
        probs = self._calibrated_probs([e for e, _ in valid], line_images)

        fused: list[LineResult] = []
        for line_idx in range(len(line_images)):
            candidates = [(res[line_idx].text, res[line_idx].confidence) for _, res in valid]
            if probs is not None:
                weights = [float(probs[line_idx, i]) * (0.5 + 0.5 * candidates[i][1]) for i in range(len(candidates))]
            else:
                weights = [c[1] for c in candidates]
            text, conf = rover_vote(candidates, weights if any(w > 0 for w in weights) else None)
            fused.append(LineResult(text, conf))

        return fused

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _active_engines(self, include_cloud: bool | None = None) -> list[OCREngine]:
        from app.ocr.engines.registry import list_engines

        if include_cloud is None:
            include_cloud = self._include_cloud
        result = []
        for eng in list_engines():
            if eng.name == "ensemble":
                continue  # avoid recursion
            if not include_cloud and eng.kind == "cloud":
                continue
            if eng.available():
                result.append(eng)
        return result

    def _calibrated_probs(self, engines: list[OCREngine], line_images: list[np.ndarray]):
        """(n_lines, n_engines) correctness probabilities, or None when no calibrator is fitted."""
        cal = self._get_calibrator()
        if cal is None:
            return None
        try:
            return cal.line_probs(engines, line_images)
        except Exception as exc:
            log.debug("calibrator failed, using raw confidences: %s", exc)
            return None

    def _get_calibrator(self):
        with self._calibrator_lock:
            if self._calibrator is not None:
                return self._calibrator
            if self._calibrator_tried:
                return None
            self._calibrator_tried = True
            try:
                from app.ocr.calibrator import ConfidenceCalibrator

                cal = ConfidenceCalibrator()
                if cal.is_fitted():
                    self._calibrator = cal
                    log.info("ensemble: CNN confidence calibrator loaded")
                else:
                    log.debug("ensemble: calibrator not yet fitted; using equal weights")
            except ImportError:
                log.debug("ensemble: torch not available, calibrator disabled")
            except Exception as exc:
                log.warning("ensemble: could not load calibrator: %s", exc)
            return self._calibrator
