"""Engine orchestration: pick engines for a script, fall back on failure, trip a circuit breaker
when an engine keeps failing so a broken engine does not slow every job down.

Supported ``engine`` values
---------------------------
``"auto"``
    Tries every available engine in registry order (ppocr → tesseract → claude),
    returning the first success.  This is the default.

``"ensemble"``
    Runs all available local engines in parallel and fuses their outputs with
    weighted ROVER voting (see ``engines/ensemble.py`` for details).
    Falls back to auto if the ensemble itself fails.

``"ensemble+cloud"``
    Same as ensemble but also includes the Claude cloud engine.

``"<name>"``
    Forces a specific engine (ppocr / tesseract / claude) with fallback to the
    others on failure.
"""

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from app.ocr.engines import EngineUnavailable, LineResult, OCREngine, get_engine, list_engines

log = logging.getLogger(__name__)


class NoEngineAvailable(RuntimeError):
    pass


@dataclass
class _Breaker:
    failures: int = 0
    open_until: float = 0.0


@dataclass
class CircuitBreaker:
    threshold: int = 3
    cooldown_s: float = 120.0
    _state: dict[str, _Breaker] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def is_open(self, name: str) -> bool:
        with self._lock:
            b = self._state.get(name)
            return bool(b and b.open_until > time.monotonic())

    def success(self, name: str) -> None:
        with self._lock:
            self._state[name] = _Breaker()

    def failure(self, name: str) -> None:
        with self._lock:
            b = self._state.setdefault(name, _Breaker())
            b.failures += 1
            if b.failures >= self.threshold:
                b.open_until = time.monotonic() + self.cooldown_s
                b.failures = 0
                log.warning("circuit opened for engine %s", name)


    def snapshot(self) -> dict[str, dict]:
        """Per-engine breaker state for health checks: {"open": bool, "retry_in_s": float, "recent_failures": int}."""
        now = time.monotonic()
        with self._lock:
            return {
                n: {"open": b.open_until > now, "retry_in_s": round(max(0.0, b.open_until - now), 1), "recent_failures": b.failures}
                for n, b in self._state.items()
            }


breaker = CircuitBreaker()

_ENSEMBLE_MODES = {"ensemble", "ensemble+cloud"}


def _usable(results: list[LineResult]) -> bool:
    """At least one line in five carries text (an empty list means there was nothing to read)."""
    if not results:
        return True
    return sum(1 for r in results if r.text.strip()) * 5 >= len(results)


def candidates(script: str, requested: str = "auto") -> list[OCREngine]:
    # Ensemble modes are handled separately in recognize(); they don't use the
    # fallback list.
    if requested in _ENSEMBLE_MODES:
        eng = get_engine("ensemble")
        if eng is None or not eng.supports(script):
            raise NoEngineAvailable(f"Ensemble engine cannot read script '{script}' — no sub-engines available")
        return [eng]

    if requested and requested != "auto":
        eng = get_engine(requested)
        if eng is None:
            raise NoEngineAvailable(f"Unknown engine '{requested}'")
        if not eng.supports(script):
            raise NoEngineAvailable(f"Engine '{eng.label}' cannot read {script} on this server")
        # An explicit choice still falls back to the others if it fails at run time.
        rest = [e for e in list_engines() if e is not eng and e.supports(script)]
        return [eng, *rest]

    found = [e for e in list_engines() if e.supports(script)]
    if not found:
        raise NoEngineAvailable(
            f"No OCR engine on this server supports {script}. "
            "Install Tesseract with the matching language data or set ANTHROPIC_API_KEY."
        )
    return found


def recognize(line_images: list[np.ndarray], script: str, requested: str = "auto") -> tuple[str, list[LineResult]]:
    from app.config import get_settings

    if requested == "auto" and get_settings().auto_use_ensemble:
        ens = get_engine("ensemble")
        # One engine can be wrong on a given line; with two or more, the ensemble picks per line.
        if ens is not None and ens.engine_count(script) >= 2:  # type: ignore[attr-defined]
            requested = "ensemble"
    # --- Ensemble path ---
    if requested in _ENSEMBLE_MODES:
        include_cloud = requested == "ensemble+cloud"
        eng = get_engine("ensemble")
        if eng is None:
            raise NoEngineAvailable("Ensemble engine not registered")

        try:
            if breaker.is_open("ensemble"):
                raise EngineUnavailable("ensemble: circuit open")
            results = eng.recognize(line_images, script, include_cloud=include_cloud)  # type: ignore[call-arg]
            breaker.success("ensemble")
            return eng.name, results
        except EngineUnavailable as exc:
            log.warning("ensemble failed (%s), falling back to auto", exc)
            breaker.failure("ensemble")
        # Fall through to auto below
        requested = "auto"

    # --- Single-engine / auto path ---
    errors = []
    best_effort: tuple[str, list[LineResult]] | None = None
    for eng in candidates(script, requested):
        if breaker.is_open(eng.name):
            errors.append(f"{eng.name}: circuit open")
            continue
        try:
            results = eng.recognize(line_images, script)
            breaker.success(eng.name)
            if not _usable(results):
                # Ran without error but read (almost) nothing: let the next engine try.
                errors.append(f"{eng.name}: returned no text")
                best_effort = best_effort or (eng.name, results)
                continue
            return eng.name, results
        except EngineUnavailable as exc:
            errors.append(f"{eng.name}: {exc}")
        except Exception as exc:  # engine crashed: count it against the breaker and try the next one
            log.exception("engine %s failed", eng.name)
            breaker.failure(eng.name)
            errors.append(f"{eng.name}: {exc}")
    if best_effort is not None:
        return best_effort  # nothing readable anywhere: a blank page is a legitimate answer
    raise NoEngineAvailable("All engines failed - " + "; ".join(errors))
