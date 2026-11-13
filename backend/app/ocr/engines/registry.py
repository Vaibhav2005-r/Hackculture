from app.ocr.engines.base import OCREngine
from app.ocr.engines.claude import ClaudeEngine
from app.ocr.engines.crnn import CRNNEngine
from app.ocr.engines.ensemble import EnsembleEngine
from app.ocr.engines.rapid import RapidEngine
from app.ocr.engines.tesseract import TesseractEngine

# Order = preference for "auto":
#   1. CRNN (Sanskrit-trained, highest accuracy when model is present)
#   2. PP-OCRv5 SVTR (fast, broad script coverage)
#   3. Tesseract 5 LSTM (Gujarati + broad fallback)
#   4. Claude vision (cloud, all scripts)
#
# EnsembleEngine is registered but NOT in the auto-fallback list (opt-in via engine="ensemble").
_ENGINES: dict[str, OCREngine] = {
    e.name: e
    for e in (CRNNEngine(), RapidEngine(), TesseractEngine(), ClaudeEngine(), EnsembleEngine())
}


def reset_ensemble_calibrator() -> None:
    """Drop the ensemble's cached calibrator so the next call reloads it (after retraining)."""
    ens = _ENGINES["ensemble"]
    with ens._calibrator_lock:  # type: ignore[attr-defined]
        ens._calibrator = None  # type: ignore[attr-defined]
        ens._calibrator_tried = False  # type: ignore[attr-defined]


def all_engines() -> list[OCREngine]:
    """Every registered engine, including the opt-in ensemble (for the UI)."""
    return list(_ENGINES.values())


def get_engine(name: str) -> OCREngine | None:
    return _ENGINES.get(name)


def list_engines() -> list[OCREngine]:
    # Exclude the ensemble from the default auto-fallback list; it is used explicitly.
    return [e for e in _ENGINES.values() if e.name != "ensemble"]
