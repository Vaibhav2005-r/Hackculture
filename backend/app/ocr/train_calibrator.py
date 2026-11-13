"""Train (or retrain) the CNN confidence calibrator from the correction history.

Usage
-----
    # From the backend/ directory:
    python -m app.ocr.train_calibrator

    # With options:
    python -m app.ocr.train_calibrator --epochs 50 --min-examples 50

The script:
  1. Loads every approved correction from the database.
  2. For each corrected line, retrieves its image from storage.
  3. Checks whether the OCR engine output matched the human correction (is_correct).
  4. Trains / fine-tunes the CNN calibrator.
  5. Invalidates the in-process calibrator cache so the ensemble picks it up.

If torch is not installed the script exits with a clear error message.
"""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "").strip()


def collect_examples() -> tuple[list[dict], list[str]]:
    """Return (examples, engine_names) from the correction table.

    An example has keys: img (np.ndarray), engine_name (str), is_correct (bool).
    We only use approved corrections so the signal is as clean as possible.
    """
    import cv2

    from app.db import SessionLocal
    from app.models import Correction, Line, Page
    from app.services.storage import load_image

    with SessionLocal() as db:
        rows = (
            db.query(Correction, Line, Page)
            .join(Line, Correction.line_id == Line.id)
            .join(Page, Line.page_id == Page.id)
            .filter(Correction.status == "approved")
            .all()
        )

    examples: list[dict] = []
    engine_names_seen: set[str] = set()

    for correction, line, page in rows:
        # We need the page image to crop the line region
        try:
            if not page.normalized_key:
                continue
            page_img = load_image(page.normalized_key)
        except Exception as exc:
            log.debug("skipping page %s: %s", page.id, exc)
            continue

        # Crop the line region (use stored bounding box)
        h, w = page_img.shape[:2]
        x0 = max(0, line.x)
        y0 = max(0, line.y)
        x1 = min(w, line.x + line.w)
        y1 = min(h, line.y + line.h)
        if x1 <= x0 or y1 <= y0:
            continue
        line_img = page_img[y0:y1, x0:x1]
        if line_img.size == 0:
            continue

        # Is the original OCR text equal to the human correction?
        is_correct = _nfc(line.ocr_text) == _nfc(correction.text)

        engine_name = line.engine or "unknown"
        engine_names_seen.add(engine_name)

        examples.append(
            {
                "img": line_img,
                "engine_name": engine_name,
                "is_correct": is_correct,
            }
        )

    engine_names = sorted(engine_names_seen)
    log.info("collected %d examples from %d engine(s): %s", len(examples), len(engine_names), engine_names)
    return examples, engine_names


def main() -> int:
    parser = argparse.ArgumentParser(description="Train CNN confidence calibrator from correction history")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default 30)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 1e-3)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-examples", type=int, default=100, help="Skip training if fewer examples")
    parser.add_argument("--force", action="store_true", help="Train even with fewer than min-examples")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        log.error("PyTorch not installed. Install with:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        return 1

    from app.ocr.calibrator import train

    examples, engine_names = collect_examples()
    if not engine_names:
        log.warning("No approved corrections found. Run OCR and have reviewers approve corrections first.")
        return 0

    min_ex = 1 if args.force else args.min_examples
    result = train(
        examples=examples,
        engine_names=engine_names,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        min_examples=min_ex,
    )

    if result["status"] == "skipped":
        log.warning("Training skipped: %s", result["reason"])
        return 0

    log.info(
        "Done — n=%d  val_loss=%.4f  val_acc=%.4f  val_auc=%.4f",
        result["n_examples"],
        result["loss"],
        result["accuracy"],
        result["auc"],
    )

    # Invalidate in-process calibrator cache (a no-op when run as a separate process)
    from app.ocr.engines.registry import reset_ensemble_calibrator

    reset_ensemble_calibrator()

    return 0


if __name__ == "__main__":
    sys.exit(main())
