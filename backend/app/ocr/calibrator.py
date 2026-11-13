"""CNN Confidence Calibrator — trains on the correction history to learn which
engine to trust more for a given script / image difficulty.

Architecture
------------
A lightweight convolutional network (MobileNet-style depthwise separable
convolutions) maps a 32×128 grayscale line image → a scalar "quality score"
in [0, 1] per engine.  During training, examples come from the correction
table: when a human corrected a line we know the original OCR engine output
was wrong (or less accurate) and the corrected text is ground truth.

Training signal
---------------
  positive  — line images where the engine's OCR output exactly matches the
               human correction (ocr_text == correction.text after NFC normalisation)
  negative  — line images where the engine's output differed from the correction

The network is trained as a binary classifier per engine and kept on disk as a
single-file checkpoint ``calibrator.pt`` inside ``data/``.  If torch is not
installed the whole module raises ImportError and the ensemble silently skips
calibration.

Retraining
----------
Call ``python -m app.ocr.train_calibrator`` or use the
``/api/ops/train-calibrator`` endpoint (admin only) to trigger a training run.
The calibrator file is overwritten atomically and the ensemble picks it up
on the next request.

Limitations (deliberately honest)
----------------------------------
*  The model is small and runs on CPU.  It will not outperform the engines
   themselves on raw character recognition — it only re-weights their outputs.
*  With fewer than ~200 labelled corrections the calibrator is unlikely to be
   useful; the ensemble falls back to equal weights automatically in that case.
*  The model is script-agnostic by default; pass ``script`` to get a
   script-specific sub-model (not yet implemented — single global model is used
   for now to keep data requirements low).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Try importing torch; propagate ImportError so callers can decide what to do.
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "calibrator.pt"

# Input shape expected by the CNN
_IMG_H, _IMG_W = 32, 128


# ---------------------------------------------------------------------------
# Model definition: tiny depthwise-separable CNN
# ---------------------------------------------------------------------------


def _dw_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Depthwise-separable block (MobileNet style)."""
    return nn.Sequential(
        # Depthwise
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU6(inplace=True),
        # Pointwise
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


class _CalibCNN(nn.Module):
    """
    Input: (B, 1, 32, 128) grayscale line image, normalised to [0, 1].
    Output: (B, n_engines) logit for each engine.
    """

    def __init__(self, n_engines: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),  # → 16×16×64
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            _dw_block(16, 32, stride=2),  # → 32×8×32
            _dw_block(32, 64, stride=2),  # → 64×4×16
            _dw_block(64, 96, stride=2),  # → 96×2×8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(96, n_engines)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, H, W)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)          # (B, 96, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 96)
        x = self.dropout(x)
        return self.head(x)        # (B, n_engines)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class ConfidenceCalibrator:
    """Load a fitted calibrator from disk.  Provides engine weight vectors."""

    def __init__(self, model_path: Path = _MODEL_PATH) -> None:
        self._path = model_path
        self._model: _CalibCNN | None = None
        self._engine_names: list[str] = []
        self._device = torch.device("cpu")

    def is_fitted(self) -> bool:
        """True when a checkpoint exists and can be loaded."""
        if not self._path.exists():
            return False
        try:
            self._ensure_loaded()
            return True
        except Exception:
            return False

    def line_probs(self, engines: list, line_images: list[np.ndarray]) -> np.ndarray:
        """P(engine reads this line correctly), shape (n_lines, n_engines), in ``engines`` order.

        Engines the calibrator was not trained on get a neutral 0.5.
        """
        self._ensure_loaded()
        assert self._model is not None
        out = np.full((len(line_images), len(engines)), 0.5, dtype=np.float32)
        if not line_images:
            return out
        batch = torch.stack([_preprocess_image(img) for img in line_images]).to(self._device)
        self._model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self._model(batch)).cpu().numpy()  # (N, n_engines_trained)
        for col, eng in enumerate(engines):
            if eng.name in self._engine_names:
                idx = self._engine_names.index(eng.name)
                if idx < probs.shape[1]:
                    out[:, col] = probs[:, idx]
        return out

    def engine_weights(self, engines: list, line_images: list[np.ndarray], script: str) -> list[float]:
        """Page-level weight per engine: the mean of ``line_probs`` (kept for callers that need one number)."""
        return [float(w) for w in self.line_probs(engines, line_images[:8]).mean(0)] if line_images else [1.0] * len(engines)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        checkpoint = torch.load(self._path, map_location=self._device, weights_only=True)
        n_engines = checkpoint["n_engines"]
        self._engine_names = checkpoint["engine_names"]
        model = _CalibCNN(n_engines=n_engines)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self._model = model

    def reload(self) -> None:
        """Force a reload from disk (call after retraining)."""
        self._model = None
        self._engine_names = []
        self._ensure_loaded()


# ---------------------------------------------------------------------------
# Training helpers (used by train_calibrator.py)
# ---------------------------------------------------------------------------


def _preprocess_image(img: np.ndarray) -> torch.Tensor:
    """Resize a BGR or grayscale line image to (1, H, W) float32 tensor in [0,1]."""
    import cv2

    if img is None or img.size == 0:
        return torch.zeros(1, _IMG_H, _IMG_W, dtype=torch.float32)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    resized = cv2.resize(gray, (_IMG_W, _IMG_H), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W)
    return t


def build_dataset(
    engine_names: list[str],
    examples: list[dict],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build (X, Y, M) tensors from labelled examples.

    Each example is ``{"img": ndarray, "labels": {engine_name: bool}}`` (one entry per engine that was
    run on the line); the legacy ``{"img", "engine_name", "is_correct"}`` form is accepted too.

    Returns:
      X : (N, 1, H, W) float32 line images
      Y : (N, n_engines) float32, 1 = that engine read the line correctly
      M : (N, n_engines) float32 mask, 1 where Y is known.  Engines that were not run on a line are
          masked out of the loss instead of being counted as wrong.
    """
    n_eng = len(engine_names)
    name_to_idx = {name: i for i, name in enumerate(engine_names)}

    X_list, Y_list, M_list = [], [], []
    for ex in examples:
        labels = ex.get("labels")
        if labels is None:
            labels = {ex["engine_name"]: ex["is_correct"]}
        y = torch.zeros(n_eng, dtype=torch.float32)
        m = torch.zeros(n_eng, dtype=torch.float32)
        for name, ok in labels.items():
            idx = name_to_idx.get(name)
            if idx is not None:
                y[idx] = 1.0 if ok else 0.0
                m[idx] = 1.0
        X_list.append(_preprocess_image(ex["img"]))
        Y_list.append(y)
        M_list.append(m)
    return torch.stack(X_list), torch.stack(Y_list), torch.stack(M_list)


def train(
    examples: list[dict],
    engine_names: list[str],
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    model_path: Path = _MODEL_PATH,
    min_examples: int = 100,
) -> dict:
    """
    Train or fine-tune the calibrator.

    Returns a metrics dict with ``loss``, ``accuracy``, ``n_examples``.
    Saves the checkpoint atomically (write to .tmp, then rename).
    """
    if len(examples) < min_examples:
        return {
            "status": "skipped",
            "reason": f"only {len(examples)} examples (need ≥ {min_examples})",
            "n_examples": len(examples),
        }

    log.info("calibrator: training on %d examples, engines=%s", len(examples), engine_names)
    X, Y, M = build_dataset(engine_names, examples)
    n_engines = len(engine_names)

    # Hold out 15% so the reported accuracy/AUC is not measured on training data.
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(X), generator=g)
    n_val = max(1, int(0.15 * len(X)))
    va, tr = perm[:n_val], perm[n_val:]

    model = _CalibCNN(n_engines=n_engines)
    # If we have an existing checkpoint for the same engines, fine-tune from it.
    if model_path.exists():
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            if ckpt.get("engine_names") == engine_names:
                model.load_state_dict(ckpt["model_state"])
                log.info("calibrator: fine-tuning from existing checkpoint")
        except Exception as exc:
            log.warning("calibrator: could not load existing checkpoint for fine-tuning: %s", exc)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def masked_loss(logits, y, m):
        return (bce(logits, y) * m).sum() / m.sum().clamp(min=1)

    best_state, best_val, final_loss = None, float("inf"), 0.0
    for epoch in range(epochs):
        model.train()
        order = tr[torch.randperm(len(tr), generator=g)]
        epoch_loss = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            if len(idx) < 2:
                continue  # BatchNorm needs more than one sample
            optimizer.zero_grad()
            loss = masked_loss(model(X[idx]), Y[idx], M[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stability on small datasets
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        scheduler.step()
        final_loss = epoch_loss / max(len(order), 1)
        model.eval()
        with torch.no_grad():
            val_loss = masked_loss(model(X[va]), Y[va], M[va]).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            log.info("calibrator epoch %d/%d  train=%.4f  val=%.4f", epoch + 1, epochs, final_loss, val_loss)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X[va]))
    yv, mv = Y[va], M[va]
    acc = float((((probs > 0.5).float() == yv).float() * mv).sum() / mv.sum().clamp(min=1))
    # AUC over the held-out (line, engine) pairs: P(score of a correct read > score of a wrong one)
    pos, neg = probs[(yv == 1) & (mv == 1)], probs[(yv == 0) & (mv == 1)]
    auc = float((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean()) if len(pos) and len(neg) else float("nan")

    # Atomic save
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = model_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "n_engines": n_engines,
            "engine_names": engine_names,
            "train_loss": float(final_loss),
            "train_acc": acc,
            "val_auc": auc,
            "n_examples": len(examples),
        },
        tmp,
    )
    os.replace(tmp, model_path)
    log.info("calibrator saved to %s  val_loss=%.4f  val_acc=%.4f  val_auc=%.4f", model_path, best_val, acc, auc)

    return {
        "status": "trained",
        "n_examples": len(examples),
        "loss": round(float(best_val), 4),
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "epochs": epochs,
        "engine_names": engine_names,
    }
