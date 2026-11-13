"""CRNN (Convolutional Recurrent Neural Network) OCR engine.

Architecture: CNN feature extractor → Bidirectional LSTM → CTC decoder.
This is the established architecture for sequence recognition on text line images
(Shi et al., "An End-to-End Trainable Neural Network for Image-based Sequence
Recognition", TPAMI 2017).

The model is trained on the ``snskrt/Sanskrit_OCR_Parallel_Corpus`` dataset
(6,265 Sanskrit manuscript line image + Devanagari ground-truth pairs) via
``train_crnn.py``.  At inference it runs fully on CPU via PyTorch; no ONNX
Runtime or cloud dependency.

Weights are stored at ``backend/data/crnn_sanskrit.pt``.  If the file does not
exist, the engine reports itself as unavailable and the pipeline falls back to
``ppocr`` / ``tesseract`` transparently.

CTC decoding
------------
We use greedy (argmax) decoding with blank-token collapsing for speed.  This
is accurate enough for high-confidence predictions.  Beam-search decoding is
available but not enabled by default because it adds ~2× latency per batch.

Supported scripts
-----------------
Currently Devanagari only (the training dataset is exclusively Sanskrit in
Devanagari).  The checkpoint stores the character vocabulary so other scripts
can be supported by training separate checkpoints and swapping them in via
``CRNN_MODEL_PATH`` env var or the ``crnn_model_path`` setting.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings
from app.ocr.engines.base import EngineUnavailable, LineResult, OCREngine

log = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "crnn_sanskrit.pt"

# Input height for the CNN (width is variable — the LSTM handles sequence length)
_IMG_H = 48
# Lines are squeezed horizontally (Devanagari at H=48 is ~15 px/char): 3x less compute, still >2 CTC steps/char.
_WIDTH_SQUEEZE = 0.6

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True

    class _ConvBlock(nn.Sequential):
        def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int] | None = (2, 2)) -> None:
            layers: list[nn.Module] = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(pool, pool))
            super().__init__(*layers)

    class CRNN(nn.Module):
        """
        CNN encoder + 2-layer biLSTM + linear CTC head.

        Input:  (B, 1, H=48, W) - grayscale line image, ink dark on light; W is variable.
        Output: (T, B, n_classes) log-softmax over the vocabulary, with T = W // 4.

        Height is collapsed to 1 by pooling; width is only reduced 4x so Devanagari lines
        (many glyphs per pixel of width) still get enough CTC time-steps.
        """

        WIDTH_STRIDE = 4

        def __init__(self, n_classes: int) -> None:
            super().__init__()
            self.cnn = nn.Sequential(
                _ConvBlock(1, 24, pool=(2, 2)),     # H 48->24, W/2
                _ConvBlock(24, 48, pool=(2, 2)),    # H 24->12, W/4
                _ConvBlock(48, 96, pool=(2, 1)),    # H 12->6
                _ConvBlock(96, 96, pool=None),
                _ConvBlock(96, 192, pool=(2, 1)),  # H 6->3
                _ConvBlock(192, 192, pool=None),
                nn.AdaptiveAvgPool2d((1, None)),    # H 3->1
                nn.Dropout2d(0.1),
            )
            self.rnn = nn.LSTM(192, 192, num_layers=2, batch_first=False, bidirectional=True, dropout=0.25)
            self.fc = nn.Linear(192 * 2, n_classes)

        @staticmethod
        def out_len(width: int) -> int:
            return max(1, width // CRNN.WIDTH_STRIDE)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            feat = self.cnn(x)                    # (B, 192, 1, W/4)
            feat = feat.squeeze(2).permute(2, 0, 1)  # (T, B, 192)
            rnn_out, _ = self.rnn(feat)           # (T, B, 384)
            return self.fc(rnn_out).log_softmax(2)

except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# CTC greedy decoder
# ---------------------------------------------------------------------------


def ctc_greedy_decode(log_probs: "torch.Tensor", blank: int = 0) -> list[list[int]]:
    """Greedy CTC decode: argmax + blank removal + duplicate collapsing.

    Args:
        log_probs: (T, B, C) log-softmax output from the model.
        blank:     Index of the CTC blank token.

    Returns:
        List of B integer sequences (class indices, not including blank/dupes).
    """
    best = log_probs.argmax(2).T.cpu().numpy()  # (B, T)
    results = []
    for row in best:
        keep = np.ones(len(row), bool)
        keep[1:] = row[1:] != row[:-1]  # collapse repeats
        keep &= row != blank
        results.append(row[keep].tolist())
    return results


# ---------------------------------------------------------------------------
# CRNNEngine
# ---------------------------------------------------------------------------


class CRNNEngine(OCREngine):
    """CRNN line recogniser trained on Sanskrit verse lines and synthetic Devanagari lines.

    The engine is available only when:
      - PyTorch is installed,
      - ``data/crnn_sanskrit.pt`` exists (produced by ``train_crnn.py``), AND
      - the checkpoint's validation CER is at most ``settings.crnn_max_val_cer``, so a
        half-trained model is never used.

    It supports Devanagari by default.  The checkpoint embeds the vocabulary
    and supported script list so multi-script checkpoints are possible without
    code changes.
    """

    name = "crnn"
    label = "CRNN (CNN+biLSTM) — Sanskrit-trained"
    kind = "local"
    architecture = "6-layer CNN encoder + 2-layer biLSTM + CTC decoder (H=48), trained on Sanskrit/Devanagari line images"

    def __init__(self, model_path: Path | None = None) -> None:
        self._path = model_path or _DEFAULT_MODEL_PATH
        self._model: "CRNN | None" = None
        self._vocab: list[str] = []
        self._scripts: set[str] = set()
        self._lock = threading.Lock()
        self._load_tried = False
        self.val_cer: float | None = None

    def describe(self) -> dict:
        info = super().describe()
        info["val_cer"] = self.val_cer
        return info

    # ------------------------------------------------------------------
    # OCREngine interface
    # ------------------------------------------------------------------

    def available(self) -> bool:
        if not _TORCH_AVAILABLE:
            return False
        if not self._path.exists():
            return False
        if self._load_tried and self._model is None:
            return False
        try:
            self._ensure_loaded()
        except Exception:
            return False
        return self.val_cer is None or self.val_cer <= get_settings().crnn_max_val_cer

    def supported_scripts(self) -> set[str]:
        if not self.available():
            return set()
        return self._scripts

    def recognize(self, line_images: list[np.ndarray], script: str) -> list[LineResult]:
        if not _TORCH_AVAILABLE:
            raise EngineUnavailable("PyTorch not installed")
        if not self._path.exists():
            raise EngineUnavailable(f"CRNN model not found at {self._path}. Run train_crnn.py first.")
        self._ensure_loaded()
        assert self._model is not None

        import torch

        results: list[LineResult] = []
        self._model.eval()
        with torch.no_grad():
            for img in line_images:
                if img is None or img.size == 0:
                    results.append(LineResult("", 0.0))
                    continue
                tensor = _img_to_tensor(img)                  # (1, 1, 48, W)
                log_probs = self._model(tensor)                # (T, 1, C)
                decoded = ctc_greedy_decode(log_probs, blank=0)
                indices = decoded[0]
                # Confidence: mean max-probability over the frames that emitted a character.
                # Blank frames are excluded - they are almost always near 1.0 and would inflate it.
                probs = log_probs[:, 0, :].exp()              # (T, C)
                max_probs, argmax = probs.max(1)              # (T,), (T,)
                emitted = max_probs[argmax != 0]
                conf = float(emitted.mean().item()) if len(emitted) > 0 else 0.0
                text = "".join(self._vocab[i] for i in indices if i < len(self._vocab))
                results.append(LineResult(text.strip(), conf if text.strip() else 0.0))
        return results

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            self._load_tried = True
            import torch

            ckpt = torch.load(self._path, map_location=torch.device("cpu"), weights_only=True)
            vocab = ckpt["vocab"]          # list of characters (index 0 = CTC blank)
            scripts = set(ckpt.get("scripts", ["devanagari"]))
            model = CRNN(n_classes=len(vocab))
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self._vocab = vocab
            self.val_cer = ckpt.get("val_cer")
            self._scripts = scripts
            self._model = model
            log.info("CRNN model loaded: vocab=%d scripts=%s", len(vocab), sorted(scripts))

    def reload(self) -> None:
        with self._lock:
            self._model = None
            self.val_cer = None
            self._load_tried = False


# ---------------------------------------------------------------------------
# Image preprocessing for the CRNN
# ---------------------------------------------------------------------------


def line_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalise any line crop to an (_IMG_H, W) uint8 grayscale array, W a multiple of 4.

    Shared by inference and training so both see identical inputs.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    new_w = max(8, round(w * _IMG_H / max(h, 1) * _WIDTH_SQUEEZE))
    new_w += (-new_w) % 4
    new_w = min(new_w, 4096)
    return cv2.resize(gray, (new_w, _IMG_H), interpolation=cv2.INTER_AREA if h >= _IMG_H else cv2.INTER_CUBIC)


def _img_to_tensor(img: np.ndarray) -> "torch.Tensor":
    """Convert a BGR or grayscale line image to a (1, 1, 48, W) float tensor in [0, 1]."""
    import torch

    if img.size == 0:
        return torch.ones(1, 1, _IMG_H, 8, dtype=torch.float32)
    t = torch.from_numpy(line_to_uint8(img).astype(np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0)
