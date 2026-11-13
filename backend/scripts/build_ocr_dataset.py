"""Build the line-image training/evaluation set for the CRNN recogniser.

Sources (both public on Hugging Face):
  * snskrt/Sanskrit_OCR_Parallel_Corpus (Apache-2.0): 6,265 two-line Sanskrit verse crops with the
    text in LABELS/labels.csv.  Each crop is split into its two text lines with our own
    segmenter; crops whose line count does not match the label are dropped (no guessing).
    Split BY SOURCE PAGE (the 4-digit prefix of the file name) so neighbouring verses of one page
    never straddle train and test.
  * iamkushagratomar/devanagari-ocr-synthetic-80k (MIT): synthetic Hindi lines in many fonts, used
    to teach the network the Devanagari writing system; never used for the headline number.

Usage (from backend/):
    python -m scripts.build_ocr_dataset --snskrt-parquet X.parquet --labels labels.csv \
        --synth-dir synth/ --out lines.pkl --synth-train 30000
"""

import argparse
import collections
import csv
import hashlib
import pickle
import unicodedata

import cv2
import numpy as np
import pyarrow.parquet as pq

from app.ocr.engines.crnn import line_to_uint8
from app.ocr.segment import segment_lines


def split_two_lines(img: np.ndarray):
    """Return the text-line crops of a verse image, ignoring neighbour fragments cut by the border."""
    _, ink = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h = img.shape[0]
    boxes = [b for b in segment_lines(ink) if b.y > 1 and b.y + b.h < h - 1]
    if not boxes:
        return []
    widest = max(b.w for b in boxes)
    boxes = sorted((b for b in boxes if b.w >= 0.5 * widest and b.h >= 8), key=lambda b: b.y)
    pad = 2
    return [img[max(0, b.y - pad) : b.y + b.h + pad, b.x : b.x + b.w] for b in boxes]


def crop_ink(img: np.ndarray, margin: int = 4) -> np.ndarray:
    _, ink = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(ink > 0)
    if len(xs) == 0:
        return img
    y0, y1 = max(0, ys.min() - margin), min(img.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(img.shape[1], xs.max() + margin + 1)
    return img[y0:y1, x0:x1]


def group_split(group: str) -> str:
    """Deterministic 85/7.5/7.5 split on the source page id."""
    v = int(hashlib.md5(group.encode()).hexdigest(), 16) % 1000
    return "train" if v < 850 else "val" if v < 925 else "test"


def nfc(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snskrt-parquet", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synth-train", type=int, default=30000)
    ap.add_argument("--synth-val", type=int, default=2000)
    ap.add_argument("--min-char-count", type=int, default=5)
    args = ap.parse_args()

    labels = {
        r["image_name"]: [nfc(l) for l in r["shlok"].split("\n") if l.strip()]
        for r in csv.DictReader(open(args.labels, encoding="utf-8"))
    }

    data = {"real": {"train": [], "val": [], "test": []}, "synth": {"train": [], "val": []}}
    stats = collections.Counter()
    for r in pq.read_table(args.snskrt_parquet).to_pylist():
        name = r["image"]["path"].split("/")[-1]
        want = labels.get(name)
        img = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
        if want is None or img is None:
            stats["no_label"] += 1
            continue
        crops = split_two_lines(img)
        if len(crops) != len(want):
            stats["line_count_mismatch"] += 1
            continue
        split = group_split(name.split("_")[0])
        for crop, text in zip(crops, want):
            data["real"][split].append((line_to_uint8(crop), text, name))
        stats["pages_kept"] += 1
    print("snskrt:", dict(stats), {k: len(v) for k, v in data["real"].items()})

    def synth(files, limit, key):
        n = 0
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=500, columns=["image", "text"]):
                for row in batch.to_pylist():
                    img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
                    text = nfc(row["text"] or "")
                    if img is None or not text:
                        continue
                    data["synth"][key].append((line_to_uint8(crop_ink(img)), text, "synth"))
                    n += 1
                    if n >= limit:
                        return

    d = args.synth_dir.rstrip("/")
    synth([f"{d}/train-00000-of-00002.parquet", f"{d}/train-00001-of-00002.parquet"], args.synth_train, "train")
    synth([f"{d}/validation-00000-of-00001.parquet"], args.synth_val, "val")
    print("synth:", {k: len(v) for k, v in data["synth"].items()})

    # Vocabulary from training text only; rare characters are dropped together with their lines.
    counts = collections.Counter(ch for src in data.values() for ex in src.get("train", []) for ch in ex[1])
    vocab = [""] + sorted(ch for ch, c in counts.items() if c >= args.min_char_count)
    allowed = set(vocab)
    dropped = 0
    for src in data.values():
        for split, items in src.items():
            keep = [ex for ex in items if set(ex[1]) <= allowed]
            dropped += len(items) - len(keep)
            src[split] = keep
    print(f"vocab={len(vocab) - 1} chars, dropped {dropped} lines with rare characters")
    print({s: {k: len(v) for k, v in src.items()} for s, src in data.items()})

    with open(args.out, "wb") as fh:
        pickle.dump({"vocab": vocab, "data": data}, fh, protocol=4)


if __name__ == "__main__":
    main()
