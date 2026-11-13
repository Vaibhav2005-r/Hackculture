"""Compose handwritten-style text LINES from the IIIT-INDIC-HW-WORDS-Hindi word images (CC-BY-4.0).

Real manuscripts are lines of connected pen-written letters, often with no spaces between words; the
printed training data of the first CRNN never showed it a pen stroke.  Each composed line joins 2-7
real handwritten word images (random gaps; when the gap is closed the label has no space).  Splits
follow the dataset's own train / validation / test word splits, so no writer-word pair leaks.

    python -m scripts.build_hw_lines --hw-dir hw/ --vocab-from lines.pkl --out hw_lines.pkl
"""

import argparse
import pickle
import random

import cv2
import numpy as np
import pyarrow.parquet as pq

from app.ocr.engines.crnn import line_to_uint8


def ink_columns(g):
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cols = np.where(b.sum(axis=0) > 0)[0]
    return (cols.min(), cols.max() + 1) if len(cols) else (0, g.shape[1])


def load_words(files, vocab, limit):
    out = []
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=400, columns=["image", "text"]):
            for r in batch.to_pylist():
                text = " ".join((r["text"] or "").split())
                if not text or not set(text) <= vocab:
                    continue
                g = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                a, b = ink_columns(g)
                out.append((g[:, a:b], text))
                if len(out) >= limit:
                    return out
    return out


def compose(words, n_lines, rng):
    lines = []
    for _ in range(n_lines):
        k = rng.randint(2, 7)
        pick = [rng.choice(words) for _ in range(k)]
        h = max(g.shape[0] for g, _ in pick)
        parts, text, spaced = [], "", rng.random() < 0.5
        for i, (g, t) in enumerate(pick):
            pad = np.full((h, g.shape[1]), 255, np.uint8)
            top = (h - g.shape[0]) // 2
            pad[top : top + g.shape[0]] = g
            parts.append(pad)
            if i < k - 1:
                gap = rng.randint(30, 70) if spaced else rng.randint(0, 6)
                parts.append(np.full((h, gap), 255, np.uint8))
            text += t + (" " if spaced and i < k - 1 else "")
        lines.append((line_to_uint8(np.hstack(parts)), text, "hand"))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw-dir", required=True)
    ap.add_argument("--vocab-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-lines", type=int, default=20000)
    ap.add_argument("--val-lines", type=int, default=1500)
    ap.add_argument("--test-lines", type=int, default=1500)
    args = ap.parse_args()
    vocab = set(pickle.load(open(args.vocab_from, "rb"))["vocab"]) - {""}
    d = args.hw_dir.rstrip("/")
    rng = random.Random(3)
    train_words = load_words([f"{d}/train-0000{i}-of-00003.parquet" for i in range(3)], vocab, 30000)
    val_words = load_words([f"{d}/validation-00000-of-00001.parquet"], vocab, 4000)
    test_words = load_words([f"{d}/test-00000-of-00001.parquet"], vocab, 4000)
    print("usable words:", len(train_words), len(val_words), len(test_words))
    out = {"train": compose(train_words, args.train_lines, rng), "val": compose(val_words, args.val_lines, rng),
           "test": compose(test_words, args.test_lines, rng)}
    print({k: len(v) for k, v in out.items()})
    pickle.dump(out, open(args.out, "wb"), protocol=4)


if __name__ == "__main__":
    main()
