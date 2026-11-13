"""Fit app/ocr/confidence.py on held-out validation lines (never the test pages).

    python -m scripts.fit_confidence --snskrt-parquet X --labels Y --synth-val Z
"""

import argparse
import json

import numpy as np
from rapidfuzz.distance import Levenshtein

from app.ocr.confidence import PATH, agreement, features, variants
from app.ocr.engines import get_engine
from scripts.eval_engines import load_synth
from scripts.fit_calibrator import load_real_val


def auc(scores, y):
    pos, neg = scores[y == 1], scores[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snskrt-parquet", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--synth-val", required=True)
    ap.add_argument("--ok-cer", type=float, default=0.30)
    args = ap.parse_args()

    items = load_real_val(args.snskrt_parquet, args.labels) + load_synth(args.synth_val, 1400)[400:1400]
    pp = get_engine("ppocr")
    X, y, raw = [], [], []
    for i in range(0, len(items), 32):
        chunk = items[i : i + 32]
        imgs = [x[0] for x in chunk]
        base = pp.recognize(imgs, "devanagari")
        thin, small = variants(imgs)
        rt, rs = pp.recognize(thin, "devanagari"), pp.recognize(small, "devanagari")
        for (img, ref), a, t, sm in zip(chunk, base, rt, rs):
            X.append(features(a.confidence, agreement(a.text, t.text), agreement(a.text, sm.text), a.text))
            y.append(1.0 if Levenshtein.distance(a.text, ref) / max(len(ref), 1) <= args.ok_cer else 0.0)
            raw.append(a.confidence)
    X, y, raw = np.array(X), np.array(y), np.array(raw)
    # standardise the non-bias columns while fitting, then fold back into raw-feature weights
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
    Z = np.hstack([X[:, :1], (X[:, 1:] - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(400):  # IRLS / Newton with a small ridge
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + 1e-3 * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + 1e-3 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    wr = np.concatenate([[w[0] - float(((w[1:] * mu) / sd).sum())], w[1:] / sd])
    p = 1 / (1 + np.exp(-X @ wr))
    print(f"{len(y)} lines, {y.mean():.1%} read within {args.ok_cer:.0%} CER by PP-OCR")
    print(f"AUC raw PP-OCR confidence: {auc(raw, y):.3f}   calibrated (agreement-aware): {auc(p, y):.3f}")
    bins = np.linspace(0, 1, 6)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi + (hi == 1))
        if m.sum():
            print(f"  predicted {lo:.1f}-{hi:.1f}: {m.sum():4d} lines, actually good {y[m].mean():.2f}")
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps({"w": wr.tolist(), "ok_cer": args.ok_cer, "n": int(len(y)), "auc_raw": auc(raw, y), "auc_calibrated": auc(p, y)}))
    print("saved", PATH)


if __name__ == "__main__":
    main()
