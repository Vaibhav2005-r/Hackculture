"""Fit the CNN confidence calibrator on held-out *validation* lines.

For every validation line we run each local engine and record whether it read the line essentially
correctly (CER <= --ok-cer).  The calibrator then learns, from the line image alone, which engine to
trust.  Validation pages never overlap the test pages used by ``eval_engines.py``.

    python -m scripts.fit_calibrator --snskrt-parquet X --labels Y --synth-val Z
"""

import argparse
import csv
import logging

import cv2
import numpy as np
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

from app.ocr.calibrator import train
from app.ocr.engines import get_engine
from scripts.build_ocr_dataset import group_split, nfc, split_two_lines
from scripts.eval_engines import load_synth

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_real_val(parquet, labels_csv):
    labels = {r["image_name"]: [nfc(l) for l in r["shlok"].split("\n") if l.strip()] for r in csv.DictReader(open(labels_csv, encoding="utf-8"))}
    out = []
    for r in pq.read_table(parquet).to_pylist():
        name = r["image"]["path"].split("/")[-1]
        if group_split(name.split("_")[0]) != "val":
            continue
        img = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
        crops = split_two_lines(img) if img is not None else []
        if len(crops) == len(labels.get(name, [])):
            out += [(cv2.cvtColor(c, cv2.COLOR_GRAY2BGR), t) for c, t in zip(crops, labels[name])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snskrt-parquet", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--synth-val", required=True)
    ap.add_argument("--synth-skip", type=int, default=400, help="first N synthetic val lines are reserved for eval_engines")
    ap.add_argument("--synth-take", type=int, default=800)
    ap.add_argument("--ok-cer", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    items = load_real_val(args.snskrt_parquet, args.labels) + load_synth(args.synth_val, args.synth_skip + args.synth_take)[args.synth_skip :]
    engines = [e for e in (get_engine("crnn"), get_engine("ppocr")) if e is not None and e.supports("devanagari")]
    names = [e.name for e in engines]
    print(f"{len(items)} validation lines, engines={names}")

    preds = {e.name: [] for e in engines}
    for e in engines:
        for i in range(0, len(items), 32):
            preds[e.name] += [r.text for r in e.recognize([x[0] for x in items[i : i + 32]], "devanagari")]

    examples, correct_rate = [], {n: 0 for n in names}
    for k, (img, ref) in enumerate(items):
        labels = {}
        for n in names:
            ok = Levenshtein.distance(nfc(preds[n][k]), ref) / max(len(ref), 1) <= args.ok_cer
            labels[n] = ok
            correct_rate[n] += ok
        examples.append({"img": img, "labels": labels})
    print("share of lines each engine reads (CER <= %.0f%%): %s" % (100 * args.ok_cer, {n: round(c / len(items), 3) for n, c in correct_rate.items()}))
    print(train(examples=examples, engine_names=names, epochs=args.epochs, min_examples=100))


if __name__ == "__main__":
    main()
