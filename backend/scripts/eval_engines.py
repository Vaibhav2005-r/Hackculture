"""Head-to-head accuracy of every OCR engine on held-out line images.

Two test sets, both never used for training or model selection:
  * real: the held-out source pages of snskrt/Sanskrit_OCR_Parallel_Corpus (printed Sanskrit verses,
    scanned from a book), split into lines exactly as in training but kept at ORIGINAL resolution so
    third-party engines are not handicapped by our 48 px squeeze;
  * synth: the validation split of iamkushagratomar/devanagari-ocr-synthetic-80k (Hindi, many fonts).

    python -m scripts.eval_engines --snskrt-parquet X --labels Y --synth-val Z --out report.json
"""

import argparse
import csv
import json
import time

import cv2
import numpy as np
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

from app.ocr.engines import get_engine
from app.ocr.chain import recognize as chain_recognize
from scripts.build_ocr_dataset import group_split, nfc, split_two_lines


def load_real(parquet, labels_csv, limit):
    labels = {r["image_name"]: [nfc(l) for l in r["shlok"].split("\n") if l.strip()] for r in csv.DictReader(open(labels_csv, encoding="utf-8"))}
    out = []
    for r in pq.read_table(parquet).to_pylist():
        name = r["image"]["path"].split("/")[-1]
        if group_split(name.split("_")[0]) != "test":
            continue
        img = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
        crops = split_two_lines(img)
        if img is None or len(crops) != len(labels.get(name, [])):
            continue
        out += [(cv2.cvtColor(c, cv2.COLOR_GRAY2BGR), t) for c, t in zip(crops, labels[name])]
    return out[:limit]


def load_synth(parquet, limit):
    out = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=200, columns=["image", "text"]):
        for row in batch.to_pylist():
            img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if img is not None and row["text"]:
                out.append((img, nfc(row["text"])))
            if len(out) >= limit:
                return out
    return out


def score(predict, items, batch=32):
    edits = refs = exact = 0
    per_line = []
    t0 = time.time()
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        preds = predict([c[0] for c in chunk])
        for (_, ref), pred in zip(chunk, preds):
            pred = nfc(pred)
            d = Levenshtein.distance(pred, ref)
            edits += d
            refs += len(ref)
            exact += pred == ref
            per_line.append(d / max(len(ref), 1))
    n = len(items)
    return {"cer": edits / max(refs, 1), "exact_line": exact / max(n, 1), "median_line_cer": float(np.median(per_line)),
            "p90_line_cer": float(np.percentile(per_line, 90)), "ms_per_line": 1000 * (time.time() - t0) / max(n, 1), "lines": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snskrt-parquet", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--synth-val", required=True)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sets = {"real_sanskrit_test": load_real(args.snskrt_parquet, args.labels, args.limit), "synthetic_hindi_val": load_synth(args.synth_val, args.limit)}
    print({k: len(v) for k, v in sets.items()})
    report = {}
    names = [n for n in ("crnn", "ppocr", "tesseract") if get_engine(n) is not None and get_engine(n).supports("devanagari")]
    for set_name, items in sets.items():
        report[set_name] = {}
        for n in names:
            eng = get_engine(n)
            report[set_name][n] = score(lambda imgs, e=eng: [r.text for r in e.recognize(imgs, "devanagari")], items)
        ens = get_engine("ensemble")
        if ens is not None and ens.supports("devanagari"):
            report[set_name]["ensemble"] = score(lambda imgs: [r.text for r in ens.recognize(imgs, "devanagari")], items)
        report[set_name]["auto(chain)"] = score(lambda imgs: [r.text for r in chain_recognize(imgs, "devanagari", "auto")[1]], items)
        print(f"\n{set_name}")
        for n, m in report[set_name].items():
            print(f"  {n:12s} CER {100 * m['cer']:6.2f}%  exact-line {100 * m['exact_line']:5.1f}%  median-line CER {100 * m['median_line_cer']:5.1f}%  p90 {100 * m['p90_line_cer']:5.1f}%  {m['ms_per_line']:.0f} ms/line")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
