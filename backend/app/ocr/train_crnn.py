"""Train the CRNN line recogniser (CNN + biLSTM + CTC).

Data comes from ``scripts/build_ocr_dataset.py`` (a pickle of line images + text, split by source
page so test pages are never seen in training).

    python -m scripts.build_ocr_dataset --snskrt-parquet ... --labels ... --synth-dir ... --out lines.pkl
    python -m app.ocr.train_crnn --data lines.pkl --epochs 30

Model selection uses the REAL validation split only; the REAL test split is evaluated once at the
end and written to ``data/crnn_metrics.json`` next to the checkpoint.  The metric is character
error rate (CER = edit distance / reference length, on NFC code points), the same definition the
OCR Lab uses.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time

import cv2
import numpy as np

log = logging.getLogger("train_crnn")


# --------------------------------------------------------------------------- augmentation


def augment(img: np.ndarray, rng: random.Random, strong: bool) -> np.ndarray:
    """Cheap photometric/geometric jitter so the model sees scan-like variation."""
    h, w = img.shape
    out = img
    if rng.random() < 0.7:  # horizontal stretch
        out = cv2.resize(out, (max(8, int(w * rng.uniform(0.85, 1.2))), h), interpolation=cv2.INTER_LINEAR)
        h, w = out.shape
    if rng.random() < 0.5:  # small rotation + shear
        ang, shear = rng.uniform(-1.5, 1.5), rng.uniform(-0.15, 0.15)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        m[0, 1] += shear
        out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.35:  # stroke thickness
        k = np.ones((2, 2), np.uint8)
        out = cv2.erode(out, k) if rng.random() < 0.5 else cv2.dilate(out, k)
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.2))
    f = out.astype(np.float32)
    if rng.random() < 0.8:  # contrast / brightness / ink fading
        c = rng.uniform(0.55, 1.1) if strong else rng.uniform(0.8, 1.1)
        f = (f - 255.0) * c + 255.0 + rng.uniform(-25, 15)
    if rng.random() < 0.5:  # paper tone gradient
        grad = np.linspace(rng.uniform(-18, 18), rng.uniform(-18, 18), w, dtype=np.float32)
        f += grad[None, :]
    if rng.random() < 0.5:
        f += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 10), f.shape).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- batching

_WIDTH_BUCKETS = (384, 512, 640, 768, 1024, 1280, 1536, 2048)


def make_batch(items, vocab_index, out_len, aug_rng=None, strong=False):
    import torch

    imgs = []
    for img, text, _ in items:
        imgs.append(augment(img, aug_rng, strong) if aug_rng else img)
    max_w = max(i.shape[1] for i in imgs)
    max_w = next((b for b in _WIDTH_BUCKETS if b >= max_w), max_w + (-max_w) % 64)  # few distinct shapes: MPS caches per shape
    batch = np.full((len(imgs), 1, imgs[0].shape[0], max_w), 255, np.uint8)
    for k, im in enumerate(imgs):
        batch[k, 0, :, : im.shape[1]] = im
    targets = [torch.tensor([vocab_index[c] for c in t], dtype=torch.long) for _, t, _ in items]
    in_lens = torch.tensor([min(out_len(im.shape[1]), max_w // 4) for im in imgs], dtype=torch.long)
    tg_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return torch.from_numpy(batch).float().div_(255.0), torch.cat(targets), in_lens, tg_lens


def bucketed_batches(items, batch_size, rng, shuffle=True):
    """Group lines of similar width so padding (and compute) stays small."""
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    chunks = [idx[i : i + batch_size * 50] for i in range(0, len(idx), batch_size * 50)]
    batches = []
    for ch in chunks:
        ch.sort(key=lambda i: items[i][0].shape[1])
        batches += [ch[i : i + batch_size] for i in range(0, len(ch), batch_size)]
    if shuffle:  # training: full batches only, so tensor shapes repeat
        batches = [b for b in batches if len(b) == batch_size]
    if shuffle:
        rng.shuffle(batches)
    return batches


# --------------------------------------------------------------------------- evaluation


def evaluate(model, items, vocab, device, batch_size=64):
    """Return (CER, exact-match accuracy) over ``items`` using greedy CTC decoding."""
    import torch
    from rapidfuzz.distance import Levenshtein

    from app.ocr.engines.crnn import CRNN, ctc_greedy_decode

    vocab_index = {c: i for i, c in enumerate(vocab)}
    model.eval()
    edits = refs = exact = 0
    rng = random.Random(0)
    with torch.no_grad():
        for b in bucketed_batches(items, batch_size, rng, shuffle=False):
            chunk = [items[i] for i in b]
            padded = chunk + [chunk[-1]] * (batch_size - len(chunk))  # fixed batch size, extras ignored
            x, _, in_lens, _ = make_batch(padded, vocab_index, CRNN.out_len)
            lp = model(x.to(device)).cpu()
            for (_, ref, _), seq in zip(chunk, ctc_greedy_decode(lp)[: len(chunk)]):
                pred = "".join(vocab[i] for i in seq)
                edits += Levenshtein.distance(pred, ref)
                refs += len(ref)
                exact += pred == ref
    return edits / max(refs, 1), exact / max(len(items), 1)


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the CRNN line recogniser")
    ap.add_argument("--data", required=True, help="pickle from scripts.build_ocr_dataset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--real-repeat", type=int, default=2, help="oversampling factor for real lines")
    ap.add_argument("--synth-per-epoch", type=int, default=6000)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after this wall time (0 = no limit)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="checkpoint path (default: data/crnn_sanskrt.pt)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hw-data", default=None, help="pickle from scripts.build_hw_lines (handwritten-style lines)")
    ap.add_argument("--hw-per-epoch", type=int, default=9000)
    ap.add_argument("--init", default=None, help="fine-tune from this checkpoint (same vocabulary)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import torch
    import torch.nn as nn

    from app.ocr.engines.crnn import _DEFAULT_MODEL_PATH, CRNN

    out_path = type(_DEFAULT_MODEL_PATH)(args.out) if args.out else _DEFAULT_MODEL_PATH
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    blob = pickle.load(open(args.data, "rb"))
    vocab, data = blob["vocab"], blob["data"]
    vocab_index = {c: i for i, c in enumerate(vocab)}
    real_tr, real_va, real_te = data["real"]["train"], data["real"]["val"], data["real"]["test"]
    syn_tr, syn_va = data["synth"]["train"], data["synth"]["val"]
    log.info("device=%s vocab=%d real train/val/test=%d/%d/%d synth train/val=%d/%d", device, len(vocab) - 1,
             len(real_tr), len(real_va), len(real_te), len(syn_tr), len(syn_va))

    hw = pickle.load(open(args.hw_data, "rb")) if args.hw_data else None
    model = CRNN(n_classes=len(vocab))
    if args.init:
        ck0 = torch.load(args.init, map_location="cpu", weights_only=True)
        assert ck0["vocab"] == vocab, "fine-tuning needs the same vocabulary as the checkpoint"
        model.load_state_dict(ck0["model_state"])
        log.info("initialised from %s", args.init)
    model = model.to(device)
    log.info("parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)
    ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = (len(real_tr) * args.real_repeat + args.synth_per_epoch + (args.hw_per_epoch if hw else 0)) // args.batch_size + 2
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.15)

    best, bad, t0 = 9e9, 0, time.time()
    history = []
    for epoch in range(1, args.epochs + 1):
        items = real_tr * args.real_repeat + rng.sample(syn_tr, min(args.synth_per_epoch, len(syn_tr)))
        if hw:
            items += rng.sample(hw["train"], min(args.hw_per_epoch, len(hw["train"])))
        batches = bucketed_batches(items, args.batch_size, rng)
        model.train()
        total, n = 0.0, 0
        for step, b in enumerate(batches):
            chunk = [items[i] for i in b]
            strong = [c[2] == "synth" for c in chunk]
            x, tg, in_lens, tg_lens = make_batch(chunk, vocab_index, CRNN.out_len, rng, strong=any(strong))
            lp = model(x.to(device)).float().cpu()  # CTC runs on CPU (not implemented on MPS)
            loss = ctc(lp, tg, in_lens, tg_lens)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            total += loss.item()
            n += 1
            if device.type == "mps" and step % 50 == 0:
                torch.mps.empty_cache()
            if step % 100 == 0:
                log.info("epoch %d step %d/%d loss %.3f (%.1f min)", epoch, step, len(batches), loss.item(), (time.time() - t0) / 60)
        if device.type == "mps":
            torch.mps.empty_cache()
        cer_real, acc_real = evaluate(model, real_va, vocab, device)
        cer_syn, _ = evaluate(model, syn_va[:600], vocab, device)
        cer_hw = evaluate(model, hw["val"][:700], vocab, device)[0] if hw else 0.0
        # Model selection: printed-Sanskrit CER, plus handwritten CER when that data is used, so the
        # model cannot trade one for the other unnoticed.
        score = cer_real + cer_hw
        history.append({"epoch": epoch, "train_loss": total / n, "val_cer_real": cer_real, "val_cer_synth": cer_syn, "val_cer_hw": cer_hw})
        log.info("== epoch %d  loss %.3f  real val CER %.2f%% (exact %.1f%%)  synth val CER %.2f%%  handwritten val CER %.2f%%  [%.1f min]",
                 epoch, total / n, 100 * cer_real, 100 * acc_real, 100 * cer_syn, 100 * cer_hw, (time.time() - t0) / 60)
        if score < best:
            best, bad = score, 0
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".pt.tmp")
            torch.save({"model_state": {k: v.cpu() for k, v in model.state_dict().items()}, "vocab": vocab,
                        "scripts": ["devanagari"], "val_cer": float(cer_real), "val_acc": float(1 - cer_real), "val_cer_hw": float(cer_hw),
                        "val_loss": float(total / n), "n_examples": len(real_tr) + len(syn_tr), "epochs": epoch}, tmp)
            os.replace(tmp, out_path)
            log.info("saved best checkpoint (real val CER %.2f%%)", 100 * cer_real)
        else:
            bad += 1
            if bad >= args.patience:
                log.info("early stop (no improvement for %d epochs)", bad)
                break
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            log.info("time budget reached")
            break

    # Final, one-shot test on held-out pages with the best checkpoint.
    ck = torch.load(out_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state"])
    model.to(device)
    te_cer, te_acc = evaluate(model, real_te, vocab, device)
    sy_cer, sy_acc = evaluate(model, syn_va, vocab, device)
    hw_cer = evaluate(model, hw["test"], vocab, device)[0] if hw else None
    metrics = {"test_handwritten_cer": hw_cer, "test_real_cer": te_cer, "test_real_exact": te_acc, "n_test_real": len(real_te),
               "synth_val_cer": sy_cer, "synth_val_exact": sy_acc, "best_val_cer_real": best,
               "epochs_run": len(history), "minutes": (time.time() - t0) / 60, "history": history}
    (out_path.parent / "crnn_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("TEST (held-out real pages, %d lines): CER %.2f%%  exact-line %.1f%%  | synth val CER %.2f%%",
             len(real_te), 100 * te_cer, 100 * te_acc, 100 * sy_cer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
