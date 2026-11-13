"""What a worker actually does for each job kind."""

import logging
import time

import cv2
import numpy as np
from sqlalchemy import delete

from app.config import get_settings
from app.db import SessionLocal
from app.models import Job, Line, Manuscript, Page
from app.ocr import chain
from app.ocr.chain import NoEngineAvailable
from app.ocr.online import OnlineError, align, read_page
from app.ocr.postprocess import normalize_text
from app.services import runtime
from app.ocr.metrics import cer, wer
from app.ocr.pipeline import run_page_ocr
from app.services import audit, jobs, search
from app.services.storage import load_image, put_image, storage
from app.services.synthetic import make_page

log = logging.getLogger(__name__)


class PermanentError(RuntimeError):
    """Retrying will not help (bad input, no engine for the script)."""


def run_job(job_id: str) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            return
        kind = job.kind
    try:
        if kind == "ocr":
            engine_used, result = _ocr(job_id)
        elif kind == "enhance":
            engine_used, result = _enhance(job_id)
        elif kind == "benchmark":
            engine_used, result = _benchmark(job_id)
        else:
            raise PermanentError(f"unknown job kind {kind}")
        jobs.finish(job_id, engine_used, result)
        if kind == "ocr":
            _maybe_queue_enhance(job_id)
    except OnlineError as exc:  # already retried inside read_page; a 3-minute retry loop helps nobody
        log.warning("job %s: online reading failed: %s", job_id, exc)
        chain.breaker.failure("nvidia")
        jobs.fail(job_id, f"Online reading failed ({exc}). The offline reading is unchanged.", retryable=False)
    except (PermanentError, NoEngineAvailable) as exc:
        log.warning("job %s failed permanently: %s", job_id, exc)
        jobs.fail(job_id, str(exc), retryable=False)
    except Exception as exc:
        log.exception("job %s crashed", job_id)
        jobs.fail(job_id, f"{type(exc).__name__}: {exc}", retryable=True)


def _maybe_queue_enhance(ocr_job_id: str) -> None:
    """When online mode is on, follow every automatic local reading with an online pass."""
    if not runtime.online_active() or chain.breaker.is_open("nvidia"):
        return
    with SessionLocal() as db:
        job = db.get(Job, ocr_job_id)
        if not job or job.status != "done" or job.engine_requested not in ("auto", "online"):
            return
        jobs.enqueue(db, kind="enhance", page_id=job.page_id, engine="nvidia", user_id=job.created_by, touch_page=False)
        db.commit()


def _enhance(job_id: str) -> tuple[str, dict]:
    """Replace unverified local readings with the hosted VLM's transcription of the page."""
    if not runtime.online_active():
        raise PermanentError("online reading is switched off (offline mode)")
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        page = db.get(Page, job.page_id) if job.page_id else None
        if not page:
            raise PermanentError("page no longer exists")
        ms = db.get(Manuscript, page.manuscript_id)
        script, key = ms.script, page.normalized_key or page.image_key
        local = [(ln.id, ln.text, ln.verified) for ln in db.query(Line).filter(Line.page_id == page.id).order_by(Line.line_no)]
    if not local:
        raise PermanentError("this page has no line boxes yet; run the normal reading first")
    jobs.report(job_id, 10, "Sending the page to the online reader")
    image = load_image(key)
    label = {"devanagari": "Devanagari (Sanskrit / Prakrit / Hindi)"}.get(script, script)
    online_lines = read_page(image, label, beat=lambda: jobs.report(job_id, 30, "Waiting for the online reader"))
    chain.breaker.success("nvidia")
    if not runtime.online_active():  # switched to offline mode while we were waiting: discard the answer
        raise PermanentError("online reading was switched off while this page was being read; nothing was changed")
    jobs.report(job_id, 85, "Placing the transcription on the line boxes")
    pairs = align([t for _, t, _ in local], online_lines)
    replaced = 0
    with SessionLocal() as db:
        page = db.get(Page, job.page_id)
        if not page:
            raise PermanentError("page was deleted while processing")
        changed = []
        for i, j in pairs:
            ln = db.get(Line, local[i][0])
            if ln is None or ln.verified:
                continue
            text = normalize_text(online_lines[j], script)
            if text and text != ln.text:
                ln.text = ln.ocr_text = text
                ln.confidence, ln.engine = 0.8, "nvidia"
                changed.append(ln)
                replaced += 1
        db.flush()
        search.index_lines(db, changed)
        matched = len(pairs) / max(len(local), 1)
        page.engine = "nvidia+local"
        page.status = "done" if matched >= 0.7 else "partial"
        page.quality_note = "" if matched >= 0.7 else f"online reading matched only {len(pairs)} of {len(local)} line boxes"
        scores = [ln.confidence for ln in page.lines if ln.text]
        page.mean_confidence = round(float(np.mean(scores)), 4) if scores else page.mean_confidence
        audit.record(db, job.created_by, "ocr.enhanced", "page", page.id, replaced=replaced, online_lines=len(online_lines))
        db.commit()
    return "nvidia", {"replaced": replaced, "matched": len(pairs), "local_lines": len(local), "online_lines": len(online_lines),
                      "model": get_settings().nvidia_model}


def resolve_auto_engine(engine: str, material: str) -> str:
    """"auto" means: printed books -> our CRNN+PP-OCR ensemble (the CRNN was trained on printed lines);
    manuscripts -> PP-OCR first (it read real manuscript folios clearly better than the CRNN), with the
    other engines as fallbacks.  An explicit engine choice is never overridden."""
    if engine == "auto" and material != "print":
        return "ppocr"
    return engine


def _ocr(job_id: str) -> tuple[str, dict]:
    settings = get_settings()
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        page = db.get(Page, job.page_id) if job.page_id else None
        if not page:
            raise PermanentError("page no longer exists")
        ms = db.get(Manuscript, page.manuscript_id)
        script, engine, image_key = ms.script, job.engine_requested, page.image_key
        engine = resolve_auto_engine(engine, ms.material)
        page.status = "processing"
        db.commit()

    jobs.report(job_id, 5, "Loading image")
    image = load_image(image_key)
    result = run_page_ocr(
        image, script, engine, max_side=settings.processing_max_side, progress=lambda p, s: jobs.report(job_id, p, s)
    )

    base = image_key.rsplit("/", 1)[0]
    normalized_key = put_image(f"{base}/normalized.jpg", result.pre.color)
    binary_key = put_image(f"{base}/binary.png", 255 - result.pre.binary)

    with SessionLocal() as db:
        page = db.get(Page, job.page_id)
        if not page:
            raise PermanentError("page was deleted while processing")
        search.unindex_page(db, page.id)
        db.execute(delete(Line).where(Line.page_id == page.id))
        new_lines = [
            Line(
                page_id=page.id,
                line_no=i + 1,
                x=ln.box.x,
                y=ln.box.y,
                w=ln.box.w,
                h=ln.box.h,
                text=ln.text,
                ocr_text=ln.text,
                confidence=ln.confidence,
                engine=result.engine,
            )
            for i, ln in enumerate(result.lines)
        ]
        db.add_all(new_lines)
        db.flush()
        search.index_lines(db, new_lines)
        h, w = result.pre.color.shape[:2]
        page.normalized_key, page.binary_key = normalized_key, binary_key
        page.norm_width, page.norm_height = w, h
        quality, reasons = result.quality
        page.status, page.engine = ("done" if quality == "complete" else "partial"), result.engine
        page.quality_note = "; ".join(reasons)[:300]
        page.mean_confidence = result.mean_confidence
        page.skew_angle = result.pre.skew_angle
        audit.record(db, job.created_by, "ocr.completed", "page", page.id, lines=len(new_lines), engine=result.engine)
        db.commit()
    return result.engine, {
        "quality": quality.upper(),  # COMPLETE | PARTIAL | EMPTY - never a silent success
        "quality_reasons": reasons,
        "lines_segmented": result.segmented,
        "lines": len(result.lines),
        "mean_confidence": result.mean_confidence,
        "skew_angle": result.pre.skew_angle,
        "timings_ms": result.timings_ms,
    }


VARIANTS = {
    "stock": "Off-the-shelf PP-OCR (own detector)",
    "raw": "Our segmentation, no clean-up",
    "full": "LipiSetu full pipeline",
}


def _benchmark(job_id: str) -> tuple[str, dict]:
    from app.ocr.engines import get_engine

    with SessionLocal() as db:
        job = db.get(Job, job_id)
        opts = dict(job.options or {})
    script = opts.get("script", "devanagari")
    pages = int(opts.get("pages", 6))
    difficulty = float(opts.get("difficulty", 1.0))
    rapid = get_engine("ppocr")
    if not rapid or not rapid.supports(script):
        raise PermanentError(f"The benchmark needs the PP-OCR engine for {script}")

    totals = {k: {"cer": 0.0, "wer": 0.0, "ms": 0} for k in VARIANTS}
    per_page = []
    samples = []
    for i in range(pages):
        jobs.report(job_id, 5 + int(90 * i / pages), f"Page {i + 1} of {pages}")
        page = make_page(script, seed=1000 + i, difficulty=difficulty)
        truth = " ".join(page.lines)
        row = {"seed": 1000 + i, "lines": len(page.lines), "skew": page.skew, "font": page.font}

        t = time.perf_counter()
        stock = rapid.read_page_stock(page.image, script)
        ms = int((time.perf_counter() - t) * 1000)
        row["stock"] = {"cer": round(cer(stock, truth), 4), "wer": round(wer(stock, truth), 4), "ms": ms}

        for variant, use_pre in (("raw", False), ("full", True)):
            t = time.perf_counter()
            res = run_page_ocr(page.image, script, "ppocr", use_preprocessing=use_pre)
            ms = int((time.perf_counter() - t) * 1000)
            text = " ".join(ln.text for ln in res.lines)
            row[variant] = {
                "cer": round(cer(text, truth), 4),
                "wer": round(wer(text, truth), 4),
                "ms": ms,
                "lines_found": len(res.lines),
            }
            if variant == "full" and i < 3:
                prefix = f"lab/{job_id}/p{i}"
                samples.append(
                    {
                        "original": storage.url(put_image(f"{prefix}-original.jpg", page.image)),
                        "normalized": storage.url(put_image(f"{prefix}-normalized.jpg", _draw_boxes(res))),
                        "binary": storage.url(put_image(f"{prefix}-binary.png", 255 - res.pre.binary)),
                        "truth": page.lines,
                        "predicted": [ln.text for ln in res.lines],
                        "skew_true": page.skew,
                        "skew_estimated": res.pre.skew_angle,
                    }
                )
        for k in VARIANTS:
            for m in ("cer", "wer", "ms"):
                totals[k][m] += row[k][m]
        per_page.append(row)

    summary = {
        k: {
            "label": VARIANTS[k],
            "cer": round(totals[k]["cer"] / pages, 4),
            "wer": round(totals[k]["wer"] / pages, 4),
            "ms": round(totals[k]["ms"] / pages),
        }
        for k in VARIANTS
    }
    jobs.report(job_id, 96, "Measuring real manuscripts")
    return "ppocr", {
        "script": script,
        "pages": pages,
        "difficulty": difficulty,
        "summary": summary,
        "per_page": per_page,
        "samples": samples,
        "real_coverage": _real_coverage(rapid, script),
    }


def _real_coverage(rapid, script: str, limit: int = 8) -> list[dict]:
    """Real handwritten folios have no ground truth, so we compare how much text each approach finds:
    the stock detector on the original photo vs the lines LipiSetu read (stored in the database)."""
    from sqlalchemy import func, select

    with SessionLocal() as db:
        rows = db.execute(
            select(Page, Manuscript)
            .join(Manuscript)
            .where(Manuscript.script == script, Manuscript.material != "print", Page.status == "done")
            .order_by(Manuscript.title, Page.page_no)
            .limit(limit)
        ).all()
        counts = dict(
            db.execute(
                select(Line.page_id, func.sum(func.length(func.replace(Line.text, " ", "")))).where(Line.page_id.in_([p.id for p, _ in rows])).group_by(Line.page_id)
            ).all()
        )
        line_counts = dict(
            db.execute(select(Line.page_id, func.count(Line.id)).where(Line.page_id.in_([p.id for p, _ in rows])).group_by(Line.page_id)).all()
        )
    out = []
    for page, ms in rows:
        try:
            stock_text = rapid.read_page_stock(load_image(page.image_key), script)
        except Exception:  # a single unreadable image should not fail the whole run
            continue
        out.append(
            {
                "manuscript": ms.title,
                "page_id": page.id,
                "page_no": page.page_no,
                "stock_chars": len(stock_text.replace(" ", "")),
                "ours_chars": int(counts.get(page.id) or 0),
                "our_lines": int(line_counts.get(page.id) or 0),
            }
        )
    return out


def _draw_boxes(res) -> np.ndarray:
    vis = res.pre.color.copy()
    for ln in res.lines:
        b = ln.box
        color = (60, 160, 60) if ln.confidence >= 0.85 else (40, 160, 230) if ln.confidence >= 0.6 else (60, 60, 220)
        cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
    return vis
