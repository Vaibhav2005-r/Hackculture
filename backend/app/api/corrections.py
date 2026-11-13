from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, selectinload

from app.api import get_or_404
from app.api.lines import _correction_out, apply_correction
from app.db import get_db
from app.models import Correction, Line, Manuscript, Page, User, utcnow
from app.schemas import ReviewIn
from app.security import current_user, require_role
from app.services import audit

router = APIRouter(prefix="/corrections", tags=["corrections"])


def _with_context(db: Session, items: list[Correction]) -> list[dict]:
    page_ids = {c.line.page_id for c in items if c.line}
    pages = {
        p.id: (p, ms)
        for p, ms in db.execute(select(Page, Manuscript).join(Manuscript).where(Page.id.in_(page_ids))).all()
    }
    out = []
    for c in items:
        d = _correction_out(c)
        page, ms = pages.get(d["page_id"], (None, None))
        d["page_no"] = page.page_no if page else None
        d["manuscript_id"] = ms.id if ms else None
        d["manuscript_title"] = ms.title if ms else None
        d["script"] = ms.script if ms else None
        out.append(d)
    return out


@router.get("")
def list_corrections(
    status_: str = Query("pending", alias="status", pattern="^(pending|approved|rejected|all)$"),
    limit_: int = Query(50, alias="limit", ge=1, le=200),
    user: User = Depends(require_role("curator")),
    db: Session = Depends(get_db),
):
    stmt = select(Correction).options(selectinload(Correction.line), selectinload(Correction.user))
    if status_ != "all":
        stmt = stmt.where(Correction.status == status_)
    items = db.scalars(stmt.order_by(Correction.created_at.desc()).limit(limit_)).all()
    counts = dict(db.execute(select(Correction.status, func.count(Correction.id)).group_by(Correction.status)).all())
    return {"counts": counts, "items": _with_context(db, list(items))}


def _flagged_filter(max_conf: float):
    pending_lines = select(Correction.line_id).where(Correction.status == "pending")
    return (
        (Line.verified.is_(False))
        & (Line.confidence < max_conf)
        & (Line.text != "")
        & (Page.status.in_(("done", "partial")))
        & (Line.id.not_in(pending_lines))  # lines with a pending suggestion are already in the other tab
    )


@router.get("/flagged")
def flagged_lines(
    limit_: int = Query(50, alias="limit", ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(require_role("curator")),
    db: Session = Depends(get_db),
):
    """Unverified lines the machine is not sure about (yellow and red), least confident first."""
    from app.config import get_settings

    max_conf = get_settings().review_confidence_threshold
    where = _flagged_filter(max_conf)
    total = db.scalar(select(func.count(Line.id)).join(Page, Line.page_id == Page.id).where(where)) or 0
    rows = db.execute(
        select(Line, Page, Manuscript)
        .join(Page, Line.page_id == Page.id)
        .join(Manuscript, Page.manuscript_id == Manuscript.id)
        .where(where)
        .order_by(Line.confidence.asc(), Manuscript.title, Page.page_no, Line.line_no)
        .limit(limit_)
        .offset(offset)
    ).all()
    return {
        "total": total,
        "threshold": max_conf,
        "items": [
            {
                "line_id": ln.id,
                "page_id": pg.id,
                "line_no": ln.line_no,
                "page_no": pg.page_no,
                "text": ln.text,
                "confidence": ln.confidence,
                "tone": "red" if ln.confidence < 0.6 else "yellow",
                "engine": ln.engine,
                "manuscript_id": ms.id,
                "manuscript_title": ms.title,
                "script": ms.script,
            }
            for ln, pg, ms in rows
        ],
    }


@router.post("/flagged/{line_id}/confirm")
def confirm_flagged_line(line_id: str, user: User = Depends(require_role("curator")), db: Session = Depends(get_db)):
    """A curator checked the line and the text is right as it stands: mark it verified."""
    line = get_or_404(db, Line, line_id, "Line")
    if line.verified:
        raise HTTPException(status.HTTP_409_CONFLICT, "This line is already verified")
    line.verified = True
    audit.record(db, user.id, "line.confirmed", "line", line.id, page_id=line.page_id)
    db.commit()
    return {"line_id": line.id, "verified": True}


@router.get("/mine")
def my_corrections(user: User = Depends(current_user), db: Session = Depends(get_db)):
    items = db.scalars(
        select(Correction)
        .options(selectinload(Correction.line), selectinload(Correction.user))
        .where(Correction.user_id == user.id)
        .order_by(Correction.created_at.desc())
        .limit(100)
    ).all()
    return {"items": _with_context(db, list(items))}


@router.post("/{correction_id}/review")
def review(correction_id: str, body: ReviewIn, user: User = Depends(require_role("curator")), db: Session = Depends(get_db)):
    c = get_or_404(db, Correction, correction_id, "Correction")
    if c.status != "pending":
        raise HTTPException(status.HTTP_409_CONFLICT, f"This correction was already {c.status}")
    if not db.get(Line, c.line_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The line no longer exists")
    # Atomic claim: only one concurrent reviewer can flip pending -> reviewing.  The UPDATE takes the
    # database write lock, so a second reviewer waits, then sees a non-pending row and gets a 409.
    claimed = db.execute(
        update(Correction).where(Correction.id == c.id, Correction.status == "pending").values(status="reviewing")
    ).rowcount
    if claimed != 1:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "This correction was just reviewed by someone else")
    db.refresh(c)
    if body.action == "approve":
        apply_correction(db, c, user, body.text)
        c.review_note = body.note
    else:
        c.status, c.review_note, c.reviewer_id, c.reviewed_at = "rejected", body.note, user.id, utcnow()
    audit.record(db, user.id, f"correction.{body.action}d", "correction", c.id, line_id=c.line_id)
    db.commit()
    db.refresh(c)
    return _with_context(db, [c])[0]
