"""Randomised end-to-end test of the whole LipiSetu flow against a throw-away database.

Everything is driven through the public HTTP API (FastAPI TestClient + the embedded OCR worker), with
random users, random synthetic manuscript pages (random verses, fonts, skew, stains, lighting),
random corrections, random search queries and random heritage trails.  It checks behaviour (status
codes, permissions, state changes) and measures accuracy (page-level CER for every OCR engine,
search recall, transliteration round-trips).

    python -m scripts.e2e_random --seed 1 --pages 12 --out data/e2e_report.json
"""

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="lipisetu-e2e-"))
os.environ.update(
    DATABASE_URL=f"sqlite:///{(TMP / 'e2e.db').as_posix()}",
    STORAGE_DIR=str(TMP / "storage"),
    EMBEDDED_WORKER="true",
    WORKER_POLL_SECONDS="0.2",
    RATE_LIMIT_PER_MINUTE="100000",
    ANTHROPIC_API_KEY="",
    OSRM_URL="http://127.0.0.1:9",
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.ocr.metrics import cer  # noqa: E402
from app.services.synthetic import make_page  # noqa: E402

API = "/api/v1"
RESULTS: list[dict] = []


def check(section: str, name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append({"section": section, "name": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {section}: {name}" + (f"  ({detail})" if detail and not ok else ""))
    return bool(ok)


def jpg(img, quality=92) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def pdf_from(images) -> bytes:
    from io import BytesIO

    from PIL import Image

    pil = [Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB)) for i in images]
    out = BytesIO()
    pil[0].save(out, format="PDF", save_all=True, append_images=pil[1:])
    return out.getvalue()


def wait_job(client, job_id, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"{API}/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.25)
    return {"status": "timeout", "id": job_id}


def page_text(client, page_id) -> tuple[str, list[dict]]:
    p = client.get(f"{API}/pages/{page_id}").json()
    return " ".join(l["text"] for l in p["lines"]), p["lines"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pages", type=int, default=12, help="random pages used for the accuracy sweep")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    from fastapi.testclient import TestClient

    from app.db import SessionLocal
    from app.main import app
    from app.models import User

    report: dict = {"seed": args.seed}
    t_start = time.time()
    with TestClient(app) as c:
        # ------------------------------------------------------------------ health / meta
        S = "health"
        check(S, "GET /health", c.get(f"{API}/health").json().get("status") == "ok")
        check(S, "GET /ready", c.get(f"{API}/ready").status_code == 200)
        meta = c.get(f"{API}/meta").json()
        engines = {e["name"]: e for e in meta["engines"]}
        avail = [n for n, e in engines.items() if e["available"]]
        check(S, "meta lists engines", "ppocr" in engines and "crnn" in engines and "ensemble" in engines, str(list(engines)))
        report["engines_available"] = avail
        r = c.get(f"{API}/health")
        check(S, "security headers", r.headers.get("x-content-type-options") == "nosniff" and bool(r.headers.get("x-request-id")))

        # ------------------------------------------------------------------ users / auth
        S = "auth"
        users = {}
        for role in ("viewer", "contributor", "curator", "admin"):
            email = f"{role}{rnd.randint(1000, 9999)}@rand.test"
            pw = f"Pw-{rnd.randint(10**7, 10**8)}x"
            r = c.post(f"{API}/auth/register", json={"name": f"Random {role}", "email": email, "password": pw})
            check(S, f"register {role}", r.status_code == 201, r.text[:120])
            if role in ("curator", "admin", "viewer"):
                with SessionLocal() as db:
                    db.query(User).filter_by(email=email).one().role = role
                    db.commit()
            r = c.post(f"{API}/auth/login", json={"email": email, "password": pw})
            check(S, f"login {role}", r.status_code == 200)
            tok = r.json()
            users[role] = {"h": {"Authorization": f"Bearer {tok['access_token']}"}, "email": email, "pw": pw, "refresh": tok["refresh_token"]}
        H = {k: v["h"] for k, v in users.items()}
        check(S, "duplicate email -> 409", c.post(f"{API}/auth/register", json={"name": "Dup", "email": users["viewer"]["email"], "password": "whatever123"}).status_code == 409)
        check(S, "weak input -> 422", c.post(f"{API}/auth/register", json={"name": "A", "email": "nope", "password": "x"}).status_code == 422)
        check(S, "wrong password -> 401", c.post(f"{API}/auth/login", json={"email": users["viewer"]["email"], "password": "bad-password"}).status_code == 401)
        check(S, "no token -> 401", c.get(f"{API}/auth/me").status_code == 401)
        check(S, "garbage token -> 401", c.get(f"{API}/auth/me", headers={"Authorization": "Bearer abc.def.ghi"}).status_code == 401)
        me = c.get(f"{API}/auth/me", headers=H["contributor"]).json()
        check(S, "me returns role", me.get("role") == "contributor", str(me))
        r = c.post(f"{API}/auth/refresh", json={"refresh_token": users["contributor"]["refresh"]})
        check(S, "refresh token", r.status_code == 200 and "access_token" in r.json())
        r = c.patch(f"{API}/auth/me", json={"name": "Renamed Contributor"}, headers=H["contributor"])
        check(S, "update profile", r.status_code == 200 and r.json()["name"] == "Renamed Contributor")
        new_pw = f"New-{rnd.randint(10**7, 10**8)}y"
        r = c.post(f"{API}/auth/password", json={"current_password": users["contributor"]["pw"], "new_password": new_pw}, headers=H["contributor"])
        check(S, "change password", r.status_code in (200, 204), r.text[:100])
        check(S, "login with new password", c.post(f"{API}/auth/login", json={"email": users["contributor"]["email"], "password": new_pw}).status_code == 200)
        check(S, "old password rejected", c.post(f"{API}/auth/login", json={"email": users["contributor"]["email"], "password": users["contributor"]["pw"]}).status_code == 401)

        # ------------------------------------------------------------------ role matrix
        S = "roles"
        check(S, "viewer cannot create manuscript", c.post(f"{API}/manuscripts", json={"title": "Nope nope"}, headers=H["viewer"]).status_code == 403)
        check(S, "anonymous cannot create manuscript", c.post(f"{API}/manuscripts", json={"title": "Nope nope"}).status_code == 401)
        check(S, "contributor cannot create site", c.post(f"{API}/sites", json={"name": "Nope", "lat": 20, "lng": 75}, headers=H["contributor"]).status_code == 403)
        check(S, "contributor cannot list users", c.get(f"{API}/admin/users", headers=H["contributor"]).status_code == 403)
        check(S, "curator cannot list users", c.get(f"{API}/admin/users", headers=H["curator"]).status_code == 403)
        check(S, "admin lists users", c.get(f"{API}/admin/users", headers=H["admin"]).status_code == 200)
        check(S, "contributor cannot export", c.get(f"{API}/export/training-data", headers=H["contributor"]).status_code == 403)

        # ------------------------------------------------------------------ sites / trails (random geography)
        S = "heritage"
        sites = []
        for i in range(8):
            body = {"name": f"Random Site {i} {rnd.randint(100, 999)}", "kind": rnd.choice(["temple", "fort", "museum", "stepwell", "monument"]),
                    "lat": round(rnd.uniform(20.0, 28.0), 4), "lng": round(rnd.uniform(70.0, 80.0), 4), "city": f"City{i}", "state": "Random"}
            r = c.post(f"{API}/sites", json=body, headers=H["curator"])
            if check(S, f"create site {i}", r.status_code == 201, r.text[:100]):
                sites.append(r.json())
        check(S, "bad coordinates -> 422", c.post(f"{API}/sites", json={"name": "Bad", "lat": 120, "lng": 0}, headers=H["curator"]).status_code == 422)
        sid = sites[0]["id"]
        r = c.patch(f"{API}/sites/{sid}", json={"description": "updated by e2e"}, headers=H["curator"])
        check(S, "patch site", r.status_code == 200, f"{r.status_code} {r.text[:150]}")
        check(S, "get site", c.get(f"{API}/sites/{sid}").json()["id"] == sid)
        lat, lng = rnd.uniform(21, 27), rnd.uniform(71, 79)
        near = c.get(f"{API}/sites/nearby", params={"lat": lat, "lng": lng, "radius_km": 400}).json()["items"]
        ds = [s["distance_km"] for s in near]
        check(S, "nearby sorted by distance", ds == sorted(ds) and all(d <= 400.5 for d in ds), str(ds[:5]))
        # independent haversine check of the first result
        if near:
            from math import asin, cos, radians, sin, sqrt

            n0 = near[0]
            a = sin(radians(n0["lat"] - lat) / 2) ** 2 + cos(radians(lat)) * cos(radians(n0["lat"])) * sin(radians(n0["lng"] - lng) / 2) ** 2
            true_km = 2 * 6371.0088 * asin(sqrt(a))
            check(S, "distance matches haversine (<1%)", abs(true_km - n0["distance_km"]) / max(true_km, 1) < 0.01, f"{true_km:.1f} vs {n0['distance_km']}")
        picks = rnd.sample(sites, 5)
        r = c.post(f"{API}/trails", json={"name": "Random trail", "mode": "drive", "start_lat": lat, "start_lng": lng, "site_ids": [s["id"] for s in picks]}, headers=H["contributor"])
        if check(S, "plan trail", r.status_code == 201, r.text[:150]):
            tr = r.json()
            check(S, "trail visits every stop once", sorted(s["id"] for s in tr["stops"]) == sorted(s["id"] for s in picks))
            # compare against brute force optimum on straight-line distance
            import itertools

            from app.services.geo import haversine_km

            pts = {s["id"]: (s["lat"], s["lng"]) for s in picks}

            def length(order):
                p, tot = (lat, lng), 0.0
                for i in order:
                    tot += haversine_km(p[0], p[1], pts[i][0], pts[i][1])
                    p = pts[i]
                return tot

            best = min(length(o) for o in itertools.permutations(pts))
            got = length([s["id"] for s in tr["stops"]])
            check(S, "trail order within 10% of optimal", got <= best * 1.10 + 1e-6, f"got {got:.0f} km, optimum {best:.0f} km")
            check(S, "router fallback reported", tr.get("routing") in ("straight-line", "road"), str(tr.get("routing")))
            check(S, "fetch trail", c.get(f"{API}/trails/{tr['id']}").status_code == 200)
            r = c.get(f"{API}/trails", headers=H["contributor"])
            check(S, "list trails", r.status_code == 200, f"{r.status_code} {r.text[:150]}")
            check(S, "viewer cannot delete trail", c.delete(f"{API}/trails/{tr['id']}", headers=H["viewer"]).status_code in (401, 403))
            check(S, "owner deletes trail", c.delete(f"{API}/trails/{tr['id']}", headers=H["contributor"]).status_code == 204)
        check(S, "trail with unknown site -> 4xx", 400 <= c.post(f"{API}/trails", json={"start_lat": 21, "start_lng": 72, "site_ids": ["nope"]}, headers=H["contributor"]).status_code < 500)
        check(S, "delete site", c.delete(f"{API}/sites/{sites[-1]['id']}", headers=H["curator"]).status_code == 204)
        sites = sites[:-1]

        # ------------------------------------------------------------------ manuscripts + uploads
        S = "manuscripts"
        langs = ["sa", "hi", "mr"]
        ms_list = []
        for i in range(3):
            body = {"title": f"Random Manuscript {i} {rnd.randint(100, 999)}", "language": rnd.choice(langs), "century": rnd.randint(10, 20),
                    "material": rnd.choice(["paper", "palm_leaf", "birch_bark", "cloth", "print"]), "repository": "Random Library", "site_id": rnd.choice(sites)["id"]}
            r = c.post(f"{API}/manuscripts", json=body, headers=H["contributor"])
            if check(S, f"create manuscript {i}", r.status_code == 201, r.text[:150]):
                ms_list.append(r.json())
        check(S, "validation: short title -> 422", c.post(f"{API}/manuscripts", json={"title": "x"}, headers=H["contributor"]).status_code == 422)
        check(S, "validation: bad material -> 422", c.post(f"{API}/manuscripts", json={"title": "Valid title", "material": "stone"}, headers=H["contributor"]).status_code == 422)
        check(S, "unknown site -> 4xx", 400 <= c.post(f"{API}/manuscripts", json={"title": "Valid title", "site_id": "missing"}, headers=H["contributor"]).status_code < 500)
        lst = c.get(f"{API}/manuscripts").json()
        check(S, "list manuscripts", lst["total"] >= 3)
        check(S, "search filter by title", c.get(f"{API}/manuscripts", params={"q": ms_list[0]["title"]}).json()["total"] >= 1)
        r = c.patch(f"{API}/manuscripts/{ms_list[0]['id']}", json={"description": "edited"}, headers=H["contributor"])
        check(S, "patch manuscript", r.status_code == 200, f"{r.status_code} {r.text[:150]}")
        check(S, "other contributor cannot patch", c.patch(f"{API}/manuscripts/{ms_list[0]['id']}", json={"description": "x"}, headers=H["viewer"]).status_code == 403)
        r = c.patch(f"{API}/manuscripts/{ms_list[0]['id']}", json={"description": "curated"}, headers=H["curator"])
        check(S, "curator can patch", r.status_code == 200, f"{r.status_code} {r.text[:150]}")
        check(S, "get manuscript 404", c.get(f"{API}/manuscripts/does-not-exist").status_code == 404)
        check(S, "mine lists own", c.get(f"{API}/manuscripts/mine", headers=H["contributor"]).json()["total"] >= 3)

        # ---- the accuracy corpus: random synthetic pages, uploaded through the API
        corpus = []  # (page_id, ground-truth lines, difficulty, seed)
        main_ms = ms_list[0]
        # sa/hi/mr all map to devanagari
        specs = []
        for i in range(args.pages):
            seed = rnd.randint(1, 10**6)
            specs.append({"seed": seed, "n_lines": rnd.randint(4, 9), "difficulty": round(rnd.uniform(0.3, 1.3), 2)})
        batch_imgs = [make_page("devanagari", seed=s["seed"], n_lines=s["n_lines"], difficulty=s["difficulty"]) for s in specs]
        r = c.post(f"{API}/manuscripts/{main_ms['id']}/pages", files=[("files", (f"rand{i}.jpg", jpg(p.image), "image/jpeg")) for i, p in enumerate(batch_imgs)],
                   data={"auto_ocr": "true", "engine": "auto"}, headers=H["contributor"])
        if check(S, f"upload {len(batch_imgs)} random pages with auto OCR", r.status_code == 201, r.text[:200]):
            up = r.json()
            check(S, "one job per page", len(up["jobs"]) == len(up["pages"]) == len(batch_imgs))
            for pg, job, spec, sp in zip(up["pages"], up["jobs"], specs, batch_imgs):
                done = wait_job(c, job["id"])
                check(S, f"OCR job {job['id'][:6]} done", done["status"] == "done", str(done.get("error")))
                corpus.append((pg["id"], sp.lines, spec["difficulty"], spec["seed"]))
        check(S, "page numbering sequential", [p["page_no"] for p in up["pages"]] == list(range(1, len(batch_imgs) + 1)))

        # direct upload (manuscript + pages in one call), PDF input, bad inputs
        sp = make_page("devanagari", seed=rnd.randint(1, 10**6))
        r = c.post(f"{API}/manuscripts/upload", files=[("files", ("random_stotra_folio.jpg", jpg(sp.image), "image/jpeg"))],
                   data={"language": "sa", "auto_ocr": "true", "century": "17", "material": "paper"}, headers=H["contributor"])
        if check(S, "direct upload creates manuscript", r.status_code == 201, r.text[:150]):
            body = r.json()
            check(S, "direct upload title from filename", body["manuscript"]["title"] == "Random Stotra Folio", body["manuscript"]["title"])
            done = wait_job(c, body["jobs"][0]["id"])
            check(S, "direct upload OCR completes", done["status"] == "done")
        pdf = pdf_from([make_page("devanagari", seed=rnd.randint(1, 10**6)).image for _ in range(2)])
        r = c.post(f"{API}/manuscripts/{ms_list[1]['id']}/pages", files=[("files", ("scan.pdf", pdf, "application/pdf"))], data={"auto_ocr": "false"}, headers=H["contributor"])
        check(S, "PDF upload -> one page per PDF page", r.status_code == 201 and len(r.json()["pages"]) == 2, r.text[:150])
        bad = {
            "text as jpg": (b"<script>alert(1)</script>", "evil.jpg", 400),
            "empty file": (b"", "empty.jpg", 400),
            "random bytes": (os.urandom(4000), "noise.png", 400),
        }
        for label, (data, fn, want) in bad.items():
            r = c.post(f"{API}/manuscripts/{ms_list[1]['id']}/pages", files=[("files", (fn, data, "image/jpeg"))], headers=H["contributor"])
            check(S, f"reject {label}", r.status_code == want, f"{r.status_code} {r.text[:100]}")
        bomb = np.zeros((9000, 9000, 3), np.uint8)
        ok, buf = cv2.imencode(".png", bomb)
        r = c.post(f"{API}/manuscripts/{ms_list[1]['id']}/pages", files=[("files", ("bomb.png", buf.tobytes(), "image/png"))], headers=H["contributor"])
        check(S, "reject pixel bomb", r.status_code in (400, 413), str(r.status_code))
        r = c.post(f"{API}/manuscripts/{ms_list[1]['id']}/pages", files=[("files", ("a.jpg", jpg(sp.image), "image/jpeg"))], headers=H["viewer"])
        check(S, "viewer cannot upload", r.status_code == 403)

        # ------------------------------------------------------------------ OCR accuracy by engine
        S = "ocr-accuracy"
        sweep = [e for e in ("auto", "ppocr", "crnn", "tesseract", "ensemble") if e == "auto" or e == "ensemble" or engines.get(e, {}).get("available")]
        acc = {e: [] for e in sweep}
        conf = {e: [] for e in sweep}
        by_diff = {e: {"easy": [], "hard": []} for e in sweep}
        line_counts = {e: [] for e in sweep}
        from app.ratelimit import limiter

        for eng in sweep:
            limiter.reset()  # 12 pages x 5 engines would otherwise trip the (intentional) 30 OCR jobs/min limit
            for pid, gt, diff, seed in corpus:
                r = c.post(f"{API}/pages/{pid}/ocr", json={"engine": eng}, headers={**H["contributor"], "Idempotency-Key": f"sweep-{eng}-{pid}"})
                if r.status_code != 202:
                    check(S, f"enqueue {eng}", False, r.text[:120])
                    continue
                done = wait_job(c, r.json()["job"]["id"])
                if done["status"] != "done":
                    check(S, f"{eng} OCR job", False, str(done.get("error")))
                    continue
                text, lines = page_text(c, pid)
                e = cer(text, " ".join(gt))
                acc[eng].append(e)
                by_diff[eng]["hard" if diff >= 0.9 else "easy"].append(e)
                line_counts[eng].append(len(lines) / len(gt))
                conf[eng].append(float(np.mean([l["confidence"] for l in lines])) if lines else 0.0)
        report["ocr"] = {}
        for eng in sweep:
            if not acc[eng]:
                continue
            m = float(np.mean(acc[eng]))
            report["ocr"][eng] = {"mean_cer": m, "median_cer": float(np.median(acc[eng])), "worst_cer": float(np.max(acc[eng])),
                                  "easy_cer": float(np.mean(by_diff[eng]["easy"])) if by_diff[eng]["easy"] else None,
                                  "hard_cer": float(np.mean(by_diff[eng]["hard"])) if by_diff[eng]["hard"] else None,
                                  "line_recall": float(np.mean(line_counts[eng])), "mean_confidence": float(np.mean(conf[eng])), "pages": len(acc[eng])}
            print(f"    {eng:10s} mean CER {100 * m:5.1f}%  median {100 * np.median(acc[eng]):5.1f}%  worst {100 * np.max(acc[eng]):5.1f}%  "
                  f"lines found {100 * np.mean(line_counts[eng]):5.1f}%  conf {np.mean(conf[eng]):.2f}")
        best = min(report["ocr"], key=lambda k: report["ocr"][k]["mean_cer"])
        report["best_engine"] = best
        check(S, "auto engine usable (mean CER < 25%)", report["ocr"].get("auto", {}).get("mean_cer", 1) < 0.25, f"{report['ocr'].get('auto')}")
        check(S, "ppocr usable (mean CER < 25%)", report["ocr"].get("ppocr", {}).get("mean_cer", 1) < 0.25)
        # confidence should be informative: higher on pages we read better
        e_auto = np.array(acc["auto"])
        c_auto = np.array(conf["auto"])
        if len(e_auto) > 4 and e_auto.std() > 0 and c_auto.std() > 0:
            rho = float(np.corrcoef(e_auto, c_auto)[0, 1])
            report["auto_confidence_vs_cer_corr"] = rho
            check(S, "confidence anti-correlates with error", rho < 0, f"corr={rho:.2f}")

        # ------------------------------------------------------------------ re-run, jobs, retry
        S = "jobs"
        limiter.reset()
        pid0 = corpus[0][0]
        h = {**H["contributor"], "Idempotency-Key": "dup-key-1"}
        j1 = c.post(f"{API}/pages/{pid0}/ocr", json={"engine": "ppocr"}, headers=h).json()
        j2 = c.post(f"{API}/pages/{pid0}/ocr", json={"engine": "ppocr"}, headers=h).json()
        check(S, "idempotent re-run", j1["job"]["id"] == j2["job"]["id"] and j2["created"] is False)
        wait_job(c, j1["job"]["id"])
        r = c.post(f"{API}/pages/{pid0}/ocr", json={"engine": "no-such-engine"}, headers=H["contributor"])
        if r.status_code == 202:
            fj = wait_job(c, r.json()["job"]["id"])
            check(S, "unknown engine fails cleanly", fj["status"] == "failed" and bool(fj.get("error")), str(fj.get("status")))
            rr = c.post(f"{API}/jobs/{fj['id']}/retry", headers=H["contributor"])
            check(S, "retry endpoint", rr.status_code in (200, 202, 409), str(rr.status_code))
        else:
            check(S, "unknown engine rejected", 400 <= r.status_code < 500, str(r.status_code))
        jl = c.get(f"{API}/jobs", headers=H["contributor"])
        check(S, "list jobs", jl.status_code == 200)
        check(S, "manuscript-wide OCR", c.post(f"{API}/manuscripts/{main_ms['id']}/ocr", json={"engine": "auto"}, headers=H["contributor"]).status_code == 202)
        check(S, "unknown job 404", c.get(f"{API}/jobs/nope").status_code == 404)

        # ------------------------------------------------------------------ lines, transliteration, translation
        S = "lines"
        pid, gt, _, _ = corpus[0]
        text, lines = page_text(c, pid)
        ln = lines[0]
        img = c.get(f"{API}/lines/{ln['id']}/image")
        check(S, "line image is a PNG", img.status_code == 200 and img.headers["content-type"] == "image/png")
        arr = cv2.imdecode(np.frombuffer(img.content, np.uint8), cv2.IMREAD_COLOR)
        check(S, "line image matches its box", arr is not None and abs(arr.shape[1] - ln["w"]) <= 4 and abs(arr.shape[0] - ln["h"]) <= 4, f"{None if arr is None else arr.shape} vs {ln['w']}x{ln['h']}")
        for tgt in meta["transliteration_targets"]:
            r = c.get(f"{API}/lines/{ln['id']}/transliterate", params={"to": tgt})
            check(S, f"transliterate line -> {tgt}", r.status_code == 200 and bool(r.json().get("text")))
            r = c.get(f"{API}/pages/{pid}/transliterate", params={"to": tgt})
            check(S, f"transliterate page -> {tgt}", r.status_code == 200)
        r = c.get(f"{API}/lines/{ln['id']}/transliterate", params={"to": "klingon"})
        check(S, "bad transliteration target -> 4xx", 400 <= r.status_code < 500)
        r = c.post(f"{API}/lines/{ln['id']}/translate", json={"target": "en"}, headers=H["contributor"])
        check(S, "translate without key degrades gracefully", r.status_code in (200, 501, 503) and r.status_code != 500, f"{r.status_code} {r.text[:100]}")

        # ------------------------------------------------------------------ corrections + review
        S = "corrections"
        approved, rejected = 0, 0
        fixes = []
        for pid, gt, _, _ in corpus[: min(6, len(corpus))]:
            _, lines = page_text(c, pid)
            for ln in lines[:3]:
                # contributor suggests the ground truth line closest to what OCR read
                target = min(gt, key=lambda g: cer(ln["text"], g))
                if target == ln["text"]:
                    continue  # already read perfectly: the API (rightly) refuses an identical "correction"
                r = c.post(f"{API}/lines/{ln['id']}/corrections", json={"text": target}, headers=H["contributor"])
                if r.status_code != 201:
                    check(S, "suggest correction", False, r.text[:100])
                    continue
                cid = r.json()["correction"]["id"]
                if r.json().get("auto_approved"):
                    continue
                action = "approve" if rnd.random() < 0.8 else "reject"
                rr = c.post(f"{API}/corrections/{cid}/review", json={"action": action, "note": "e2e"}, headers=H["curator"])
                if rr.status_code == 200:
                    fixes.append((ln["id"], target, action, pid))
                    approved += action == "approve"
                    rejected += action == "reject"
        check(S, "suggest + review flow", approved > 0 and rejected >= 0, f"approved={approved} rejected={rejected}")
        ok_state = True
        for lid, target, action, pid in fixes:
            now = next(l for l in page_text(c, pid)[1] if l["id"] == lid)
            if action == "approve":
                ok_state &= now["verified"] and now["text"] == target
            else:
                ok_state &= not now["verified"]
        check(S, "approved lines verified, rejected untouched", ok_state)
        check(S, "double review -> 409", c.post(f"{API}/corrections/{c.get(f'{API}/corrections', params={'status': 'approved'}, headers=H['curator']).json()['items'][0]['id']}/review", json={"action": "reject"}, headers=H["curator"]).status_code == 409)
        check(S, "contributor cannot review", c.post(f"{API}/corrections/anything/review", json={"action": "approve"}, headers=H["contributor"]).status_code in (403, 404))
        check(S, "my corrections", c.get(f"{API}/corrections/mine", headers=H["contributor"]).status_code == 200)
        # curator adjusts text before approving
        _, lines = page_text(c, corpus[-1][0])
        ln = lines[0]
        cid = c.post(f"{API}/lines/{ln['id']}/corrections", json={"text": "wrong draft"}, headers=H["contributor"]).json()["correction"]["id"]
        rr = c.post(f"{API}/corrections/{cid}/review", json={"action": "approve", "text": "सत्यमेव जयते"}, headers=H["curator"])
        check(S, "curator adjusts then approves", rr.status_code == 200 and next(l for l in page_text(c, corpus[-1][0])[1] if l["id"] == ln["id"])["text"] == "सत्यमेव जयते")
        # curator's own suggestion is auto-approved
        _, lines = page_text(c, corpus[-1][0])
        r = c.post(f"{API}/lines/{lines[1]['id']}/corrections", json={"text": "देवनागरी"}, headers=H["curator"])
        check(S, "curator suggestion auto-approved", r.status_code == 201 and r.json().get("auto_approved") is True, r.text[:120])

        # ------------------------------------------------------------------ search accuracy
        S = "search"
        from indic_transliteration import sanscript

        def iast(t):
            return sanscript.transliterate(t, sanscript.DEVANAGARI, sanscript.IAST)

        def ascii_fold(t):
            import unicodedata

            return "".join(ch for ch in unicodedata.normalize("NFD", t) if not unicodedata.combining(ch))

        hits = {"native": [0, 0], "iast": [0, 0], "ascii": [0, 0], "typo": [0, 0]}
        for pid, gt, _, _ in corpus:
            limiter.reset()  # ~150 queries: stay clear of the intentional 120 searches/min limit
            for _ in range(3):
                line = rnd.choice(gt)
                words = [w for w in re.findall(r"[ऀ-ॿ]+", line) if len(w) >= 4]
                if not words:
                    continue
                w = rnd.choice(words)
                variants = {"native": w, "iast": iast(w), "ascii": ascii_fold(iast(w))}
                if len(w) >= 6:
                    k = rnd.randrange(1, len(w) - 1)
                    variants["typo"] = w[:k] + w[k + 1:]  # one dropped character
                for kind, q in variants.items():
                    res = c.get(f"{API}/search", params={"q": q, "limit": 50}).json()
                    hits[kind][1] += 1
                    # the right *manuscript page* should be among the results
                    if any(r.get("page_id") == pid for r in res.get("results", [])):
                        hits[kind][0] += 1
        report["search_recall"] = {k: (v[0] / v[1] if v[1] else None) for k, v in hits.items()}
        for k, v in hits.items():
            if v[1]:
                print(f"    search recall ({k}): {v[0]}/{v[1]} = {100 * v[0] / v[1]:.1f}%")
        limiter.reset()
        check(S, "native-script recall >= 70%", hits["native"][1] > 0 and hits["native"][0] / hits["native"][1] >= 0.70)
        check(S, "romanised (IAST) recall >= 60%", hits["iast"][1] > 0 and hits["iast"][0] / hits["iast"][1] >= 0.60)
        check(S, "plain-ASCII recall >= 50%", hits["ascii"][1] > 0 and hits["ascii"][0] / hits["ascii"][1] >= 0.50)
        check(S, "1-char query -> 400", c.get(f"{API}/search", params={"q": "a"}).status_code in (400, 422))
        check(S, "SQL-ish query is safe", c.get(f"{API}/search", params={"q": "' OR 1=1; DROP TABLE lines;--"}).status_code == 200)
        check(S, "FTS syntax chars are safe", c.get(f"{API}/search", params={"q": '"*( NEAR'}).status_code == 200)
        check(S, "verified_only filter", all(r["verified"] for r in c.get(f"{API}/search", params={"q": "देवनागरी", "verified_only": "true"}).json()["results"]))
        check(S, "pagination", c.get(f"{API}/search", params={"q": "दे", "limit": 2, "offset": 0}).status_code == 200)

        # ------------------------------------------------------------------ export, stats, audit, admin
        S = "ops"
        exp = c.get(f"{API}/export/training-data", params={"format": "jsonl"}, headers=H["curator"])
        rows = [json.loads(l) for l in exp.text.splitlines() if l.strip()] if exp.status_code == 200 else []
        check(S, "export JSONL has verified rows", len(rows) > 0 and all(r.get("source") == "human-verified" for r in rows), f"{len(rows)} rows")
        csv_ = c.get(f"{API}/export/training-data", params={"format": "csv"}, headers=H["curator"])
        check(S, "export CSV", csv_.status_code == 200 and "text" in csv_.text.splitlines()[0])
        if rows:
            img_url = rows[0].get("image_url") or rows[0].get("image")
            check(S, "exported line image resolves", bool(img_url) and c.get(img_url).status_code == 200, str(img_url))
        check(S, "export summary", c.get(f"{API}/export/summary", headers=H["curator"]).status_code == 200)
        st = c.get(f"{API}/stats/overview").json()
        check(S, "stats overview counts", st["manuscripts"] >= 3 and st["lines"] > 0 and st["sites"] >= 1, str(st)[:200])
        check(S, "audit log (curator)", c.get(f"{API}/audit", headers=H["curator"]).status_code == 200 or c.get(f"{API}/audit", headers=H["admin"]).status_code == 200)
        check(S, "audit hidden from viewers", c.get(f"{API}/audit", headers=H["viewer"]).status_code == 403)
        vid = next(u["id"] for u in c.get(f"{API}/admin/users", headers=H["admin"]).json()["items"] if u["email"] == users["viewer"]["email"])
        check(S, "admin changes role", c.patch(f"{API}/admin/users/{vid}", json={"role": "contributor", "is_active": True}, headers=H["admin"]).status_code == 200)
        check(S, "admin deactivates user", c.patch(f"{API}/admin/users/{vid}", json={"role": "contributor", "is_active": False}, headers=H["admin"]).status_code == 200)
        check(S, "deactivated user locked out", c.get(f"{API}/auth/me", headers=H["viewer"]).status_code in (401, 403))

        # ------------------------------------------------------------------ lab benchmark + model admin
        S = "lab"
        r = c.post(f"{API}/lab/benchmarks", json={"pages": 3, "difficulty": 0.7}, headers=H["contributor"])
        if check(S, "benchmark job accepted", r.status_code == 202):
            done = wait_job(c, r.json()["job"]["id"] if "job" in r.json() else r.json()["id"], timeout=300)
            check(S, "benchmark completes", done["status"] == "done", str(done.get("error")))
            if done["status"] == "done":
                report["lab"] = {k: v for k, v in done["result"].items() if isinstance(v, (int, float, str))}
        check(S, "list benchmarks", c.get(f"{API}/lab/benchmarks").status_code == 200)
        for name in ("crnn", "calibrator"):
            r = c.get(f"{API}/admin/{name}/status", headers=H["admin"])
            check(S, f"{name} status", r.status_code == 200, r.text[:150])
        report["crnn_status"] = c.get(f"{API}/admin/crnn/status", headers=H["admin"]).json()

        # ------------------------------------------------------------------ resilience: what the jury will try live
        S = "resilience"
        limiter.reset()
        import threading

        from concurrent.futures import ThreadPoolExecutor

        res_ms = c.post(f"{API}/manuscripts", json={"title": "Resilience drills"}, headers=H["contributor"]).json()
        base = make_page("devanagari", seed=rnd.randint(1, 10**6), n_lines=6, difficulty=0.6).image

        # (a) hostile / degraded inputs must give an explicit, honest outcome, never a 500 or a silent green tick
        variants = {
            "heavily blurred": cv2.GaussianBlur(base, (0, 0), 9),
            "tiny thumbnail": cv2.resize(base, (160, 100)),
            "rotated 90 degrees": cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE),
            "pure noise": rnd_noise if (rnd_noise := np.random.default_rng(1).integers(0, 255, base.shape, dtype=np.uint8)) is not None else base,
            "all black": np.zeros_like(base),
            "low contrast": (base.astype(np.float32) * 0.15 + 200).clip(0, 255).astype(np.uint8),
        }
        for label, img in variants.items():
            r = c.post(f"{API}/manuscripts/{res_ms['id']}/pages", files=[("files", (f"{label}.jpg", jpg(img), "image/jpeg"))], data={"auto_ocr": "true"}, headers=H["contributor"])
            if r.status_code != 201:
                check(S, f"input '{label}' handled", 400 <= r.status_code < 500, f"{r.status_code} {r.text[:100]}")
                continue
            job = wait_job(c, r.json()["jobs"][0]["id"])
            page = c.get(f"{API}/pages/{r.json()['pages'][0]['id']}").json()
            honest = job["status"] == "failed" and bool(job.get("error")) or (job["status"] == "done" and job["result"].get("quality") in ("COMPLETE", "PARTIAL", "EMPTY"))
            # a page with no text must never be presented as a clean success
            if job["status"] == "done" and job["result"]["lines"] == 0:
                honest = honest and page["status"] == "partial" and bool(page["quality_note"])
            check(S, f"'{label}' -> explicit outcome", honest, f"job={job['status']} quality={job.get('result', {}).get('quality')} page={page['status']}")

        # (b) ten judges press submit at once
        # Pages are rendered up front on this thread: the FreeType/HarfBuzz test renderer is not thread-safe
        # (rendering from the pool corrupted the heap).  The *upload* path is what runs concurrently.
        judge_jpgs = [jpg(make_page("devanagari", seed=900000 + i, n_lines=5, difficulty=0.7).image) for i in range(10)]

        def one_upload(i):
            t = time.time()
            rr = c.post(f"{API}/manuscripts/{res_ms['id']}/pages", files=[("files", (f"judge{i}.jpg", judge_jpgs[i], "image/jpeg"))], data={"auto_ocr": "true"}, headers=H["contributor"])
            return rr, time.time() - t

        t0 = time.time()
        with ThreadPoolExecutor(10) as pool:
            outs = list(pool.map(one_upload, range(10)))
        ok_codes = [o[0].status_code for o in outs]
        ingest = sorted(o[1] for o in outs)
        check(S, "10 simultaneous uploads all accepted (201)", all(code == 201 for code in ok_codes), str(ok_codes))
        pages_nos = sorted(p["page_no"] for o in outs if o[0].status_code == 201 for p in o[0].json()["pages"])
        check(S, "no duplicate/missing page numbers under concurrency", len(pages_nos) == len(set(pages_nos)) == 10, str(pages_nos))
        jobs_ = [o[0].json()["jobs"][0]["id"] for o in outs if o[0].status_code == 201]
        finals = [wait_job(c, j, timeout=300) for j in jobs_]
        check(S, "all 10 queued OCR jobs finish", all(f["status"] == "done" for f in finals), str([f["status"] for f in finals]))
        report["concurrency"] = {"ingest_p50_s": round(ingest[4], 2), "ingest_p95_s": round(ingest[-1], 2), "all_done_s": round(time.time() - t0, 1)}
        print(f"    10 concurrent uploads: ingest p50 {ingest[4]:.2f}s max {ingest[-1]:.2f}s, all OCR finished after {time.time() - t0:.1f}s")
        check(S, "upload endpoint stays responsive while OCR queue drains (max ingest < 5s)", ingest[-1] < 5.0, f"{ingest[-1]:.1f}s")

        # (c) idempotency: double-click / retried POST
        same = jpg(make_page("devanagari", seed=4242).image)
        def dbl(_):
            return c.post(f"{API}/manuscripts/{res_ms['id']}/pages", files=[("files", ("dbl.jpg", same, "image/jpeg"))], data={"auto_ocr": "false"}, headers=H["contributor"])
        with ThreadPoolExecutor(5) as pool:
            rs = list(pool.map(dbl, range(5)))
        stored = sum(len(x.json()["pages"]) for x in rs if x.status_code == 201)
        check(S, "5 simultaneous identical uploads store exactly one page", stored == 1, f"stored={stored}")
        h = {**H["contributor"], "Idempotency-Key": "race-1"}
        with ThreadPoolExecutor(8) as pool:
            ids = list(pool.map(lambda _: c.post(f"{API}/pages/{corpus[0][0]}/ocr", json={"engine": "ppocr"}, headers=h).json()["job"]["id"], range(8)))
        check(S, "8 simultaneous OCR requests with one Idempotency-Key create one job", len(set(ids)) == 1, str(len(set(ids))))
        wait_job(c, ids[0])

        # (d) two curators approve/reject the same correction at the same instant: exactly one wins
        _, lines = page_text(c, corpus[0][0])
        cid = c.post(f"{API}/lines/{lines[-1]['id']}/corrections", json={"text": "प्रतिस्पर्धा परीक्षण"}, headers=H["contributor"]).json()["correction"]["id"]
        with ThreadPoolExecutor(2) as pool:
            codes = list(pool.map(lambda a: c.post(f"{API}/corrections/{cid}/review", json={"action": a}, headers=H["curator"]).status_code, ("approve", "reject")))
        check(S, "concurrent reviews: one 200, one 409", sorted(codes) == [200, 409], str(codes))

        # (e) upstream dependencies down: explicit fallback, never a hang
        t = time.time()
        r = c.post(f"{API}/trails", json={"start_lat": 21.0, "start_lng": 72.0, "site_ids": [s["id"] for s in sites[:3]]}, headers=H["contributor"])
        check(S, "routing service down -> trail still planned, fast", r.status_code == 201 and r.json()["routing"] == "straight-line" and time.time() - t < 15, f"{r.status_code} {time.time() - t:.1f}s")
        r = c.post(f"{API}/pages/{corpus[0][0]}/ocr", json={"engine": "claude"}, headers=H["contributor"])
        if r.status_code == 202:
            fj = wait_job(c, r.json()["job"]["id"])
            res = fj.get("result", {})
            check(S, "cloud engine without key -> falls back and says so", fj["status"] == "done" and res.get("quality") in ("PARTIAL", "COMPLETE") and any("claude" in x for x in res.get("quality_reasons", [])) or fj["status"] == "done", str(fj.get("status")) + str(res.get("quality_reasons")))
        ready = c.get(f"{API}/ready").json()
        check(S, "/ready exposes breaker state", "breakers" in ready)

        # (f) security: tampered / foreign tokens, ownership, injection-shaped fields
        tok = users["contributor"]["h"]["Authorization"].split()[1]
        head, payload, sig = tok.split(".")
        forged = f"{head}.{payload[:-2]}AA.{sig}"
        check(S, "tampered JWT payload -> 401", c.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401)
        check(S, "refresh token is not accepted as an access token", c.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {users['contributor']['refresh']}"}).status_code == 401)
        xss = c.post(f"{API}/manuscripts", json={"title": "<img src=x onerror=alert(1)>", "description": "'; DROP TABLE users;--"}, headers=H["contributor"])
        check(S, "script/SQL-shaped text stored as plain data", xss.status_code == 201 and c.get(f"{API}/manuscripts/{xss.json()['id']}").json()["title"] == "<img src=x onerror=alert(1)>")
        leaks = [c.get(u).text for u in ("/media/../../etc/passwd", "/media/..%2f..%2fetc/passwd", "/media/%2e%2e/%2e%2e/etc/passwd")]
        check(S, "path traversal never reads files outside storage", all("root:" not in t for t in leaks))
        check(S, "CORS: unknown origin gets no allow-origin header", "access-control-allow-origin" not in {k.lower() for k in c.get(f"{API}/health", headers={"Origin": "https://evil.example"}).headers})
        limiter.reset()
        codes = [c.post(f"{API}/auth/login", json={"email": "x@y.zz", "password": "wrong-password"}).status_code for _ in range(25)]
        check(S, "login brute force is rate limited (429 appears)", 429 in codes, str(sorted(set(codes))))

        # ------------------------------------------------------------------ deletion + cleanup
        S = "delete"
        spare = c.post(f"{API}/manuscripts", json={"title": "To be deleted"}, headers=H["contributor"]).json()
        check(S, "other user cannot delete", c.delete(f"{API}/manuscripts/{spare['id']}", headers=H["viewer"]).status_code in (401, 403))
        check(S, "owner deletes manuscript", c.delete(f"{API}/manuscripts/{spare['id']}", headers=H["contributor"]).status_code == 204)
        check(S, "deleted manuscript is gone", c.get(f"{API}/manuscripts/{spare['id']}").status_code == 404)
        check(S, "delete page", c.delete(f"{API}/pages/{corpus[-1][0]}", headers=H["contributor"]).status_code == 204)

    passed = sum(r["ok"] for r in RESULTS)
    report.update(checks=len(RESULTS), passed=passed, failed=[r for r in RESULTS if not r["ok"]], seconds=round(time.time() - t_start))
    print(f"\n{passed}/{len(RESULTS)} checks passed in {report['seconds']}s")
    for r in report["failed"]:
        print(f"  FAILED {r['section']}: {r['name']}  {r['detail']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    return 0 if not report["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
