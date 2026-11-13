# LipiSetu — Grand Finale playbook (HackCult '26, Track 2: Culture & Digital Heritage)

Everything here describes what the code in this repo actually does. All numbers below were measured on the demo laptop (Apple M1, CPU inference) on 13 Nov 2026; re-run the commands in section 8 before the finale and say what *your* screen shows. Never quote a number you have not seen.

---

## 0. The 20-second answer to "what is this?"

> "Old Indian manuscripts can't be searched, so they can't be read, checked or cited. LipiSetu turns a photo of a folio into verified, searchable text, then ties each manuscript to the heritage site it belongs to and plans the trip to see it. Every line shows its own confidence, so people spend their time on the lines the machine is unsure about."

Three differentiators (say them in this order, every time):
1. **Honest OCR pipeline** — our own page clean-up and line finding, an engine chain with fallback, and an explicit COMPLETE / PARTIAL / EMPTY status. It never reports success for a page it did not read.
2. **Human-in-the-loop** — corrections, curator review with a character diff, verified lines become a training dataset.
3. **Heritage link** — cross-script search, a map of 29 real sites, and a trail planner with road routing.

---

## 1. Numbers cheat-sheet

| Claim | Number | Status |
|---|---|---|
| Off-the-shelf PP-OCRv5 on synthetic aged pages | 3.4% CER | measured (README / OCR Lab) |
| Our full pipeline on the same pages | 3.4% CER (2.8× better than same recogniser without our clean-up: 9.7%) | measured |
| Real folios, characters read | 3,467 vs 1,772 stock (2.0×); Advaita Viveka folio 71 → 741 | measured |
| Skew recovery | within 0.1° | measured |
| Our CRNN, held-out *real printed Sanskrit* lines (705 lines from pages never seen in training) | **CER 14.4%**, 11% of lines exactly right | measured (`data/crnn_metrics.json`) |
| Our CRNN, synthetic Hindi validation lines | CER 0.7% | measured |
| Training data | 4,374 verse crops → 8,745 line images (real) + 30,000 synthetic Hindi lines | built by `scripts/build_ocr_dataset.py` |
| Engine head-to-head, real Sanskrit test lines (400, original resolution) | **ours 13.1% · PP-OCRv5 27.7% · ensemble ("auto") 13.1%** (2.1× fewer errors than the stock engine) | measured (`scripts/eval_engines.py`) |
| Engine head-to-head, synthetic Hindi lines | PP-OCRv5 5.4% · ours 9.9% · ensemble 7.6% | measured — see "honest trade-off" below |
| End-to-end random pages through the live API (12 pages × 2 seeds, aged-parchment renderer) | PP-OCRv5 3.9–4.3% · ensemble/auto 6.3–6.7% page CER; every line found (100%) | measured (`scripts/e2e_random.py`) |
| Confidence is informative | correlation of page confidence vs page CER: −0.93 and −0.83 | measured |
| Search recall on random OCR'd pages (native / IAST / plain-ASCII / one dropped letter) | 89–97% / 89–97% / 89–97% / 81% | measured, 2 seeds |
| Randomised end-to-end test | **165 / 165 checks pass** (2 different seeds, ~51 s each) | measured |
| 10 simultaneous uploads | all accepted (201), page numbers unique, ingest 0.1 s each, all 10 OCR jobs finished in ~2.3 s | measured |
| Confidence calibrator (picks the better engine per line) | held-out AUC 0.81, accuracy 72% (1,511 lines) | measured |
| Automated unit/API tests | 30 passing |
| Upload limits | 25 MB/file, 50 files/request, 60 MP pixel-bomb guard, PDF supported |
| Tokens | access 8 h, refresh 7 days, Argon2id passwords |
| Job queue | 5-min lease, 3 attempts, 5 s → 10 s → 20 s backoff |

**Honest trade-off to volunteer (it shows judgement):** our CRNN wins on *real scanned Sanskrit* (13.1% vs 27.7%) but PP-OCRv5 wins on clean *synthetic* renders (3.9–4.3% vs 6.3–6.7%). Neither is trained on handwriting. That is why "auto" runs an ensemble that picks per line using a small calibrator (it never did worse than the better engine on the real test set), and why every line still carries its confidence for human review.

**Do not claim:** accuracy on handwritten manuscripts (we have no labelled handwritten test set — say so), "Vision Transformer" (our recognisers are a CNN+biLSTM+CTC model we trained and PP-OCRv5's SVTR; SVTR is transformer-style, ours is not), Tesseract results (not installed on the demo machine), Docker deployment (files exist, not tested here), 3D digital twins (not built).

---

## 2. Five-minute stage pitch (300 s) + 3-minute Q&A

| Time | Act | What you do / say |
|---|---|---|
| 0:00–0:40 | **Problem** | "A folio photographed in a library is a picture, not text. Stock OCR loses whole lines on pages with no word spaces — on the Advaita Viveka folio it read 71 characters where 741 were there. What isn't read can't be corrected, searched or cited." |
| 0:40–2:10 | **Live demo, real input** | Hand-picked *new* photo (not in the repo). Upload → progress bar (queued → cleaning → finding lines → reading) → page viewer with green/amber/red line boxes → click an amber line → IAST/Gujarati toggle → suggest a fix → (curator) approve with the diff → search `dharma` finds धर्म. |
| 2:10–3:15 | **Architecture** (slide 4) | "Browser → FastAPI → job queue in the database → worker: clean page → find lines → engine chain with circuit breaker. If an engine is down we fall back; if nothing is readable the page says PARTIAL or EMPTY with the reason. Duplicate uploads are stored once; retries are idempotent." |
| 3:15–4:10 | **Evidence** (slide 6) | The measured table: CER, engine comparison, search recall, concurrency result. "We measure ourselves — OCR Lab runs on every demo." |
| 4:10–4:45 | **Heritage + scale** | Map → pick 3 sites → trail with road route and visit time. "Scaling is more worker processes; PostgreSQL + `pg_trgm` and S3 are a configuration change behind interfaces that already exist." |
| 4:45–5:00 | **Close** | Three differentiators, one sentence each. Stop at 4:50. |

**Slides (7 + appendix):** 1 Title + live URL/QR · 2 Problem with the 71 → 741 number · 3 Workflow (Upload → Read → Verify → Search/Visit) · 4 Architecture diagram (README mermaid) · 5 ER schema (README mermaid) · 6 Metrics scoreboard · 7 Roadmap (pilot with one library → regional → national; GPU worker, PostgreSQL). Appendix: A security/roles · B failure modes (engine breaker, PARTIAL status, lease/retry) · C repo map + tests.

---

## 3. Booth (10:45–1:15): set-up and choreography

Three windows, ready before the first judge:
1. **App** at `http://localhost:8000` (phone on the same network too) — logged in as Contributor, with a second tab as Curator.
2. **Terminal** with `tail -f` of the server log (JSON lines with request IDs) and `curl localhost:8000/api/v1/ready` ready — shows the engines, worker heartbeat and breaker state.
3. **Editor** open on: `backend/app/ocr/pipeline.py`, `backend/app/ocr/chain.py`, `backend/app/services/jobs.py`, `backend/app/api/manuscripts.py`, `backend/app/config.py`.

**60-second opener:** one sentence of problem → upload a photo the judge picks (their phone, or a printed page you hold up) → point at the log scrolling → show the colour-coded result → "want to try a bad one?"

**Let the judge break it.** Drills you have already run (`scripts/e2e_random.py`, section "resilience"):
- blurry / tiny / rotated / all-black / noise / very low contrast image → explicit outcome, never a 500, never a clean green tick for an unread page
- a text file renamed `.jpg`, an empty file, a 9000×9000 pixel bomb → rejected with a clear 400/413
- same file uploaded five times at once → stored once (`duplicates_skipped`)
- ten uploads at once → all accepted, page numbers unique, OCR drains in the background
- two curators review the same correction → one 200, one 409
- forged JWT, refresh token used as access token, `'; DROP TABLE`, `../../etc/passwd`, hostile CORS origin → all refused / treated as plain text
- 25 wrong logins → HTTP 429

---

## 4. Question bank with answers grounded in this repo

### A. "Is it a wrapper? What did you build?"
**Q: Is this just an API around somebody's OCR?**
No cloud model is in the default path. We wrote: page clean-up (`ocr/preprocess.py`: crop, CLAHE, bilateral+median denoise, Sauvola with Niblack fallback for faded ink, frame removal, projection-profile deskew), our own line segmenter (`ocr/segment.py`), the engine chain with circuit breaker and empty-result fallback (`ocr/chain.py`), post-processing (`ocr/postprocess.py`), the search (`services/search.py`), the job queue (`services/jobs.py`), the trail planner (`services/geo.py`) and **we trained our own CRNN** (`ocr/engines/crnn.py`, `ocr/train_crnn.py`). PP-OCRv5 is one engine in the chain, run locally via ONNX.

**Q: Why not use the stock detector?**
Measured: it misses whole lines on dense, space-less manuscript text (1,772 characters vs our 3,467 on the same folios). Missing text can never be corrected.

**Q: You said ViT OCR in the track description — where is it?**
Honest: we do not claim a Vision Transformer. PP-OCRv5's SVTR recogniser is transformer-style; our own model is a CNN+biLSTM+CTC (2.2 M parameters). We chose them for CPU inference at ~0.3 s/page. A ViT recogniser is the roadmap item once we have GPU time and labelled handwriting.

### B. Accuracy and evaluation
**Q: How do you know it is accurate?**
Character error rate = edit distance / reference length on NFC-normalised text, computed on data the model never saw: (1) OCR Lab — synthetic aged pages with known text, scored against two baselines; (2) held-out *source pages* of the real Sanskrit corpus (split by page so neighbouring verses never straddle train/test), scripted in `scripts/eval_engines.py`; (3) end-to-end randomised pages through the live API (`scripts/e2e_random.py`) with page-level CER per engine.

**Q: What is the accuracy on real handwritten manuscripts?**
We do not have a labelled handwritten test set, so we do not quote one. What we measured on real folios is *coverage* (characters read vs stock) and confidence; every line carries its confidence and low ones are routed to humans. Real folios average ~70% line confidence — which is why review exists.

**Q: The training data is printed. Doesn't that limit you?**
Yes — and we say so. The Sanskrit corpus is a scanned printed book (and its labels contain some noise, e.g. a verse where the label and image differ), which puts a floor under measurable CER. It teaches Devanagari conjuncts and Sanskrit vocabulary; handwritten accuracy improves as curators verify lines — the export (`/export/training-data`) is exactly that fine-tuning set.

**Q: How do you stop leakage?**
Split by source page id (hash → 85/7.5/7.5), vocabulary from the training split only, model selection on validation only, test run once at the end (`train_crnn.py` writes `data/crnn_metrics.json`).

**Q: Is the confidence meaningful?**
We test it: across random pages the correlation between a page's confidence and its CER is −0.93 and −0.83 (strongly negative: low confidence really means more errors).

**Q: Search quality?**
Trigram FTS5 over native text *and* romanised text, re-ranked by edit distance (RapidFuzz). Measured recall on random OCR'd pages: native 89–97%, IAST 89–97%, plain-ASCII 89–97%, one dropped letter 81% (two random seeds).

### C. Live resilience
**Q: Upstream is slow or down — what happens?**
OCR runs locally, so venue Wi-Fi does not touch the core flow. Optional/remote pieces degrade explicitly: OSRM down → straight-line trail with ×1.3 road factor, labelled `routing: straight-line`; no Anthropic key → translate says what is needed; engine failing 3× → circuit opens for 2 min (visible in `/ready` → `breakers`); engine returns no text → next engine tries; nothing readable → page status PARTIAL/EMPTY with the reason, never silent success.

**Q: What if 10 judges press upload together?**
Ingest is fast (decode, validate, store, enqueue) and independent of OCR. The first bottleneck is the single embedded OCR worker (CPU-bound, ~seconds per page) — jobs queue, progress shows per page. Measured with 10 simultaneous uploads: every request accepted, ingest 0.1 s each, all ten OCR jobs finished in about 2.3 s on this laptop. Scale-out: run more `python -m app.worker` processes (atomic conditional-UPDATE claim + lease), with PostgreSQL for multi-instance.

**Q: Idempotency / double click?**
Two layers: `Idempotency-Key` on OCR requests (same key → same job, `created: false`), and pixel-hash dedupe on uploads (same image stored once, `duplicates_skipped` returned). Both tested under concurrent requests.

**Q: What if a worker dies mid-job?**
Jobs are claimed with a 5-minute lease. If it expires another worker takes the job; failures retry with exponential backoff up to 3 attempts; permanent errors (bad page) fail fast with a message.

**Q: Silent failures?**
This was a deliberate design rule: job result carries `quality: COMPLETE|PARTIAL|EMPTY` + reasons; the UI shows a Partial badge with the reason. Triggers (all in `config.py`, one place): >50% of boxes rejected, mean confidence <50%, requested engine unavailable.

### D. Architecture, data, security
**Q: Why SQLite?** One file, WAL mode (readers don't block the writer), zero setup for judges, FTS5 trigram works for every Indic script without a stemmer. Access is through SQLAlchemy; the search layer has a portable fallback, so PostgreSQL + `pg_trgm` is a configuration change plus an index swap. Honest limit: single writer — that is why the path to scale is PostgreSQL.

**Q: Indexes?** FKs indexed (`pages.manuscript_id`, `lines.page_id`, `corrections.line_id`, status columns), composite `ix_jobs_claim(status, available_at)` for job claims, `ix_sites_latlng` for nearby search (bounding-box pre-filter, then exact haversine), FTS5 for text.

**Q: Security?** Argon2id; JWT access (8 h) and refresh (7 d) tokens, refresh tokens rejected as access tokens; roles enforced on every write on the server (viewer/contributor/curator/admin); per-IP sliding-window limits (auth, upload, OCR, search); uploads decoded not trusted by extension, format allow-list, re-encoded, pixel-bomb guard, storage paths confined to the root; security headers; SQL only via parameters; append-only audit log. Admin-only model endpoints; checkpoints loaded with `weights_only=True`.

**Q: Show the schema.** README ER diagram: USERS, MANUSCRIPTS, SITES, PAGES, JOBS, LINES, CORRECTIONS, TRAILS, AUDIT_LOG; `line_fts` beside LINES.

**Q: How many files to change a business rule?** One: `backend/app/config.py` (thresholds, limits, lease, engine gates) via environment variables. Demo: change `OCR_MIN_MEAN_CONFIDENCE` and restart.

**Q: Dependency pinning?** Every package in `backend/requirements.txt` is pinned to an exact version; `package-lock.json` for the frontend; OCR models ship with the pinned package, so there is no runtime download over venue Wi-Fi. (Training-only deps are in `requirements-train.txt`.)

### E. Code ownership and AI assistance
**Q: What did AI write?** Say it plainly: AI tools helped with scaffolding and some boilerplate; you can walk through the core (`pipeline.py`, `chain.py`, `jobs.py`, `search.py`, `geo.py`). Have two real debugging stories ready:
1. The old trainer would have trained on empty labels — the dataset ships labels in a separate CSV; found by inspecting the data before training.
2. A half-trained model returned empty text and the chain accepted it as success → fixed with an empty-result fallback, a quality gate on the checkpoint (`CRNN_MAX_VAL_CER`) and the COMPLETE/PARTIAL status.
Also: two curators reviewing one correction at the same instant both got 200 (found by our concurrency drill; fixed with an atomic pending→reviewing UPDATE, now one 200 and one 409); ROVER voting truncated candidates when an engine inserted characters (fixed, with a regression test); a shared-flag race in ensemble mode (fixed: per-call parameter).

**Q: Walk me through lines X–Y.** Know: `run_page_ocr` (preprocess → segment → recognise → filter by script → quality), `claim_next` (conditional UPDATE + lease), `plan_order` (nearest neighbour + 2-opt), `rover_vote` (align to strongest candidate, vote per column).

### F. Domain (Heritage track)
**Q: Degraded ink, no word spaces, archaic ligatures?** Contrast (CLAHE), edge-preserving denoise, Sauvola local thresholding with Niblack fallback for faded pages, frame removal, deskew, projection-profile line finding tuned for headline (shirorekha) scripts; confidence per line; human review.

**Q: Cultural/ethical side?** Provenance and licence are required fields per manuscript; images in the demo are Wikimedia Commons with credits in the README; curators, not the model, decide what becomes "verified".

**Q: Who uses it?** Libraries/archives (digitise and expose text), scholars (search across scripts), students and travellers (trails to the places where manuscripts live). Pilot: one repository (e.g. L. D. Institute of Indology, Ahmedabad — already a site on the map).

**Q: 3D digital twins / mobile?** Not built — we chose depth on OCR + search + routing. Say it, don't bluff. Roadmap only.

**Q: Business/scale economics?** Local CPU inference: no per-page API cost. ~seconds per page on a laptop CPU; GPU swap via `onnxruntime-gpu`. State per-1,000-pages cost only if you computed it; otherwise say "compute-only, no per-call fees".

---

## 5. Tough situations

| Situation | Do this |
|---|---|
| Judge uploads something strange and result is Partial | Good — point at the reason text. "This is by design: it tells you it didn't read it fully." |
| OCR takes long on the booth laptop | Show the live progress stage; open the log; mention single worker. Have a pre-processed page ready. |
| Wi-Fi dies | The core demo is local. Map tiles and road routing need internet — the trail still plans (straight-line, labelled). Phone hotspot as backup. |
| Login stuck | Demo accounts button on the login page (password in README). Keep a second browser logged in already. |
| "Prove it's not hard-coded" | Upload their image; show the request in the log with its request ID; open `ocr/pipeline.py`. |
| "What breaks first?" | The single OCR worker and SQLite's single writer. Fix: more workers, PostgreSQL. |
| "Handwritten accuracy?" | Coverage and confidence are measured; labelled handwriting isn't available — that's what the verified-line export builds. |
| Many visitors share the booth Wi-Fi / one IP and get "Too many requests" | Rate limits are per IP. Start the server with `RATE_LIMIT_MULTIPLIER=5` (scales every bucket) — and say that the limiter is why a brute-force login returns 429. |
| You don't know | "I don't have that measured. Here's what I'd measure and how." |

---

## 6. Role split (3-minute Q&A)
- **Systems/ML lead:** OCR pipeline, engines, training, concurrency, DB, security, failure modes.
- **Product/domain lead:** problem, users, heritage link, roadmap, ethics, economics.
- Rule: first person to raise a hand answers; the other adds one sentence at most. Numbers, not adjectives.

---

## 7. Pre-flight checklist (tonight)

- [ ] `git status` clean or committed; repo public with README, screenshots, demo video link
- [ ] Fresh machine test: `setup.bat` / manual steps → `start.bat` works; seed data present
- [ ] `JWT_SECRET` set; demo passwords noted (README) — rotate if exposed publicly
- [ ] App started **before** the judges arrive (engines warm up in the background on boot — first request is otherwise slow)
- [ ] Run `python -u -m scripts.e2e_random --pages 12 --out data/e2e_report.json` and read it (expect 165/165); compare with the numbers above
- [ ] Run `python -m pytest -q` (30 tests)
- [ ] 3 fresh test photos on the desktop **and** on the phone (one clean, one blurry, one with a photo of a printed page)
- [ ] Phone hotspot tested; laptop charger; HDMI adapter
- [ ] Slides exported to PDF as backup; demo video downloaded locally
- [ ] Rehearse the 5-minute pitch twice with a timer; stop at 4:50
- [ ] Rehearse five hostile questions from section 4 out loud

## 8. Commands to have in a terminal

```bash
cd backend
curl -s localhost:8000/api/v1/ready | python -m json.tool          # engines, worker heartbeat, breakers
python -m pytest -q                                                 # tests
python -m scripts.e2e_random --pages 12 --out data/e2e_report.json  # full randomised flow + accuracy
python -m scripts.eval_engines --snskrt-parquet <parquet> --labels <labels.csv> --synth-val <parquet>   # engine head-to-head
```
