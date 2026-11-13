# Hosting LipiSetu on Render, with Cloudflare in front

LipiSetu is one container: the FastAPI server, the OCR worker and the built website. Files used: `Dockerfile`,
`start.sh`, `render.yaml`. **The Docker build has not been run on the development machine (Docker is not installed
there), so watch the first build log.**

## 1. Render (the app)

1. Push the repository to GitHub (done: `Vaibhav2005-r/Hackculture`).
2. Render dashboard → **New → Blueprint** → choose the repository. Render reads `render.yaml`.
3. **Plan:** `standard` (2 GB RAM, about $25/month). The OCR models need roughly 1.5 GB; the 512 MB plans run out of
   memory. The 5 GB disk holds the SQLite database, uploaded images and the online/offline switch.
4. **Environment variables** (Render → service → Environment):
   | Variable | Value |
   |---|---|
   | `JWT_SECRET` | generated automatically |
   | `ADMIN_PASSWORD` | generated automatically: read it in the dashboard, it is the admin login |
   | `DEMO_ROLES` | `contributor,curator,viewer` (these three get one-click login buttons; admin does not) |
   | `NVIDIA_API_KEY` | optional. Enables the online-reading switch. Paste it here only, never in git. |
   | `OFFLINE_MODE` | `true` forbids every online service no matter what the dashboard says |
5. First deploy takes several minutes (build, then the first start seeds the 13 sample folios with real OCR). Check
   `https://<service>.onrender.com/api/v1/ready` → `"status": "ok"`.

Demo accounts: `contributor@lipisetu.in`, `curator@lipisetu.in`, `viewer@lipisetu.in` (password `Lipi@2026`, shown on the
README). Anyone on the internet can use them, so treat the public site as a demo: do not upload private material.
Admin is `admin@lipisetu.in` with the generated `ADMIN_PASSWORD`.

## 2. Cloudflare (domain, HTTPS, protection)

1. Add your domain to Cloudflare and point the registrar's nameservers at it.
2. Render → service → **Settings → Custom Domains** → add `lipisetu.yourdomain.com`.
3. Cloudflare → **DNS** → add `CNAME  lipisetu  →  <service>.onrender.com`, **Proxied** (orange cloud).
4. Cloudflare → **SSL/TLS** → mode **Full (strict)**.
5. Recommended: **Caching → Cache Rules**: bypass cache for `/api/*` and `/media/*`; **Security → WAF → Rate limiting**:
   e.g. 60 requests/minute per IP on `/api/v1/auth/*`.
6. The server already trusts `X-Forwarded-For`, so the per-IP rate limits see real visitor addresses.

No domain? Skip Cloudflare: the `onrender.com` URL already has HTTPS.

## 3. Things to know

- **Memory:** PyTorch is not installed in the image, so the offline CRNN engine is off and PP-OCR reads. The trained
  CRNN files are in the git-ignored `backend/data/` and are not part of the image anyway.
- **Online reading** sends page images to NVIDIA. It stays off until an admin turns it on from the Dashboard, and
  needs `NVIDIA_API_KEY`. Rotate any key that has ever been pasted into a chat.
- **Cold starts:** the standard plan stays running. The first request after a deploy may take a few seconds.
- **Backups:** the disk holds all data; take Render disk snapshots before big changes.
