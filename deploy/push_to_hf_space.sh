#!/bin/sh
# Publish the current commit to a FREE Hugging Face Gradio-SDK Space.
#   1. huggingface.co -> New Space -> SDK: Gradio -> Blank -> hardware: CPU Basic -> Public -> Create
#   2. huggingface.co/settings/tokens -> New token with "Write" access
#   3. HF_TOKEN=hf_xxx sh deploy/push_to_hf_space.sh yourname/lipisetu
# The website (frontend/dist) is built here and shipped inside the Space, because the Gradio build has no Node.js.
set -e
SPACE="${1:?usage: HF_TOKEN=... sh deploy/push_to_hf_space.sh <user/space>}"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face write token}"
ROOT="$(git rev-parse --show-toplevel)"
echo "Building the website..."
(cd "$ROOT/frontend" && npm install --no-audit --no-fund --silent && npm run build --silent)
TMP="$(mktemp -d)"
git -C "$ROOT" archive HEAD | tar -x -C "$TMP"
mkdir -p "$TMP/frontend" && rm -rf "$TMP/frontend/dist" && cp -R "$ROOT/frontend/dist" "$TMP/frontend/dist"
cp "$ROOT/deploy/hf_app.py" "$TMP/app.py"
cp "$TMP/backend/requirements.txt" "$TMP/requirements.txt"
printf 'libgl1\nlibglib2.0-0\n' > "$TMP/packages.txt"
{ cat "$ROOT/deploy/huggingface_README_header.md"; cat "$TMP/README.md"; } > "$TMP/README.new" && mv "$TMP/README.new" "$TMP/README.md"
if [ -n "$DRY_RUN" ]; then echo "Dry run: prepared files in $TMP"; exit 0; fi
cd "$TMP"
git init -q -b main
git add -A
git -c user.name="LipiSetu deploy" -c user.email="deploy@example.invalid" commit -q -m "Deploy LipiSetu"
git push -f "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}" main
echo "Pushed. Watch the build at https://huggingface.co/spaces/${SPACE}"
