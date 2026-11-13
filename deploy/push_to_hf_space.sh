#!/bin/sh
# Publish the current commit to a Hugging Face Docker Space (free hosting).
#   1. huggingface.co -> New Space -> SDK: Docker (blank) -> note its id, e.g. yourname/lipisetu
#   2. huggingface.co/settings/tokens -> New token with "Write" access
#   3. HF_TOKEN=hf_xxx sh deploy/push_to_hf_space.sh yourname/lipisetu
set -e
SPACE="${1:?usage: HF_TOKEN=... sh deploy/push_to_hf_space.sh <user/space>}"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face write token}"
ROOT="$(git rev-parse --show-toplevel)"
TMP="$(mktemp -d)"
git -C "$ROOT" archive HEAD | tar -x -C "$TMP"
{ cat "$ROOT/deploy/huggingface_README_header.md"; cat "$TMP/README.md"; } > "$TMP/README.new" && mv "$TMP/README.new" "$TMP/README.md"
cd "$TMP"
git init -q -b main
git add -A
git -c user.name="LipiSetu deploy" -c user.email="deploy@example.invalid" commit -q -m "Deploy LipiSetu"
git push -f "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}" main
echo "Pushed. Watch the build at https://huggingface.co/spaces/${SPACE}"
