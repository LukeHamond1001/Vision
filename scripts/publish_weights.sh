#!/usr/bin/env bash
# Publish the organism's weights to Hugging Face so anyone can run it.
# One-time: hf auth login   (then run this script)
# Uploads the living body + tokenizer to  <you>/one-token-organism
set -euo pipefail
REPO=${1:-one-token-organism}
WHO=$(hf auth whoami 2>/dev/null | head -1) || {
  echo "not logged in — run: hf auth login   (or prefix this script"
  echo "with HF_TOKEN=hf_...  — paste is invisible at the login"
  echo "prompt: Cmd+V then Enter still works)"; exit 1; }
echo "logged in as $WHO — creating $REPO and uploading (~1.2G)"
hf repo create "$REPO" --type model -y 2>/dev/null || true
hf upload "$REPO" data/organism_life.pt organism_life.pt
hf upload "$REPO" data/ship_tok.json ship_tok.json
hf upload "$REPO" README.md README.md
echo "done — weights at https://huggingface.co/$WHO/$REPO"
echo "update README.md quickstart link if the repo name differs."
