#!/usr/bin/env bash
# v10 single-pod boot sequence (CA-MTL-3 reality: the volume DC has
# A100 PCIe and nothing cheaper, so ONE A100 pod carries prep ->
# paid smoke -> full build -> flash; no relaunch-scarcity risk
# between stages). Each stage self-senses on the volume, so a
# relaunch of this same boot resumes wherever it died.
#
#   1. prep phases 1-5 (fetch, yield measure, judge freeze on real
#      mixes, smoke shards)
#   2. paid smoke at exact 20L shapes -> smoke.json (lam + lanes)
#   3. prep phase 6 (full corpus at the smoke's lane pick)
#   4. the flash (GO from pod env; kill-fix-relaunch inside)
set -uo pipefail
RAW=https://raw.githubusercontent.com/LukeHamond1001/Vision/main/scripts
curl -sSL "$RAW/pod_v10_prep.sh" | SKIP_TERMINATE=1 bash
curl -sSL "$RAW/pod_v10.sh"      | SKIP_TERMINATE=1 GO=0 bash
curl -sSL "$RAW/pod_v10_prep.sh" | SKIP_TERMINATE=1 bash
curl -sSL "$RAW/pod_v10.sh"      | GO="${GO:-0}" bash
