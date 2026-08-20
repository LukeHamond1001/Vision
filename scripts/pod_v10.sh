#!/bin/bash
# v10 — THE LIFETIME FLASH (500M). LAUNCH REQUIRES THE USER'S
# EXPLICIT GO: this script costs real money ($300-450 band; the
# paid smoke below settles the real number BEFORE the run commits).
#
# Protocol (V10_FLASH section 6, plan step 5): paid real-shard
# smoke -> config freeze -> flash with banking + watchdog +
# heartbeats -> kill-fix-relaunch on any KILL sentinel.
#
# ORGAN FLAGS are frozen from the gate verdicts
# (results/evidence/v10_gates.json + a76_gate.json) at launch prep:
# only winners ship. Placeholders below marked FREEZE-AT-LAUNCH.
set -euo pipefail

WORK=/workspace
REPO=$WORK/iga-scale
DATA=$WORK/v10
OUT=$WORK/v10_out
mkdir -p "$DATA" "$OUT"

# ---------- 0. environment sanity (A54b/A46 canaries) ----------
python3 - <<'EOF'
import torch
assert torch.cuda.is_available()
p = torch.cuda.get_device_properties(0)
total = p.total_memory / 2**30
free = (p.total_memory - torch.cuda.memory_allocated()) / 2**30
print(f"GPU {p.name} total {total:.1f}GiB free {free:.1f}GiB")
assert total > 70, "A100 80GB class required"
x = torch.empty(int(60e9 // 4), dtype=torch.float32, device="cuda")
del x; torch.cuda.empty_cache()
print("capacity canary: claimed 60GiB ok")
EOF

# ---------- 1. corpus (full tier, pod-side; A54c: outputs on the
# volume, sources fetched then deleted after tokenization) --------
if [ ! -f "$DATA/flash/manifest.json" ]; then
  bash "$REPO/scripts/fetch_v10_corpus.sh" full "$DATA/raw"
  # budget: ONE EPOCH of the spine — measured, not assumed. The
  # builder consumes sources to exhaustion per stage; pass the
  # measured yield printed by a dry token-count pass.
  python3 -m iga.lm_data_life prepare \
    --out "$DATA/flash" --budget "${BUDGET:?set BUDGET from the \
measured one-epoch yield}" --lives "${LIVES:?smoke-determined}" \
    --ultrachat "$DATA/raw/ultrachat_train_0.jsonl" \
    --st2-dir "$DATA/raw" --magpie-dir "$DATA/raw" --seed 10
  python3 -m iga.lm_data_life prepare \
    --out "$DATA/flash_eval" --budget 40000000 --lives 2 \
    --ultrachat "$DATA/raw/ultrachat_train_9.jsonl" --skip 0 \
    --st2-dir "$DATA/raw" --magpie-dir "$DATA/raw" --seed 999 \
    --tokenizer "$DATA/flash/tokenizer.json"
fi

# ---------- 2. PAID SMOKE at exact shapes on the real shard ------
# (A54d law: quiet-data smokes lie; this one measures holds/step
# for the lam pairing, tok/s for the cost figure, and peak memory
# for the lane count. Abort thresholds inline.)
LANES="${LANES:-6}"
python3 - <<EOF
import sys, time, torch
sys.path.insert(0, "$REPO")
from iga.lm_train import train
from iga.lm_sleep import Sleeper
sl = Sleeper(arm="C", every=8, block_chunks=2, seed=1)
sl.press_pay = (2048, 256)
t0 = time.time()
model, drive, vocab, ce0, ce1 = train(
    d=1280, lanes=$LANES, T=2048, steps=60, seed=0, device="cuda",
    arch="hybrid", store="matrix", keyed="logit", norm_mix=True,
    aux_trunk=0.2, use_xl=False, gate_init=-2.0,
    lam=0.02,  # provisional; recomputed below from measured density
    data="$DATA/flash", sleep=sl, log_every=20,
    clocks={3:1, 4:8, 5:64, 6:512})
dt = time.time() - t0
toks = 60 * $LANES * 2048
holds_per_step = len(drive.ledger) / 60
lam = min(0.25, 0.25 / max(holds_per_step, 1))   # A60f pairing
peak = torch.cuda.max_memory_allocated() / 2**30
print(f"SMOKE tok/s {toks/dt:,.0f} holds/step {holds_per_step:.1f} "
      f"lam-> {lam:.4f} peak {peak:.1f}GiB")
assert toks / dt > 8000, "throughput floor"
assert peak < 70, "memory headroom"
open("$OUT/smoke.json", "w").write(
    '{"lam": %.5f, "tok_s": %.0f, "holds": %.2f}'
    % (lam, toks/dt, holds_per_step))
EOF

echo "SMOKE PASSED. Review $OUT/smoke.json, set lam/lanes/BUDGET,"
echo "then run the flash stage below with GO=1."
[ "${GO:-0}" = "1" ] || exit 0

# ---------- 3. THE FLASH (kill-fix-relaunch loop) ----------------
# FREEZE-AT-LAUNCH: --band-widths per A71 gate; --tie-embed per
# A75; sleeper splice/novelty/homeostasis/dream per A72-A74/A76/
# A77 verdicts; lam from smoke.json; lr 4e-5 warmup 2000.
# Sleep dose ladder: sleepless infancy — the driver (v10_driver.py,
# launch-prep deliverable) flips sleeper.every at the manifest's
# stage boundaries and runs heartbeats on every checkpoint:
#   python3 scripts/heartbeat_v10.py --ckpt latest.pt \
#     --data $DATA/flash --eval-data $DATA/flash_eval \
#     --tokens N --total-tokens BUDGET --out $OUT/hb.jsonl \
#     || { echo KILL; ...bank, fix, relaunch; }
echo "flash driver: launch-prep deliverable (v10_driver.py) — see"
echo "V10_FLASH section 6 and the approved plan, step 5."
