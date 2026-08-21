#!/usr/bin/env bash
# v5.0 debug-pod bootstrap (RTX 2000, $0.24/hr). One paste, whole phase:
#   1. clone + deps
#   2. prepare a real UltraChat shard (train) + a calibration shard
#   3. talk x widths A/B at matched params  -> freeze winners in card
#   4. calibration on the real-data shard   -> freeze constants
#   5. leave everything in place for the registered 10M on the 4090
set -euo pipefail

cd /workspace
if [ ! -d iga-scale ]; then
  git clone --depth 1 https://github.com/LukeHamond1001/iga-scale.git
fi
cd iga-scale
pip install -q torch numpy datasets tokenizers

# train shard: ~8k convos (~10M tokens); calib shard: disjoint skip-range
python -m iga.lm_data_ultrachat prepare --convos 8000 \
  --out data/uc_debug --vocab 16384
python - <<'EOF'
from iga.lm_data_ultrachat import prepare
# calibration shard: conversations the training shard never saw
import iga.lm_data_ultrachat as U
orig = U.iter_convos
U.iter_convos = lambda limit, skip=0: orig(limit, skip=20000)
prepare("data/uc_calib", n_convos=1500, seed=1, vocab=16384)
EOF

# the two pre-registered-run decisions, in order:
python -m iga.lm_ab --steps 400 --d 128 --lanes 8 --chunk 512 \
  --data data/uc_debug 2>&1 | tee ab_results.txt
python -m iga.lm_calibrate --data data/uc_calib --chunks 60 \
  --out results/lm_constants_real.json 2>&1 | tee calib_results.txt

echo "=== debug phase complete ==="
echo "1. freeze talk/shape winners from ab_results.txt into the card"
echo "2. freeze results/lm_constants_real.json into the card"
echo "3. registered 10M: python -m iga.lm_train run --data data/uc_full \\"
echo "     --d <10M-tier d> --steps <budget> --device cuda"
