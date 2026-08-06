"""v5.0 calibration pass — constants data-set, not hand-set.

Runs on the CALIBRATION split only (the instrument discipline:
calibration never sees eval). Measures three families of constants
and writes results/lm_constants.json, which the drive layer and the
gates read at registered-run time:

 1. Horizons per band: from the measured ask-back gap distribution —
    horizon = 2 x p90 gap of the band's bin (floor: 4 ticks). A hold
    lives long enough that a typical fact gets asked back inside it.
 2. Chance floors per gap bin: a RANDOM-INIT model's p(answer) and
    top-1 at the calibration probes. This is the floor the G-context
    gate's margin is measured against — established empirically,
    before training, per bin.
 3. Fidelity floor: random-init band fidelity mean + 2 sigma. Below
    this, a band's predictor knows nothing; the imagination gate's
    veto threshold starts here.

Usage:
  python -m iga.lm_calibrate            # weaver calibration split
  python -m iga.lm_calibrate --data D   # prepared shard directory
"""

import argparse
import json
import os

import numpy as np
import torch

from .lm_bands import BandLM, CLOCKS, N_BANDS
from .lm_conveyor import Vocab, Conveyor, splits
from .lm_drive import GAP_BINS, BIN_BAND, gap_bin


@torch.no_grad()
def run(data=None, d=64, lanes=4, T=512, chunks=30, seed=0,
        out="results/lm_constants.json"):
    if data:
        from .lm_data_ultrachat import UltraConveyor, load_tokenizer
        conveyor = UltraConveyor(data, n_lanes=lanes, offset_frac=0.5)
        tok = load_tokenizer(os.path.join(data, "tokenizer.json"))
        vocab_size = tok.get_vocab_size()
        source = f"shard:{data}"
    else:
        vocab = Vocab()
        vocab_size = len(vocab)
        conveyor = Conveyor(vocab, n_lanes=lanes, seed=splits(seed)["calib"])
        source = "weaver:calib"
    torch.manual_seed(seed)
    model = BandLM(vocab_size, d=d)          # random init, never trained
    st = model.init_state(lanes, "cpu")
    gaps_by_bin = {b: [] for b in range(len(GAP_BINS))}
    floor_by_bin = {b: {"p": [], "top1": []} for b in range(len(GAP_BINS))}
    fid_by_band = {k: [] for k in range(1, N_BANDS)}
    for _ in range(chunks):
        x, y, events = conveyor.chunk(T)
        logits, st, ticks = model(x, st, None)
        logp = torch.log_softmax(logits, dim=-1)
        for k in range(1, N_BANDS):
            for _, fid in ticks[k]:
                fid_by_band[k].extend(fid.tolist())
        for lane, evs in enumerate(events):
            for p, kind, dd in evs:
                if kind == "probe" and p > 0:
                    b = gap_bin(dd["gap"])
                    gaps_by_bin[b].append(dd["gap"])
                    floor_by_bin[b]["p"].append(
                        float(logp[lane, p - 1, dd["answer"]].exp()))
                    floor_by_bin[b]["top1"].append(
                        int(int(logits[lane, p - 1].argmax()) == dd["answer"]))
    horizons = {}
    for b, gaps in gaps_by_bin.items():
        band = BIN_BAND[b]
        default = 4 * CLOCKS[band]
        if gaps:
            horizons[band] = int(max(2 * float(np.percentile(gaps, 90)),
                                     default))
        else:
            horizons[band] = default
    for k in range(N_BANDS):                  # bands without a bin: default
        horizons.setdefault(k, 4 * CLOCKS[k])
    floors = {}
    for b, f in floor_by_bin.items():
        if f["p"]:
            floors[b] = {"p_mean": float(np.mean(f["p"])),
                         "p_p95": float(np.percentile(f["p"], 95)),
                         "top1": float(np.mean(f["top1"])),
                         "n": len(f["p"])}
    fid_floor = {}
    for k, v in fid_by_band.items():
        if v:
            fid_floor[k] = float(np.mean(v) + 2 * np.std(v))
    constants = {"source": source, "d_probe_model": d,
                 "horizons": {str(k): v for k, v in horizons.items()},
                 "chance_floors": {str(b): v for b, v in floors.items()},
                 "fid_floor": {str(k): v for k, v in fid_floor.items()},
                 "gap_counts": {str(b): len(g)
                                for b, g in gaps_by_bin.items()}}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(constants, f, indent=1)
    print(f"calibration source: {source}")
    print("horizons (tokens/band):",
          {k: horizons[k] for k in sorted(horizons)})
    for b in sorted(floors):
        lo, hi = GAP_BINS[b]
        print(f"  bin {b} (gap {lo}-{hi}): chance p={floors[b]['p_mean']:.4f} "
              f"top1={floors[b]['top1']:.3f} n={floors[b]['n']}")
    print("fid floors:", {k: round(v, 3) for k, v in fid_floor.items()})
    print(f"-> {out}")
    return constants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--chunks", type=int, default=30)
    ap.add_argument("--out", default="results/lm_constants.json")
    a = ap.parse_args()
    run(data=a.data, chunks=a.chunks, out=a.out)


if __name__ == "__main__":
    main()
