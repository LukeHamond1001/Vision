"""v0.8 phase 1: the incapacity-prior library, entry two. Run:

    python -m iga.experiments_v08 tiny    # CPU smoke
    python -m iga.experiments_v08 full    # GPU: three encoders + routing verdicts

BankWorld's slow variable (banked count b) is pure spatial configuration —
luminance-identical dots relocate; global photometric statistics are
invariant to b BY CONSTRUCTION. Three slow pathways, pre-registered:

  P-photo   (negative control) v0.7's raw-photometric head MUST FAIL:
            slow-band |corr| with b < 0.4 — its input cannot vary with b.
  P-region  fixed 4x4 region-pooled means ADMIT b (bank-strip occupancy is
            region-visible) but also re-admit coarse position; the temporal
            prior must sort them. PASS: slow-band corr >= 0.7 and > fast.
  P-pca     control: report only.

Verdict semantics: photo-FAIL + region-PASS = the library thesis holds
(incapacity priors are per-generator-class, and temporal priors can sort
within an admitting pathway). region-FAIL with position in the slow band =
admitting pathways are insufficient — incapacity must be exclusive, a
sharper (more restrictive) law.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

from .envs.bankworld import BankWorld
from .experiments import RESULTS
from .experiments_v07 import PixelPCA, encode_all, git_push_results
from .pretrain import geodesic_pairs, pretrain_cnn_encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collect_bank_walk(steps: int, coverage_reset_every: int = 100, seed: int = 0):
    env = BankWorld(seed=seed)
    gen = torch.Generator().manual_seed(seed)
    frames, bs, xs = [], [], []
    for t in range(steps):
        a = (torch.rand(2, generator=gen) * 2 - 1) * 0.1
        env.step(a)
        frames.append((env.frame() * 255).to(torch.uint8))
        bs.append(float(env.b))
        xs.append(env.pos.clone())
        if (t + 1) % coverage_reset_every == 0:
            env.reset(banked_init=int(torch.randint(0, 7, (1,), generator=gen)))
            env.pos = torch.rand(2, generator=gen)
    return torch.stack(frames), torch.tensor(bs), torch.stack(xs)


def band_corr_multi(z: torch.Tensor, gens: dict, band_dims=(5, 3)) -> dict:
    out, off = {}, 0
    for k, bd in enumerate(band_dims):
        for name, v in gens.items():
            corr = torch.zeros(bd)
            for j in range(bd):
                corr[j] = torch.corrcoef(torch.stack([z[:, off + j], v]))[0, 1].abs()
            out[f"band{k}_{name}"] = float(corr.max())
        off += bd
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    tiny = mode == "tiny"
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"[v0.8] device={DEVICE} mode={mode}", flush=True)

    frames, b, x = collect_bank_walk(2000 if tiny else 30000, seed=0)
    frames_t, b_t, x_t = collect_bank_walk(400 if tiny else 4000, seed=77)
    print(f"[v0.8] walk {frames.shape} in {time.time()-t0:.0f}s "
          f"(b coverage: min={b.min():.0f} max={b.max():.0f})", flush=True)

    f = frames.float().div(255.0)
    fn = (f / f.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)).flatten(1)
    geo = geodesic_pairs(fn, n_nodes=150 if tiny else 1000, k=4,
                         n_pairs=1000 if tiny else 6000)
    del f, fn

    report = {}
    for smode in ("photo", "region"):
        enc = pretrain_cnn_encoder(frames.float().div(255.0), band_dims=[5, 3],
                                   taus=[40.0, 300.0], lags=[15, 60],
                                   segment_len=100, context_amp=0.3,
                                   geo=geo, lam_geo=5.0,
                                   epochs=60 if tiny else 1500,
                                   batch=256 if tiny else 1024,
                                   device=DEVICE, log_every=0,
                                   slow_mode=smode)
        z = encode_all(enc, frames_t)
        gens = {"b": b_t, "x": x_t[:, 0], "y": x_t[:, 1]}
        report[smode] = band_corr_multi(z, gens)
        r = report[smode]
        print(f"[v0.8] {smode:6s} slow: b={r['band1_b']:.3f} x={r['band1_x']:.2f} "
              f"y={r['band1_y']:.2f} | fast b={r['band0_b']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        git_push_results(f"v08-{smode}")

    pca = PixelPCA(frames)
    z = encode_all(pca, frames_t)
    report["pca"] = band_corr_multi(z, {"b": b_t, "x": x_t[:, 0], "y": x_t[:, 1]})
    print(f"[v0.8] pca    slow: b={report['pca']['band1_b']:.3f}", flush=True)

    p_photo = report["photo"]["band1_b"] < 0.4
    p_region = (report["region"]["band1_b"] >= 0.7
                and report["region"]["band1_b"] > report["region"]["band0_b"])
    print(f"[v0.8] P-photo (control MUST fail, <0.4): "
          f"{'CONFIRMED' if p_photo else 'VIOLATED'}", flush=True)
    print(f"[v0.8] P-region (>=0.7 and routed):       "
          f"{'PASS' if p_region else 'FAIL'}", flush=True)
    report["verdicts"] = {"photo_control_confirmed": bool(p_photo),
                          "region_pass": bool(p_region)}
    (RESULTS / "v08_library.json").write_text(json.dumps(report, indent=2))
    git_push_results("v08-verdicts")
    print(f"[v0.8] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
