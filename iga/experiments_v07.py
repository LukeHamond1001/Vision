"""v0.7: pixels — the freebies-law gauntlet on rendered frames. Run:

    python -m iga.experiments_v07 tiny    # CPU smoke (~2 min)
    python -m iga.experiments_v07 full    # GPU: pretraining + gates + 24-seed behavior

The slow variable (charge) is global illumination smeared over every lit
pixel of 64×64×3 frames. Pre-registered gates, in order (R4):
  G-routing   CNN slow-band |corr| with true c >= 0.85 on held-out walk,
              and beats pixel-PCA's slow slice by >= 0.1
  G-geometry  start->pad chord / per-step scale reported (rigid ~7.6);
              must exceed the v0.5 linear ceiling (2.8)
  G-behavior  (full only) LadderAgent (round-10 config) on the CNN latent
              vs pixel-PCA latent, 24 seeds x 150 episodes: max_c and return
"""

from __future__ import annotations

import json
import sys
import time

import os

import torch

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

from .envs.chargeworld import ChargeWorld
from .envs.pixelcharge import PixelChargeWorld, PixelObservationLatent, PixelRenderer
from .experiments import RESULTS, bootstrap_ci, iqm
from .heads import FixedRewardHead, SumHead
from .ladder import LadderAgent, LadderConfig
from .latent import BandedLatent
from .pretrain import EncoderCNN, geodesic_pairs, pretrain_cnn_encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collect_pixel_walk(steps: int, size: int = 64, coverage_reset_every: int = 100,
                       seed: int = 0):
    """Random walk; returns (frames uint8 [T,3,H,W], true c [T]) — c for
    evaluation only."""
    env = ChargeWorld(BandedLatent([2, 1], [6, 2], seed=0), seed=seed)
    renderer = PixelRenderer(size)
    gen = torch.Generator().manual_seed(seed)
    env.reset()
    frames, cs, ps = [], [], []
    for t in range(steps):
        a = (torch.rand(2, generator=gen) * 2 - 1) * 0.1
        env.step(a)
        f = renderer.render(env.pos, env.c, env.pad, env.door, env.threshold)
        frames.append((f * 255).to(torch.uint8))
        cs.append(env.c)
        ps.append(env.pos.clone())
        if (t + 1) % coverage_reset_every == 0:
            env.reset()
            env.pos = torch.rand(2, generator=gen)
            env.c = float(torch.rand((), generator=gen))
    return torch.stack(frames), torch.tensor(cs), torch.stack(ps)


def encode_all(encoder, frames_u8: torch.Tensor, batch: int = 512) -> torch.Tensor:
    ts = list(encoder.parameters()) or list(encoder.buffers())
    dev = ts[0].device
    out = []
    with torch.no_grad():
        for i in range(0, frames_u8.shape[0], batch):
            x = frames_u8[i:i + batch].float().div_(255.0).to(dev)
            out.append(encoder(x).cpu())
    return torch.cat(out)


class PixelPCA(torch.nn.Module):
    """Linear control encoder: PCA over downsampled grayscale frames."""

    def __init__(self, frames_u8: torch.Tensor, latent_dim: int = 8):
        super().__init__()
        with torch.no_grad():
            x = frames_u8.float().div_(255.0).mean(1)                 # grayscale
            x = torch.nn.functional.avg_pool2d(x.unsqueeze(1), 4).flatten(1)
            sub = x[:: max(1, x.shape[0] // 4000)]
            mu = sub.mean(0)
            _, _, V = torch.pca_lowrank(sub - mu, q=latent_dim)
        self.register_buffer("mu", mu)
        self.register_buffer("V", V)
        self.register_buffer("out_scale", torch.ones(()))

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        single = frame.dim() == 3
        x = frame.unsqueeze(0) if single else frame
        x = x.mean(1)
        x = torch.nn.functional.avg_pool2d(x.unsqueeze(1), 4).flatten(1)
        z = ((x - self.mu) @ self.V) * self.out_scale
        return z.squeeze(0) if single else z


def band_corr(z: torch.Tensor, c: torch.Tensor, band_dims=(5, 3)) -> dict:
    out, off = {}, 0
    for k, bd in enumerate(band_dims):
        corr = torch.zeros(bd)
        for j in range(bd):
            corr[j] = torch.corrcoef(torch.stack([z[:, off + j], c]))[0, 1].abs()
        out[f"band{k}"] = float(corr.max())
        off += bd
    return out


def build_latent(encoder, size: int = 64) -> PixelObservationLatent:
    scratch = ChargeWorld(BandedLatent([2, 1], [6, 2], seed=0))
    lat = PixelObservationLatent(encoder, [5, 3], PixelRenderer(size),
                                 scratch.pad, scratch.door, scratch.threshold)
    lat._post_scale(lat.step_scale(0))
    return lat


def build_agent(seed: int, lat: PixelObservationLatent):
    env = PixelChargeWorld(lat, seed=seed)
    gen = torch.Generator().manual_seed(seed + 4000)
    world = torch.cat([torch.rand(256, 2, generator=gen),
                       torch.rand(256, 1, generator=gen)], dim=1)
    support = torch.stack([lat.embed(w) for w in world])
    with torch.no_grad():
        rho = float(torch.linalg.vector_norm(
            env.embed_world(env.start, 0.0) - env.embed_world(env.pad, 0.0))) / 0.76
    head_pos = SumHead(
        [FixedRewardHead(env.embed_world(env.door, 1.0), sigma=0.30 * rho, proxy_samples=support),
         FixedRewardHead(env.embed_world(env.pad, 0.1), sigma=0.20 * rho, proxy_samples=support),
         FixedRewardHead(env.embed_world(env.pad, 0.6), sigma=0.20 * rho, proxy_samples=support)],
        weights=[1.0, 0.3, 0.3])
    head_neg = FixedRewardHead(env.embed_world((0.5, 0.95), 0.0), sigma=0.12 * rho,
                               proxy_samples=support)
    with torch.no_grad():
        amp_slow = float(torch.linalg.vector_norm(
            (env.embed_world((0.5, 0.5), 1.0)
             - env.embed_world((0.5, 0.5), 0.0))[lat.band_slices[1]]))
    agent = LadderAgent(
        LadderConfig(seed=seed, veto_threshold=0.5, holds=(12, 12),
                     arrive_eps=(0.08 * rho, max(0.05 * max(amp_slow, 1e-3), 0.01)),
                     leash_radius=0.15 * rho, cap_radius=max(0.1 * rho, 0.02),
                     w_prog_bands=(1.0, 1.0), grad_proposals=True, use_flinch=False),
        head_pos, head_neg, lat)
    return agent, env


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "tiny"
    tiny = mode == "tiny"
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    print(f"[v0.7] device={DEVICE} mode={mode}", flush=True)

    steps = 2000 if tiny else 30000
    frames, c_true, _ = collect_pixel_walk(steps, seed=0)
    frames_test, c_test, pos_test = collect_pixel_walk(400 if tiny else 4000, seed=77)
    print(f"[v0.7] walk collected {frames.shape} in {time.time()-t0:.0f}s", flush=True)

    # Geodesic graph (round 8): pixel distance between dot positions
    # SATURATES once discs stop overlapping (~0.07 world), so node spacing
    # must sit inside the overlap radius and the local metric must be
    # full-res — else kNN shortcuts through photometric look-alikes and the
    # targets compress (measured: corr(dgeo,|dpos|) 0.12 -> 0.69, far/near
    # 1.8 -> 4.4 with dense nodes + k=4 + illum-normalized full-res).
    f = frames.float().div(255.0)
    fn = (f / f.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)).flatten(1)
    geo = geodesic_pairs(fn, n_nodes=400 if tiny else 1000, k=4,
                         n_pairs=1500 if tiny else 6000)
    del f, fn

    enc = pretrain_cnn_encoder(frames.float().div(255.0), band_dims=[5, 3],
                               taus=[40.0, 300.0], lags=[15, 60], segment_len=100,
                               context_amp=0.3, geo=geo, lam_geo=5.0,
                               epochs=60 if tiny else 1500,
                               batch=256 if tiny else 1024,
                               device=DEVICE, log_every=20 if tiny else 100)
    print(f"[v0.7] pretraining done at {time.time()-t0:.0f}s", flush=True)

    torch.save(enc.state_dict(), RESULTS / "v07_encoder.pt")    # always save
    pca = PixelPCA(frames)
    report = {}
    for name, e in (("cnn", enc), ("pca", pca)):
        z = encode_all(e, frames_test)
        report[name] = band_corr(z, c_test)
    # DIAGNOSTIC: what did each latent dim actually learn?
    z = encode_all(enc, frames_test)
    bright = frames_test.float().div(255.0).mean((1, 2, 3))
    gens = {"c": c_test, "x": pos_test[:, 0], "y": pos_test[:, 1], "bright": bright}
    print("[v0.7:probe] dim | " + " | ".join(f"{k:>6s}" for k in gens), flush=True)
    for j in range(z.shape[1]):
        row = [abs(float(torch.corrcoef(torch.stack([z[:, j], v]))[0, 1])) for v in gens.values()]
        band = "slow" if j >= 5 else "fast"
        print(f"[v0.7:probe] z{j} ({band}) | " + " | ".join(f"{r:6.2f}" for r in row), flush=True)
    lat = build_latent(enc)
    far = float(torch.linalg.vector_norm(
        lat.embed(torch.tensor([0.5, 0.1, 0.0])) - lat.embed(torch.tensor([0.2, 0.8, 0.0]))))
    report["geometry_far_over_step"] = far / 0.1
    g_rout = report["cnn"]["band1"] >= 0.85 and \
        report["cnn"]["band1"] - report["pca"]["band1"] >= 0.1
    g_geom = far / 0.1 > 2.8
    print(f"[v0.7] routing: cnn slow={report['cnn']['band1']:.3f} fast={report['cnn']['band0']:.3f} "
          f"| pca slow={report['pca']['band1']:.3f} -> {'PASS' if g_rout else 'FAIL'}", flush=True)
    print(f"[v0.7] geometry: far/step={far/0.1:.1f} (rigid ~7.6, linear ceiling 2.8) "
          f"-> {'PASS' if g_geom else 'FAIL'}", flush=True)
    report["gates"] = {"routing": bool(g_rout), "geometry": bool(g_geom)}
    (RESULTS / "v07_gates.json").write_text(json.dumps(report, indent=2))

    if tiny or not (g_rout and g_geom):
        print(f"[v0.7] stopping after gates ({'tiny mode' if tiny else 'GATE FAILED'}), "
              f"{time.time()-t0:.0f}s total", flush=True)
        return

    torch.save(enc.state_dict(), RESULTS / "v07_encoder.pt")
    conds = {"cnn": lat, "pca": build_latent(pca)}
    results = []
    for which, l in conds.items():
        per_seed = {"return": [], "max_c": []}
        for s in range(12):
            agent, env = build_agent(s, l)
            ret, mc, scored = 0.0, 0.0, 0
            for ep in range(150):
                stats = agent.run_episode(env, max_steps=160)
                if ep % 30 == 0:
                    print(f"[v0.7:hb] {which} seed {s} ep {ep} ({time.time()-t0:.0f}s)",
                          flush=True)
                if ep >= 75:
                    ret += stats["return"]
                    mc += env.max_c
                    scored += 1
            per_seed["return"].append(ret / scored)
            per_seed["max_c"].append(mc / scored)
            print(f"[v0.7] {which} seed {s}: max_c={per_seed['max_c'][-1]:.2f} "
                  f"ret={per_seed['return'][-1]:+.3f} ({time.time()-t0:.0f}s)", flush=True)
            with open(RESULTS / "v07_behavior_progress.jsonl", "a") as fh:
                fh.write(json.dumps({"which": which, "seed": s,
                                     "max_c": per_seed["max_c"][-1],
                                     "return": per_seed["return"][-1]}) + "\n")
        row = {"which": which, "per_seed": per_seed}
        import numpy as np
        for m in ("return", "max_c"):
            v = np.array(per_seed[m])
            lo, hi = bootstrap_ci(v)
            row[m] = {"iqm": iqm(v), "ci95": [lo, hi]}
        results.append(row)
        print(f"[v0.7] {which}: max_c={row['max_c']['iqm']:.3f} CI={row['max_c']['ci95']} "
              f"return={row['return']['iqm']:+.3f} CI={row['return']['ci95']}", flush=True)
    (RESULTS / "v07_pixels.json").write_text(json.dumps(results, indent=2))
    print(f"[v0.7] done in {(time.time()-t0)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
