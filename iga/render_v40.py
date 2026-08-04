"""v4.0 demo renderer (local tool): the trace-overlay act and the
three-creatures act, from banked fleet policies.

    python -m iga.render_v40 trace      # act 6: gameplay + live goal agenda
    python -m iga.render_v40 creatures  # act 7: full vs no-proposer vs native
    python -m iga.render_v40 frames     # dump sample PNGs for visual check

Everything replays LOCALLY from results/ artifacts; no pods.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from .experiments import RESULTS
from .experiments_v40 import load_instrument, machine_cfg, make_readout
from .goal_machine import CHANNELS, GoalMachine
from .ppo_pixel import PixelActorCritic
from .render_demo import FFmpegWriter, _upscale

GAME = 320
PANEL_W = 460
H = GAME
FPS = 12


def _font(size: int):
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/Menlo.ttc",
                 "/System/Library/Fonts/Monaco.ttf",
                 "/Library/Fonts/Courier New.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_HEAD = _font(15)
F_MAIN = _font(13)
F_SMALL = _font(11)

COL_BG = (14, 16, 22)
COL_TEXT = (225, 228, 235)
COL_DIM = (120, 126, 140)
COL_STOCK = (90, 200, 250)
COL_VITAL = (255, 180, 90)
COL_ARRIVE = (120, 235, 130)
COL_TIMEOUT = (150, 150, 150)
COL_BAR_BG = (45, 50, 62)


def agenda_panel(t: int, holds: list, events: list, arrivals: int,
                 bonus: float, achv: int) -> np.ndarray:
    """The live goal-agenda panel: current wants with progress bars +
    scrolling event feed. holds: (band, label, frac_done). events:
    (kind, band, label) newest last."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (PANEL_W, H), COL_BG)
    d = ImageDraw.Draw(img)
    y = 10
    d.text((12, y), "GOAL AGENDA  (live, machine-written)", font=F_HEAD,
           fill=COL_TEXT)
    d.text((PANEL_W - 78, y + 1), f"t={t:4d}", font=F_MAIN, fill=COL_DIM)
    y += 30
    d.text((12, y), "wants held now", font=F_SMALL, fill=COL_DIM)
    y += 16
    if not holds:
        d.text((24, y), "(none — proposing)", font=F_MAIN, fill=COL_DIM)
        y += 22
    for band, label, frac in holds:
        col = COL_STOCK if band == 1 else COL_VITAL
        tag = "STOCK" if band == 1 else "VITAL"
        d.text((16, y), f"{tag}", font=F_SMALL, fill=col)
        d.text((66, y - 1), label, font=F_MAIN, fill=COL_TEXT)
        bx, bw = 250, 190
        d.rectangle([bx, y + 3, bx + bw, y + 11], fill=COL_BAR_BG)
        d.rectangle([bx, y + 3, bx + int(bw * min(max(frac, 0), 1)), y + 11],
                    fill=col)
        y += 22
    y += 8
    d.line([(12, y), (PANEL_W - 12, y)], fill=COL_BAR_BG, width=1)
    y += 8
    d.text((12, y), "recent events", font=F_SMALL, fill=COL_DIM)
    y += 16
    for kind, band, label in events[-8:]:
        col = {"arrive": COL_ARRIVE, "timeout": COL_TIMEOUT,
               "death": COL_TIMEOUT}.get(
            kind, COL_STOCK if band == 1 else COL_VITAL)
        sym = {"commit": "→", "arrive": "✓",
               "timeout": "·", "death": "†"}.get(kind, "?")
        extra = "   +bonus" if kind == "arrive" and band == 1 else ""
        d.text((16, y), f"{sym} {kind:7s} {label}{extra}", font=F_MAIN,
               fill=col)
        y += 19
    d.line([(12, H - 30), (PANEL_W - 12, H - 30)], fill=COL_BAR_BG, width=1)
    d.text((12, H - 24),
           f"arrivals {arrivals}   one-shot bonuses {bonus:.0f}   "
           f"achievements {achv}", font=F_SMALL, fill=COL_DIM)
    return np.asarray(img)


def _load_policy(name: str) -> PixelActorCritic:
    net = PixelActorCritic()
    net.load_state_dict(torch.load(RESULTS / name, map_location="cpu"))
    net.eval()
    return net


def replay_with_machine(policy_name: str, steps: int, env_seed: int,
                        sample_seed: int = 0):
    """Replay a policy locally with the goal machine attached. Returns
    per-step record: frames, holds snapshot, event log, counters, and
    per-life spans."""
    import crafter
    torch.manual_seed(sample_seed)
    inst, spec, stds = load_instrument()
    _, measure_np = make_readout(inst, spec)
    cfg = machine_cfg(stds, "full")
    gm = GoalMachine(cfg)
    net = _load_policy(policy_name)
    env = crafter.Env(seed=env_seed)
    obs = env.reset()
    rec = {"frames": [], "holds": [], "ev_len": [], "counts": [],
           "lives": []}
    events: list = []
    life_start = 0
    achv_seen: set = set()
    m = measure_np(obs[None])[0]
    gm.reset(m)
    n_arriv = 0
    trace_seen = len(gm.trace)
    for t in range(steps):
        with torch.no_grad():
            x = torch.from_numpy(obs[None]).permute(0, 3, 1, 2).float() / 255.0
            logits, _ = net(x)
            a = int(torch.distributions.Categorical(logits=logits).sample())
        obs, r, done, info = env.step(a)
        m = measure_np(obs[None])[0]
        if done:
            gm.step(gm.m.clone(), done=True)
        else:
            gm.step(m)
        # translate the machine's own trace into panel events, in order
        for (tt, kind, band, ch, lvl) in gm.trace[trace_seen:]:
            if kind == "arrive":
                n_arriv += 1
            events.append((kind, band, f"{ch}≥{lvl:.0f}"))
        trace_seen = len(gm.trace)
        if done:
            events.append(("death", 1, "life ends"))
            rec["lives"].append((life_start, t, n_arriv))
            life_start = t + 1
            achv_seen = set()
        for ach, n in info.get("achievements", {}).items():
            if n > 0:
                achv_seen.add(ach)
        holds = []
        for band, h in enumerate(gm.holds):
            if h is None:
                continue
            phi_now = gm._phi(gm.m, h.channel, h.target)
            frac = 1.0 - (phi_now / h.phi_commit if h.phi_commit > 1e-9
                          else 0.0)
            holds.append((band, f"{CHANNELS[h.channel]}≥{h.target:.0f}",
                          frac))
        if done:
            obs = env.reset()
            m = measure_np(obs[None])[0]
            gm.reset(m)
            trace_seen = len(gm.trace)
            holds = []
        rec["frames"].append(obs.copy())
        rec["holds"].append(holds)
        rec["counts"].append((n_arriv, gm.total_bonus, len(achv_seen)))
        rec["ev_len"].append(len(events))
    rec["lives"].append((life_start, steps - 1, n_arriv))
    rec["event_stream"] = events
    return rec


def act_trace(policy="v40_policy_full_s4.pt", env_seed=411, steps=4000,
              out="results/video/act6_trace_overlay.mp4"):
    rec = replay_with_machine(policy, steps, env_seed)
    # richest life: most arrivals within the life (cumulative diffs)
    scored = []
    prev_arr = 0
    for (lo, hi, arr_cum) in rec["lives"]:
        scored.append((arr_cum - prev_arr, hi - lo, lo, hi))
        prev_arr = arr_cum
    best = max(scored)
    lo, hi = best[2], best[3]
    print(f"[act6] richest life t=[{lo},{hi}] ({best[0]} arrivals, "
          f"{best[1]} steps) of {len(rec['lives'])} lives", flush=True)
    wr = FFmpegWriter(out, fps=FPS)
    for t in range(lo, hi + 1):
        game = _upscale(rec["frames"][t], GAME)
        arrivals, bonus, achv = rec["counts"][t]
        panel = agenda_panel(t - lo, rec["holds"][t],
                             rec["event_stream"][:rec["ev_len"][t]],
                             arrivals, bonus, achv)
        wr.append_data(np.concatenate([game, panel], axis=1))
    wr.close()
    print(f"[act6] wrote {out}", flush=True)


def act_creatures(seed=2, env_seed=555, steps=900,
                  out="results/video/act7_three_creatures.mp4"):
    from PIL import Image, ImageDraw
    recs = {}
    for arm, fname in [("full (drives+ladder)", f"v40_policy_full_s{seed}.pt"),
                       ("no-proposer (drives only)",
                        f"v40_policy_no-proposer_s{seed}.pt"),
                       ("native (paid per achievement)",
                        f"v40_policy_native_s{seed}.pt")]:
        recs[arm] = replay_plain(fname, steps, env_seed)
    wr = FFmpegWriter(out, fps=FPS)
    for t in range(steps):
        cols = []
        for arm, rec in recs.items():
            game = _upscale(rec["frames"][t], 280)
            img = Image.new("RGB", (280, 280 + 40), COL_BG)
            img.paste(Image.fromarray(game), (0, 40))
            d = ImageDraw.Draw(img)
            d.text((8, 6), arm, font=F_SMALL, fill=COL_TEXT)
            d.text((8, 22), f"achievements {rec['achv'][t]}",
                   font=F_SMALL, fill=COL_DIM)
            cols.append(np.asarray(img))
        wr.append_data(np.concatenate(cols, axis=1))
    wr.close()
    print(f"[act7] wrote {out}", flush=True)


def replay_plain(policy_name: str, steps: int, env_seed: int,
                 sample_seed: int = 0):
    import crafter
    torch.manual_seed(sample_seed)
    net = _load_policy(policy_name)
    env = crafter.Env(seed=env_seed)
    obs = env.reset()
    rec = {"frames": [], "achv": []}
    seen: set = set()
    for t in range(steps):
        with torch.no_grad():
            x = torch.from_numpy(obs[None]).permute(0, 3, 1, 2).float() / 255.0
            logits, _ = net(x)
            a = int(torch.distributions.Categorical(logits=logits).sample())
        obs, r, done, info = env.step(a)
        for ach, n in info.get("achievements", {}).items():
            if n > 0:
                seen.add(ach)
        if done:
            seen = set()
            obs = env.reset()
        rec["frames"].append(obs.copy())
        rec["achv"].append(len(seen))
    return rec


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "frames"
    (RESULTS / "video").mkdir(exist_ok=True)
    if mode == "trace":
        act_trace()
    elif mode == "creatures":
        act_creatures()
    elif mode == "frames":
        rec = replay_with_machine("v40_policy_full_s4.pt", 400, 411)
        from PIL import Image
        for i, t in enumerate((120, 250, 380)):
            game = _upscale(rec["frames"][t], GAME)
            arrivals, bonus, achv = rec["counts"][t]
            panel = agenda_panel(t, rec["holds"][t],
                                 rec["event_stream"][:rec["ev_len"][t]],
                                 arrivals, bonus, achv)
            img = np.concatenate([game, panel], axis=1)
            p = str(RESULTS / "video" / f"act6_sample_{i}.png")
            Image.fromarray(img).save(p)
            print("sample:", p, flush=True)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
