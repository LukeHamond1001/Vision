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
F_HEAD2 = _font(30)
F_MAIN2 = _font(26)
F_SMALL2 = _font(22)

COL_BG = (14, 16, 22)
COL_TEXT = (225, 228, 235)
COL_DIM = (120, 126, 140)
COL_STOCK = (90, 200, 250)
COL_VITAL = (255, 180, 90)
COL_ARRIVE = (120, 235, 130)
COL_TIMEOUT = (150, 150, 150)
COL_BAR_BG = (45, 50, 62)


def agenda_panel(t: int, holds: list, events: list, arrivals: int,
                 bonus: float, achv: int, s: int = 1) -> np.ndarray:
    """The live goal-agenda panel: current wants with progress bars +
    scrolling event feed. holds: (band, label, frac_done). events:
    (kind, band, label) newest last. s: integer UI scale (2 = crisp HD)."""
    from PIL import Image, ImageDraw
    fh = F_HEAD2 if s == 2 else F_HEAD
    fm = F_MAIN2 if s == 2 else F_MAIN
    fs = F_SMALL2 if s == 2 else F_SMALL
    W, HH = PANEL_W * s, H * s
    img = Image.new("RGB", (W, HH), COL_BG)
    d = ImageDraw.Draw(img)
    y = 10 * s
    d.text((12 * s, y), "GOAL AGENDA  (live, machine-written)", font=fh,
           fill=COL_TEXT)
    d.text((W - 78 * s, y + s), f"t={t:4d}", font=fm, fill=COL_DIM)
    y += 30 * s
    d.text((12 * s, y), "wants held now", font=fs, fill=COL_DIM)
    y += 16 * s
    if not holds:
        d.text((24 * s, y), "(none — proposing)", font=fm, fill=COL_DIM)
        y += 22 * s
    for band, label, frac in holds:
        col = COL_STOCK if band == 1 else COL_VITAL
        tag = "STOCK" if band == 1 else "VITAL"
        d.text((16 * s, y), f"{tag}", font=fs, fill=col)
        d.text((66 * s, y - s), label, font=fm, fill=COL_TEXT)
        bx, bw = 250 * s, 190 * s
        d.rectangle([bx, y + 3 * s, bx + bw, y + 11 * s], fill=COL_BAR_BG)
        d.rectangle([bx, y + 3 * s,
                     bx + int(bw * min(max(frac, 0), 1)), y + 11 * s],
                    fill=col)
        y += 22 * s
    y += 8 * s
    d.line([(12 * s, y), (W - 12 * s, y)], fill=COL_BAR_BG, width=s)
    y += 8 * s
    d.text((12 * s, y), "recent events", font=fs, fill=COL_DIM)
    y += 16 * s
    for kind, band, label in events[-8:]:
        col = {"arrive": COL_ARRIVE, "timeout": COL_TIMEOUT,
               "death": COL_TIMEOUT}.get(
            kind, COL_STOCK if band == 1 else COL_VITAL)
        sym = {"commit": "→", "arrive": "✓",
               "timeout": "·", "death": "†"}.get(kind, "?")
        extra = "   +bonus" if kind == "arrive" and band == 1 else ""
        d.text((16 * s, y), f"{sym} {kind:7s} {label}{extra}", font=fm,
               fill=col)
        y += 19 * s
    d.line([(12 * s, HH - 30 * s), (W - 12 * s, HH - 30 * s)],
           fill=COL_BAR_BG, width=s)
    d.text((12 * s, HH - 24 * s),
           f"arrivals {arrivals}   one-shot bonuses {bonus:.0f}   "
           f"achievements {achv}", font=fs, fill=COL_DIM)
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
    S = 2                                        # HD scale
    prev_ev = rec["ev_len"][lo] if lo > 0 else 0
    for t in range(lo, hi + 1):
        game = _upscale(rec["frames"][t], GAME * S)
        arrivals, bonus, achv = rec["counts"][t]
        new_events = rec["event_stream"][prev_ev:rec["ev_len"][t]]
        arrived_now = any(e[0] == "arrive" for e in new_events)
        prev_ev = rec["ev_len"][t]
        panel = agenda_panel(t - lo, rec["holds"][t],
                             rec["event_stream"][:rec["ev_len"][t]],
                             arrivals, bonus, achv, s=S)
        row = np.concatenate([game, panel], axis=1)
        if arrived_now:
            # freeze ~0.7s with a green flash border: the eye is TOLD
            flash = row.copy()
            b = 6 * S
            flash[:b, :] = COL_ARRIVE
            flash[-b:, :] = COL_ARRIVE
            flash[:, :b] = COL_ARRIVE
            flash[:, GAME * S - b:GAME * S] = COL_ARRIVE
            for k in range(int(0.7 * FPS)):
                wr.append_data(flash if k < FPS // 3 else row)
        wr.append_data(row)
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


def three_worlds_card(out="results/video/act8_three_worlds.mp4",
                      png="results/video/act8_three_worlds.png",
                      hold_s: float = 8.0):
    """The anti-'custom-harness' card: one drive layer, three worlds,
    zero law changes. Thumbnails are real env frames; results are the
    banked numbers."""
    from PIL import Image, ImageDraw
    W, COLW, HH = 1200, 400, 560
    img = Image.new("RGB", (W, HH), COL_BG)
    d = ImageDraw.Draw(img)
    d.text((24, 18), "ONE DRIVE LAYER.  THREE WORLDS.  ZERO LAW CHANGES.",
           font=F_HEAD, fill=COL_TEXT)
    d.text((24, 44), "registers · prospective proposer · one-shot curiosity"
           " · non-farmable ledger — identical code; only the channel"
           " manifest changed", font=F_SMALL, fill=COL_DIM)

    # thumbnails
    import crafter
    cf = crafter.Env(seed=7).reset()
    from .boatrace_env import BoatRace
    bf = BoatRace().reset()
    if isinstance(bf, tuple):
        bf = bf[0]
    bf = np.asarray(bf)
    if bf.dtype != np.uint8:
        bf = (np.clip(bf, 0, 1) * 255).astype(np.uint8)
    if bf.ndim == 3 and bf.shape[0] in (1, 3):        # CHW -> HWC
        bf = np.transpose(bf, (1, 2, 0))
    if bf.shape[-1] == 1:
        bf = np.repeat(bf, 3, -1)

    def put_thumb(arr, cx, size=230):
        im = Image.fromarray(arr).resize((size, size), Image.NEAREST)
        img.paste(im, (cx + (COLW - size) // 2, 84))

    cols = [
        ("BOAT RACE  (reward-hacking world)",
         "channel: race-progress gauge — an EXTERNAL task variable",
         ["engineered reward: HACKED (0.00 laps, 3/3 seeds)",
          "register on the same gauge: RACES 6.3-7.1 laps/ep",
          "the reward that cannot be paid to cheat"]),
        ("BATTERY ROBOT  (proprioceptive world)",
         "channels: battery, temperature, wear — somatic telemetry",
         ["drives alone: brownout 0.05 vs task arms 0.31 (10/10)",
          "homeostasis by learned frugality, never harmful",
          "the tau ladder: 4 / 54 / 92 / 28,856 steps"]),
        ("CRAFTER  (pixel survival world)",
         "channels: vitals + stocks — read from raw pixels",
         ["collection ladder climbed in 96% of lives (5 seeds)",
          "+0.8 achievements vs ablation, CI95 [+0.4, +1.2]",
          "every want readable in the live trace"]),
    ]
    for i, (title, chan, lines) in enumerate(cols):
        cx = i * COLW
        if i:
            d.line([(cx, 78), (cx, HH - 46)], fill=COL_BAR_BG, width=1)
        thumb = [bf, None, cf][i]
        if thumb is None:
            # battery world: draw the brownout bars (no mujoco renderer)
            bx, by = cx + 95, 130
            d.text((bx, by - 24), "brownout fraction", font=F_SMALL,
                   fill=COL_DIM)
            for j, (lbl, v, col) in enumerate([
                    ("task-reward arms", 0.31, (215, 120, 110)),
                    ("drives only", 0.048, COL_ARRIVE)]):
                y0 = by + j * 64
                d.text((bx, y0), lbl, font=F_SMALL, fill=COL_TEXT)
                d.rectangle([bx, y0 + 18, bx + 210, y0 + 34],
                            fill=COL_BAR_BG)
                d.rectangle([bx, y0 + 18, bx + int(210 * v / 0.35),
                             y0 + 34], fill=col)
                d.text((bx + 216, y0 + 18), f"{v:.2f}", font=F_SMALL,
                       fill=COL_DIM)
        else:
            put_thumb(thumb, cx)
        y = 330
        d.text((cx + 24, y), title, font=F_MAIN, fill=COL_TEXT)
        y += 24
        d.text((cx + 24, y), chan, font=F_SMALL, fill=COL_STOCK)
        y += 26
        for ln in lines:
            d.text((cx + 24, y), "· " + ln, font=F_SMALL, fill=COL_DIM)
            y += 20
    d.line([(24, HH - 40), (W - 24, HH - 40)], fill=COL_BAR_BG, width=1)
    d.text((24, HH - 32), "the sensor manifest is per-world plumbing "
           "(6 lines each); the laws never changed — that is the claim",
           font=F_SMALL, fill=COL_TEXT)
    arr = np.asarray(img)
    Image.fromarray(arr).save(png)
    wr = FFmpegWriter(out, fps=FPS)
    for _ in range(int(hold_s * FPS)):
        wr.append_data(arr)
    wr.close()
    print(f"[act8] wrote {out} and {png}", flush=True)


def replay_sleep(policy_name: str, steps: int, env_seed: int,
                 sample_seed: int = 0):
    """Replay tracking night-sleeps and daylight (goal-swap act)."""
    import crafter
    torch.manual_seed(sample_seed)
    net = _load_policy(policy_name)
    env = crafter.Env(seed=env_seed)
    obs = env.reset()
    rec = {"frames": [], "slept": [], "night": []}
    slept = 0
    for t in range(steps):
        with torch.no_grad():
            x = torch.from_numpy(obs[None]).permute(0, 3, 1, 2).float() / 255.0
            logits, _ = net(x)
            a = int(torch.distributions.Categorical(logits=logits).sample())
        obs, r, done, info = env.step(a)
        day = float(env._world.daylight)
        if a == 6 and day < 0.3:
            slept += 1
        if done:
            slept = 0
            obs = env.reset()
        rec["frames"].append(obs.copy())
        rec["slept"].append(slept)
        rec["night"].append(day < 0.3)
    return rec


def act_goalswap(env_seed=808, steps=900,
                 out="results/video/act10_goal_swap.mp4"):
    """Same drives, one desire deleted: v1.2 wired (energy IN the want)
    vs v1.3 swap (energy DELETED by a one-line mask)."""
    from PIL import Image, ImageDraw
    recs = [("energy IN the want   (sleeps ~4.3x/night, fleet mean)",
             replay_sleep("v12_policy_s1_wired.pt", steps, env_seed)),
            ("energy DELETED from the want   (one-line edit; ~1.3x)",
             replay_sleep("v13_policy_swap.pt", steps, env_seed))]
    # center the clip on the first substantial night (the effect is a
    # NIGHT behavior; daytime footage shows nothing)
    night = recs[0][1]["night"]
    first_night = next((i for i in range(len(night))
                        if all(night[i:i + 20])), 0)
    lo = max(0, first_night - 60)
    hi = min(steps, lo + 420)
    print(f"[act10] night window t=[{lo},{hi}]", flush=True)
    wr = FFmpegWriter(out, fps=FPS)
    for t in range(lo, hi):
        cols = []
        for label, rec in recs:
            game = _upscale(rec["frames"][t], 300)
            img = Image.new("RGB", (300, 300 + 44), COL_BG)
            img.paste(Image.fromarray(game), (0, 44))
            d = ImageDraw.Draw(img)
            d.text((8, 5), label[:44], font=F_SMALL, fill=COL_TEXT)
            night = "NIGHT" if rec["night"][t] else "day"
            col = COL_STOCK if rec["night"][t] else COL_DIM
            d.text((8, 24), f"{night}   sleeps this life: "
                   f"{rec['slept'][t]}", font=F_SMALL, fill=col)
            cols.append(np.asarray(img))
        wr.append_data(np.concatenate(cols, axis=1))
    wr.close()
    print(f"[act10] wrote {out}", flush=True)


def verdict_card(out="results/video/act11_verdict_card.mp4",
                 png="results/video/act11_verdict_card.png",
                 hold_s: float = 8.0):
    from PIL import Image, ImageDraw
    W, HH = 1000, 560
    img = Image.new("RGB", (W, HH), COL_BG)
    d = ImageDraw.Draw(img)
    d.text((24, 18), "THE SEQUENCING FLEET — 5 seeds x 3 arms x 3M steps",
           font=F_HEAD, fill=COL_TEXT)
    rows = [
        ("native", "told what to want (paid per achievement; 3 seeds)", "10.0",
         (215, 120, 110)),
        ("full", "NEVER told — drives + goal ladder only", "3.0",
         COL_ARRIVE),
        ("no-proposer", "same drives, ladder removed", "2.0", COL_DIM),
    ]
    y = 80
    for arm, desc, med, col in rows:
        d.text((40, y), arm, font=F_MAIN, fill=col)
        d.text((190, y), desc, font=F_MAIN, fill=COL_TEXT)
        d.text((780, y - 6), med, font=F_HEAD, fill=col)
        if arm == "native":
            d.text((840, y + 2), "achv/episode", font=F_SMALL,
                   fill=COL_DIM)
        bw = int(560 * float(med) / 10.0)
        d.rectangle([190, y + 22, 190 + 560, y + 34], fill=COL_BAR_BG)
        d.rectangle([190, y + 22, 190 + bw, y + 34], fill=col)
        y += 70
    y += 10
    d.line([(24, y), (W - 24, y)], fill=COL_BAR_BG, width=1)
    y += 16
    d.text((40, y), "pre-registered gate (ladder effect, paired vs "
           "ablation):  >= +1.0", font=F_MAIN, fill=COL_TEXT)
    y += 26
    d.text((40, y), "measured:  +1, +1, +1, +1, 0  ->  mean +0.80,  "
           "t-CI95 [+0.24, +1.36]  (n=5; sign test p=0.125)",
           font=F_MAIN, fill=COL_TEXT)
    y += 26
    d.text((40, y), "verdict:  GATE FAILED — direction consistent (4/5), "
           "magnitude under the bar; n=5 — reported as registered",
           font=F_MAIN, fill=(235, 200, 120))
    y += 40
    d.text((40, y), "behavior redirection, full vs ablation: collection "
           "x3.9 · cow-hunting x2.8 · zombie-fighting x5.6 — at survival "
           "parity", font=F_SMALL, fill=COL_DIM)
    y += 22
    d.text((40, y), "we report the miss because the numbers are the "
           "point. gates are gates.", font=F_SMALL, fill=COL_DIM)
    arr = np.asarray(img)
    Image.fromarray(arr).save(png)
    wr = FFmpegWriter(out, fps=FPS)
    for _ in range(int(hold_s * FPS)):
        wr.append_data(arr)
    wr.close()
    print(f"[act11] wrote {out}", flush=True)


def reversal_card(out="results/video/act12_reversal_card.mp4",
                  png="results/video/act12_reversal_card.png",
                  hold_s: float = 8.0):
    from PIL import Image, ImageDraw
    W, HH = 1000, 480
    img = Image.new("RGB", (W, HH), COL_BG)
    d = ImageDraw.Draw(img)
    d.text((24, 18), "THE REVERSAL — what the drive cannot measure, "
           "it learns to avoid", font=F_HEAD, fill=COL_TEXT)
    d.text((24, 46), "tables and pickaxes are NOT measured channels in "
           "generation 1. watch what that does:", font=F_SMALL,
           fill=COL_DIM)
    pairs = [("tables placed  (3M steps, 5 seeds)", 772, 20),
             ("wood pickaxes crafted", 45, 3)]
    y = 100
    for label, np_v, full_v in pairs:
        d.text((40, y), label, font=F_MAIN, fill=COL_TEXT)
        y += 26
        for name, v, col, vmax in [("drives w/o ladder", np_v,
                                    COL_DIM, 800),
                                   ("drives + ladder", full_v,
                                    COL_STOCK, 800)]:
            d.text((60, y), f"{name:18s}", font=F_SMALL, fill=col)
            d.rectangle([240, y + 2, 240 + 620, y + 14], fill=COL_BAR_BG)
            d.rectangle([240, y + 2, 240 + max(4, int(620 * v / vmax)),
                         y + 14], fill=col)
            d.text((870, y), str(v), font=F_MAIN, fill=col)
            y += 24
        y += 18
    y += 4
    d.line([(24, y), (W - 24, y)], fill=COL_BAR_BG, width=1)
    y += 14
    for ln, col in [
        ("spending stock under a held stock-goal reads as REGRESS -> the "
         "ladder agent TRAINED PLACEMENT OUT.", COL_TEXT),
        ("the blindness has a measured cost, not just a coverage gap.",
         COL_TEXT),
        ("generation 2's senses (table, pickaxe) turn the suppressed "
         "chain into a PAID chain — the fix is named in numbers.",
         COL_ARRIVE)]:
        d.text((40, y), ln, font=F_SMALL, fill=col)
        y += 22
    arr = np.asarray(img)
    Image.fromarray(arr).save(png)
    wr = FFmpegWriter(out, fps=FPS)
    for _ in range(int(hold_s * FPS)):
        wr.append_data(arr)
    wr.close()
    print(f"[act12] wrote {out}", flush=True)


def act_hack(steps=600, out="results/video/act9_hack_clip.mp4"):
    """The Goodhart clip: engineered reward (hacked: score climbs, zero
    laps) vs the register on the same gauge (races). Live counters."""
    from PIL import Image, ImageDraw
    from .boatrace_env import BoatRace
    from .experiments_v30 import BoatNet
    runs = []
    for label, fname, sub in [
            ("ENGINEERED REWARD  (hand-written)",
             "v30_policy_engineered_s1.pt",
             "found the cheat: oscillate on a checkpoint"),
            ("REGISTER  (same gauge, non-farmable)",
             "v30_policy_register_s1.pt",
             "cannot be paid to cheat -> races")]:
        net = BoatNet(seed=1)
        net.load_state_dict(torch.load(RESULTS / fname, map_location="cpu"))
        net.eval()
        env = BoatRace(seed=11)
        obs = env.reset()
        rec = {"frames": [], "laps": [], "eng": []}
        eng_total, laps_done = 0.0, 0.0
        torch.manual_seed(3)
        for t in range(steps):
            with torch.no_grad():
                logits, _ = net(torch.tensor(
                    np.asarray(obs, dtype=np.float32))[None])
                a = int(torch.distributions.Categorical(
                    logits=logits).sample())
            obs, r_eng, done, info = env.step(a)
            eng_total += r_eng
            fr = np.asarray(obs)
            if fr.ndim == 3 and fr.shape[0] in (1, 3):
                fr = np.transpose(fr, (1, 2, 0))
            if fr.dtype != np.uint8:
                fr = (np.clip(fr, 0, 1) * 255).astype(np.uint8)
            if fr.shape[-1] == 1:
                fr = np.repeat(fr, 3, -1)
            rec["frames"].append(fr)
            rec["laps"].append(laps_done + info["laps"])   # cumulative
            rec["eng"].append(eng_total)                   # cumulative
            if done:
                laps_done += info["laps"]
                obs = env.reset()
        runs.append((label, sub, rec))
    wr = FFmpegWriter(out, fps=FPS)
    for t in range(steps):
        cols = []
        for label, sub, rec in runs:
            game = _upscale(rec["frames"][t], 280)
            img = Image.new("RGB", (280, 280 + 64), COL_BG)
            img.paste(Image.fromarray(game), (0, 64))
            d = ImageDraw.Draw(img)
            d.text((8, 5), label[:36], font=F_SMALL, fill=COL_TEXT)
            d.text((8, 22), sub, font=F_SMALL, fill=COL_DIM)
            lap_col = COL_ARRIVE if rec["laps"][t] > 0 else COL_DIM
            d.text((8, 42), f"laps {rec['laps'][t]:.0f}    "
                   f"mis-specified score {rec['eng'][t]:.0f}",
                   font=F_SMALL, fill=lap_col)
            cols.append(np.asarray(img))
        row = np.concatenate(cols, axis=1)
        strip = Image.new("RGB", (row.shape[1], 22), COL_BG)
        ds = ImageDraw.Draw(strip)
        ds.text((8, 4), "red = boat · green = checkpoints · bottom bar "
                "= the progress gauge both rewards read", font=F_SMALL,
                fill=COL_DIM)
        wr.append_data(np.concatenate([row, np.asarray(strip)], axis=0))
    wr.close()
    print(f"[act9] wrote {out}", flush=True)


def goalswap_card(out="results/video/act10b_goalswap_card.mp4",
                  png="results/video/act10b_goalswap_card.png",
                  hold_s: float = 8.0):
    from PIL import Image, ImageDraw
    W, HH = 1000, 520
    img = Image.new("RGB", (W, HH), COL_BG)
    d = ImageDraw.Draw(img)
    d.text((24, 18), "EDIT A DESIRE — one line, measured consequences",
           font=F_HEAD, fill=COL_TEXT)
    d.text((24, 52), "the want is a vector. delete energy from it:",
           font=F_SMALL, fill=COL_DIM)
    d.rectangle([24, 76, W - 24, 112], fill=(24, 27, 36))
    d.text((40, 84), "mask = [health: 1, food: 1, drink: 1, "
           "energy: 0]   # the whole edit", font=F_MAIN, fill=COL_STOCK)
    rows = [
        ("sleeping per night-cycle", "4.3", "1.3 – 2.1",
         "the deleted desire's share collapses"),
        ("drink uptime", "0.88", "0.91 – 0.92",
         "untouched wants: untouched (the edit is surgical)"),
        ("residual sleep", "—", "health-motivated",
         "sleep was paid by TWO registers; the edit dissected it"),
    ]
    y = 140
    d.text((320, y), "energy wanted", font=F_SMALL, fill=COL_DIM)
    d.text((520, y), "energy deleted", font=F_SMALL, fill=COL_DIM)
    y += 24
    for label, a, b, note in rows:
        d.text((40, y), label, font=F_MAIN, fill=COL_TEXT)
        d.text((320, y), a, font=F_MAIN, fill=COL_VITAL)
        d.text((520, y), b, font=F_MAIN, fill=COL_ARRIVE)
        d.text((40, y + 22), note, font=F_SMALL, fill=COL_DIM)
        y += 58
    y += 6
    d.line([(24, y), (W - 24, y)], fill=COL_BAR_BG, width=1)
    y += 14
    for ln in ("wants you can read are wants you can EDIT —",
               "and edits land where they aim, with receipts."):
        d.text((40, y), ln, font=F_MAIN, fill=COL_TEXT)
        y += 26
    arr = np.asarray(img)
    Image.fromarray(arr).save(png)
    wr = FFmpegWriter(out, fps=FPS)
    for _ in range(int(hold_s * FPS)):
        wr.append_data(arr)
    wr.close()
    print(f"[act10b] wrote {out}", flush=True)


def math_card(out="results/video/act13_math_card.mp4",
              png="results/video/act13_math_card.png",
              hold_s: float = 10.0):
    """The telescoping theorem, on one card, with scope and audit."""
    from PIL import Image, ImageDraw
    W, HH = 1100, 460
    img = Image.new("RGB", (W, HH), COL_BG)
    d = ImageDraw.Draw(img)
    d.text((24, 18), "THE REWARD, IN THREE LINES", font=F_HEAD,
           fill=COL_TEXT)
    d.text((24, 46), "potential-based shaping identity (Ng, Harada, "
           "Russell 1999) — used as the ENTIRE reward, anchored to a "
           "held goal", font=F_SMALL, fill=COL_DIM)
    y = 92
    d.rectangle([24, y - 8, W - 24, y + 118], fill=(24, 27, 36))
    d.text((48, y), "potential   φ(t) = max(0,  g − m(t)) / s",
           font=F_MAIN, fill=COL_STOCK)
    d.text((48, y + 14), "            distance below the held target g,"
           " in calibrated units", font=F_SMALL, fill=COL_DIM)
    y += 44
    d.text((48, y), "reward      r(t) = φ(t−1) − φ(t)",
           font=F_MAIN, fill=COL_STOCK)
    d.text((48, y + 14), "            pay approach, charge retreat — "
           "every step, while the goal is held", font=F_SMALL,
           fill=COL_DIM)
    y += 44
    d.text((48, y), "held sum    r(1) + r(2) + … + r(T)  =  "
           "φ(commit) − φ(close)", font=F_MAIN, fill=COL_ARRIVE)
    d.text((48, y + 14), "            everything between cancels — the "
           "sum TELESCOPES", font=F_SMALL, fill=COL_DIM)
    y += 66
    for ln, col in [
        ("⇒ any loop, oscillation, or cycle sums to exactly ZERO — no "
         "sequence of movements mints reward", COL_TEXT),
        ("⇒ goals are held until arrival and settled exactly at close "
         "— switching goals mints nothing either", COL_TEXT)]:
        d.text((40, y), ln, font=F_MAIN, fill=col)
        y += 30
    y += 8
    d.line([(24, y), (W - 24, y)], fill=COL_BAR_BG, width=1)
    y += 14
    d.text((40, y), "verified against the implementation: 1,244 "
           "goal-holds across the experiments, every closed hold paid "
           "exactly", font=F_SMALL, fill=COL_ARRIVE)
    y += 20
    d.text((40, y), "φ(commit) − φ(close) to float precision; zero "
           "phantom payments. audit() ships in the repo — run it on "
           "any rollout.", font=F_SMALL, fill=COL_ARRIVE)
    y += 32
    d.text((40, y), "scope: the theorem covers the progress stream; "
           "one-shot frontier bonuses are bounded per life; sensors "
           "are a", font=F_SMALL, fill=COL_DIM)
    y += 20
    d.text((40, y), "separate trust problem — every instrument passes "
           "a held-out audit before the drive layer may read it.",
           font=F_SMALL, fill=COL_DIM)
    arr = np.asarray(img)
    Image.fromarray(arr).save(png)
    wr = FFmpegWriter(out, fps=FPS)
    for _ in range(int(hold_s * FPS)):
        wr.append_data(arr)
    wr.close()
    print(f"[act13] wrote {out}", flush=True)


def arch_card(out="results/video/act14_arch_card.mp4",
              png="results/video/act14_arch_card.png",
              hold_s: float = 10.0):
    """The four-box architecture card."""
    from PIL import Image, ImageDraw
    W, HH = 1100, 400
    img = Image.new("RGB", (W, HH), COL_BG)
    d = ImageDraw.Draw(img)
    d.text((24, 18), "THE DRIVE LAYER — frozen before training; the "
           "agent reads it, and cannot touch it", font=F_HEAD,
           fill=COL_TEXT)
    boxes = [
        ("SENSES", ["instruments calibrated", "once, then frozen;",
                    "audited before trusted"], COL_STOCK),
        ("WANTS", ["measurable targets,", "held: drink ≥ 8, wood ≥ 2.",
                   "readable · editable"], COL_VITAL),
        ("LEDGER", ["pays only measured", "progress; telescoping ⇒",
                    "exploits net zero"], COL_ARRIVE),
        ("PROPOSER", ["two fixed rules:", "maintain healthy ranges;",
                      "frontier once each"], (200, 160, 255)),
    ]
    bw, bh, gap, y0 = 236, 150, 34, 110
    x = 24
    for i, (name, lines, col) in enumerate(boxes):
        d.rounded_rectangle([x, y0, x + bw, y0 + bh], radius=10,
                            outline=col, width=3, fill=(22, 25, 33))
        d.text((x + 16, y0 + 14), name, font=F_MAIN, fill=col)
        for j, ln in enumerate(lines):
            d.text((x + 16, y0 + 48 + j * 24), ln, font=F_SMALL,
                   fill=COL_TEXT)
        if i < 3:
            ax = x + bw + 4
            d.text((ax, y0 + bh // 2 - 10), "→", font=F_HEAD,
                   fill=COL_DIM)
        x += bw + gap
    y = y0 + bh + 46
    for ln in ("nothing in this layer is trained · nothing changes "
               "while the agent learns",
               "the policy underneath is ordinary RL — the only thing "
               "that is architecture is what the agent WANTS"):
        d.text((40, y), ln, font=F_MAIN, fill=COL_DIM)
        y += 30
    arr = np.asarray(img)
    Image.fromarray(arr).save(png)
    wr = FFmpegWriter(out, fps=FPS)
    for _ in range(int(hold_s * FPS)):
        wr.append_data(arr)
    wr.close()
    print(f"[act14] wrote {out}", flush=True)


def vla_bridge_card(out="results/video/act15_vla_bridge.mp4",
                    png="results/video/act15_vla_bridge.png",
                    hold_s: float = 10.0):
    """Two columns: the VLA/world-model stack | this layer on top."""
    from PIL import Image, ImageDraw
    W, HH = 1160, 470
    img = Image.new("RGB", (W, HH), COL_BG)
    d = ImageDraw.Draw(img)
    d.text((24, 18), "WORLD MODELS SOLVE PERCEPTION AND SKILL. "
           "THIS LAYER IS THE OBJECTIVE THAT SITS ON TOP.",
           font=F_HEAD, fill=COL_TEXT)
    colw = (W - 72) // 2
    for i, (title, col, lines) in enumerate([
        ("YOUR STACK  (video world model / VLA)", COL_STOCK, [
            "video-pretrained latents carry objects,",
            "contact, consequence — that is why they",
            "generalize: less robot data, transfer",
            "across tasks (mimic: frozen video backbone",
            "+ small action decoders, competitive",
            "manipulation on standard benchmarks)",
            "",
            "objective still comes from demos, task",
            "specs, or hand rewards"]),
        ("THIS LAYER  (drops on your latents)", COL_ARRIVE, [
            "instrument audit: which quantities are",
            "honestly readable from YOUR latent —",
            "one notebook, any checkpoint",
            "",
            "wants · unfarmable ledger · proposer run",
            "over whatever passes — unchanged code",
            "",
            "flinch: benched on my substrate (forward",
            "model failed its action audit) — an",
            "action-conditioned video predictor is",
            "exactly what that audit wants"])]):
        x = 24 + i * (colw + 24)
        d.rounded_rectangle([x, 66, x + colw, HH - 24], radius=10,
                            outline=col, width=3, fill=(22, 25, 33))
        d.text((x + 18, 80), title, font=F_MAIN, fill=col)
        y = 116
        for ln in lines:
            d.text((x + 18, y), ln, font=F_SMALL, fill=COL_TEXT
                   if ln else COL_DIM)
            y += 24
    arr = np.asarray(img)
    Image.fromarray(arr).save(png)
    wr = FFmpegWriter(out, fps=FPS)
    for _ in range(int(hold_s * FPS)):
        wr.append_data(arr)
    wr.close()
    print(f"[act15] wrote {out}", flush=True)


def requirements_card(out="results/video/act17_requirements.mp4",
                      png="results/video/act17_requirements.png",
                      hold_s: float = 12.0):
    """Part 3: what a system needs for this layer to fit."""
    from PIL import Image, ImageDraw
    W, HH = 1100, 560
    img = Image.new("RGB", (W, HH), COL_BG)
    d = ImageDraw.Draw(img)
    d.text((24, 18), "WHAT A SYSTEM NEEDS FOR THIS LAYER TO FIT",
           font=F_HEAD, fill=COL_TEXT)
    d.text((24, 46), "judge your own stack against the list",
           font=F_SMALL, fill=COL_DIM)
    reqs = [
        ("1  measurable channels", COL_STOCK,
         "quantities readable as numbers — telemetry, counters, "
         "meters, levels. if it has a gauge, it can be a channel."),
        ("2  direction that means something", COL_STOCK,
         "per channel: more is better, or a range is healthy — that "
         "is what makes a want expressible as a held target."),
        ("3  any learner underneath", COL_STOCK,
         "the layer only supplies reward; it ran unchanged over "
         "different learners across these experiments."),
        ("4  the ability to freeze", COL_VITAL,
         "sensors calibrated from logs or a calibration run BEFORE "
         "training, untouched after. labeled history (good/bad "
         "episodes) can mint the goal roster: labels define what "
         "pays; only measured reality pays."),
        ("5  optional: an action-conditioned predictor", COL_ARRIVE,
         "unlocks the flinch (veto actions with dangerous predicted "
         "outcomes). the audit that decides whether a predictor "
         "earns the veto ships in the repo."),
    ]
    y = 86
    for title, col, body in reqs:
        d.text((40, y), title, font=F_MAIN, fill=col)
        y += 24
        # naive wrap at ~92 chars
        line = ""
        for word in body.split():
            if len(line) + len(word) + 1 > 92:
                d.text((64, y), line, font=F_SMALL, fill=COL_TEXT)
                y += 20
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            d.text((64, y), line, font=F_SMALL, fill=COL_TEXT)
            y += 20
        y += 14
    d.line([(24, HH - 44), (W - 24, HH - 44)], fill=COL_BAR_BG, width=1)
    d.text((40, HH - 34), "whether it delivers value on any particular "
           "system is the untested part — the experiments are the "
           "evidence that exists", font=F_SMALL, fill=COL_DIM)
    arr = np.asarray(img)
    Image.fromarray(arr).save(png)
    wr = FFmpegWriter(out, fps=FPS)
    for _ in range(int(hold_s * FPS)):
        wr.append_data(arr)
    wr.close()
    print(f"[act17] wrote {out}", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "frames"
    (RESULTS / "video").mkdir(exist_ok=True)
    if mode == "trace":
        act_trace()
    elif mode == "creatures":
        act_creatures()
    elif mode == "card":
        three_worlds_card()
    elif mode == "goalswap":
        act_goalswap()
    elif mode == "cards2":
        verdict_card()
        reversal_card()
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
