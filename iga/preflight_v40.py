"""v4.0 pre-flight (ALL LOCAL — pods only for the final fleet).

Step A (this file, mode 'counters'): inventory-counter routing at full
scale, locally — closed-form heads need no pods.
  A1  extended walk recording inventory truth (wood, sapling, stone)
  A2  coverage audit: which counters VARY under random play (stone
      needs a pickaxe chain -> expected unroutable; documented, not
      hidden)
  A3  exclusive-change differencing -> counter digit windows
  A4  closed-form heads + held-out routing gates (>= 0.9 each varying
      counter)

Later steps (separate modes): 'harness' scripted-chain mechanics audit,
'audit' proposal-agenda inspection, 'reach' fixed-target reachability,
'rehearsal' one-seed full dress run.
"""

from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch

from .crafter_support import HUD_ROWS, closed_form_head, measure_rho
from .experiments import RESULTS

ITEMS = ("wood", "sapling", "stone")


def collect_inventory_walk(steps: int, seed: int, phase_random: bool = True):
    import crafter
    rng = np.random.default_rng(seed)
    env = crafter.Env(seed=seed)
    obs = env.reset()
    if phase_random:
        env._step = int(rng.integers(0, 300))
    frames, ep_ids = [], []
    truth = {k: [] for k in ITEMS + ("food", "drink", "daylight")}
    ep = 0
    for t in range(steps):
        # collection-biased walk (discovery protocol, truth-free): 35%
        # 'do' presses, rest random — random play barely chops (wood
        # changed 125x in 40k steps; counters need events to be
        # discoverable). Same legitimacy class as phase randomization.
        u = rng.random()
        if u < 0.40:
            a = 5                                   # 'do' (chop/collect)
        elif u < 0.70 and t > 0:
            pass                                    # repeat last move —
            # persistence reaches trees that pure jitter never does
        else:
            a = int(rng.integers(1, 5))             # a movement action
        obs, r, done, info = env.step(a)
        frames.append(torch.from_numpy(obs.copy()).permute(2, 0, 1))
        inv = info["inventory"]
        for k in ITEMS:
            truth[k].append(float(inv.get(k, 0)))
        truth["food"].append(float(inv.get("food", 0)))
        truth["drink"].append(float(inv.get("drink", 0)))
        truth["daylight"].append(float(env._world.daylight))
        ep_ids.append(ep)
        if done:
            ep += 1
            obs = env.reset()
            if phase_random:
                env._step = int(rng.integers(0, 300))
    return (torch.stack(frames), {k: torch.tensor(v) for k, v in truth.items()},
            torch.tensor(ep_ids))


def step_a() -> None:
    t0 = time.time()
    frames, truth, ep_ids = collect_inventory_walk(60_000, seed=0)
    frames_t, truth_t, ep_t = collect_inventory_walk(15_000, seed=77)
    print(f"[pf-A] walk {frames.shape} ({time.time()-t0:.0f}s)", flush=True)

    # A2 coverage audit
    varying = []
    for k in ITEMS:
        mx, changes = float(truth[k].max()), int((truth[k][1:] != truth[k][:-1]).sum())
        print(f"[pf-A] coverage {k}: max {mx:.0f}, changes {changes}", flush=True)
        if changes >= 200:
            varying.append(k)
    print(f"[pf-A] routable candidates: {varying} "
          f"(others documented as coverage-excluded)", flush=True)

    # A3 exclusive-change differencing over the full frame (counters may
    # render outside the vitals strip)
    d = {k: (truth[k][1:] - truth[k][:-1]) for k in varying}
    others = {k: (truth[k][1:] - truth[k][:-1])
              for k in ("food", "drink")}
    same = (ep_ids[1:] == ep_ids[:-1])
    windows = {}
    for k in varying:
        m = same & (d[k].abs() >= 1)
        for j, dv in d.items():
            if j != k:
                m = m & (dv == 0)
        for j, dv in others.items():
            m = m & (dv == 0)
        if int(m.sum()) < 15:
            print(f"[pf-A] {k}: too few exclusive events ({int(m.sum())})",
                  flush=True)
            continue
        idx = m.nonzero().squeeze(1)
        diff = (frames[idx + 1].float() - frames[idx].float()).abs().mean(dim=(0, 1))
        hot = (diff > diff.max() * 0.35).nonzero()
        r0, r1 = int(hot[:, 0].min()), int(hot[:, 0].max()) + 1
        c0, c1 = int(hot[:, 1].min()), int(hot[:, 1].max()) + 1
        windows[k] = ((r0, r1), (c0, c1))
        print(f"[pf-A] {k}: n={int(m.sum())} window rows[{r0},{r1}) "
              f"cols[{c0},{c1})", flush=True)

    # A4 closed-form routing on held-out
    rho = measure_rho(truth["wood"], ep_ids, 20) if "wood" in varying else 0.9
    tau = -20 / math.log(max(min(float(rho), 0.995), 0.05))
    gates = {}
    for k, ((r0, r1), (c0, c1)) in windows.items():
        X = frames[:, :, r0:r1, c0:c1].float().div(255.0).flatten(1)
        Xt = frames_t[:, :, r0:r1, c0:c1].float().div(255.0).flatten(1)
        w, b = closed_form_head(X, ep_ids, 20, tau)
        z = Xt @ w + b
        c = abs(float(torch.corrcoef(torch.stack([z, truth_t[k]]))[0, 1])) \
            if float(truth_t[k].std()) > 1e-6 else float("nan")
        gates[k] = c
        print(f"[pf-A] routing {k}: {c:.3f} "
              f"({'PASS' if c >= 0.9 else 'FAIL'} vs 0.9)", flush=True)
    (RESULTS / "v40_preflight_A.json").write_text(json.dumps(
        {"varying": varying, "windows": {k: list(map(list, v))
                                         for k, v in windows.items()},
         "routing": gates}, indent=2))
    print(f"[pf-A] done {time.time()-t0:.0f}s", flush=True)


# ====================================================================
# Step B: the scripted-chain mechanics harness ('harness' mode).
# Audits the goal machine's ACCOUNTING before any learner touches it:
#   B1 telescoping exactness   every closed hold pays exactly
#                              w*(phi_commit - phi_close)
#   B2 mint bound              max prefix of pay <= w*phi_commit
#   B3 anti-farm               oscillation nets zero; chop-place cycles
#                              pay once (C6+C7-gain+C1)
#   B4 death boundary          claims settle at the terminal state, no
#                              pay across reset
#   B5 agenda live             on a real Crafter walk: frontier arrivals
#                              AND vitals restores both occur
#   B6 instrument loop         same audits with the CALIBRATED heads in
#                              the loop; arrival events agree with the
#                              truth-driven machine
# Synthetic sequences give exact expectations; the env walk gives
# integration under real mechanics (placement consumption, night,
# death). All local, $0.
# ====================================================================

from .goal_machine import (CHANNELS, GMConfig, GoalMachine,
                           calibrate_instrument, channel_windows,
                           extract_window)


def _walk_rule(rng, A, place_p: float, plant_p: float, sleep_p: float):
    """The B/C/F walk protocol as a closure: persistent collection +
    consumption/sleep overrides. One definition, every collector."""
    state = {"a": 0}

    def pick(t: int, wood: float, sapling: float, energy: float) -> int:
        u = rng.random()
        if u < 0.40:
            state["a"] = A["do"]
        elif u < 0.70 and t > 0:
            pass                                    # repeat last action
        else:
            state["a"] = int(rng.integers(1, 5))
        u2 = rng.random()
        if wood >= 1 and u2 < place_p:
            state["a"] = A["place_table"]
        elif sapling >= 1 and u2 < place_p + plant_p:
            state["a"] = A["place_plant"]
        elif energy < 4 and rng.random() < sleep_p:
            state["a"] = A["sleep"]
        return state["a"]

    return pick


def collect_harness_walk(steps: int, seed: int, stock_windows: dict,
                         place_p: float = 0.20, plant_p: float = 0.15,
                         sleep_p: float = 0.30):
    """Persistent-collection walk (the A protocol) + consumption rules:
    place a table when wood is held, plant a sapling when held, sleep
    when energy sags. Gives the walk real consume events and full
    vitals dynamics. Records per-channel WINDOW pixels (not frames),
    truth in truth units, dones, and consume events."""
    import crafter
    rng = np.random.default_rng(seed)
    env = crafter.Env(seed=seed)
    obs = env.reset()
    env._step = int(rng.integers(0, 300))
    names = list(env.action_names)
    A = {n: i for i, n in enumerate(names)}
    for n in ("do", "sleep", "place_table", "place_plant"):
        assert n in A, f"action '{n}' not in {names}"
    pick = _walk_rule(rng, A, place_p, plant_p, sleep_p)
    spec = channel_windows(stock_windows)
    wins = {k: [] for k in CHANNELS}
    truth = {k: [] for k in CHANNELS}
    dones, consume_events = [], []
    prev_wood = 0.0
    for t in range(steps):
        inv_wood = prev_wood
        a = pick(t, inv_wood,
                 truth["sapling"][-1] if truth["sapling"] else 0.0,
                 truth["energy"][-1] if truth["energy"] else 9.0)
        obs, r, done, info = env.step(a)
        inv = info["inventory"]
        frame = torch.from_numpy(obs.copy()).permute(2, 0, 1)
        for k in CHANNELS:
            truth[k].append(float(inv.get(k, 0)))
            wins[k].append(extract_window(frame, spec[k]))
        if a == A["place_table"] and inv.get("wood", 0) < inv_wood:
            consume_events.append(t)
        prev_wood = float(inv.get("wood", 0))
        dones.append(bool(done))
        if done:
            obs = env.reset()
            env._step = int(rng.integers(0, 300))
            prev_wood = 0.0
    return ({k: torch.stack(v) for k, v in wins.items()},
            {k: torch.tensor(v) for k, v in truth.items()},
            torch.tensor(dones), consume_events)


def _m(health=9.0, food=9.0, drink=9.0, energy=9.0, wood=0.0, sapling=0.0):
    return torch.tensor([health, food, drink, energy, wood, sapling])


def _drive(gm: GoalMachine, M: torch.Tensor, dones: torch.Tensor):
    """Run a machine over recorded truth rows; returns per-step rewards.
    The first row of each life seeds reset (its true m0 is one step
    earlier and unrecorded — an acceptable audit approximation)."""
    rewards = np.zeros(M.shape[0])
    fresh = True
    for t in range(M.shape[0]):
        if fresh:
            gm.t = t          # keep the machine clock on walk-row indices
            gm.reset(M[t])    # (before reset: reset's commits stamp t)
            fresh = False
            continue
        r, _ = gm.step(M[t], done=bool(dones[t]))
        rewards[t] = r
        if bool(dones[t]):
            fresh = True
    return rewards


def _audit_ledger(led: list[dict], tol: float = 1e-5):
    exact = all(abs(h["paid"] - h["w"] * (h["phi_commit"] - h["phi_close"]))
                < tol for h in led)
    bound = all(h["max_prefix"] <= h["w"] * h["phi_commit"] + tol for h in led)
    return exact, bound


def harness() -> None:
    t0 = time.time()
    pa = json.loads((RESULTS / "v40_preflight_A.json").read_text())
    stock_windows = {k: tuple(map(tuple, v)) for k, v in pa["windows"].items()}

    # ---------------- synthetic exactness suite (no env, exact numbers)
    cfg = GMConfig(stds=(1.0,) * 6, w_bands=(1.0, 2.0), holds=(6, 180))
    gm = GoalMachine(cfg)

    print("[pf-B] S1 chain + dissolved trap", flush=True)
    gm.reset(_m())                       # expects: commit wood>=1
    r1, _ = gm.step(_m(wood=1))          # climb + arrive + bonus
    r2, _ = gm.step(_m(wood=0))          # place table: wood UNHELD now
    assert abs(r1 - 3.0) < 1e-6, r1      # 2*(1-0) + bonus 1
    assert abs(r2) < 1e-6, r2            # trap dissolved by rotation
    s1 = (r1, r2)

    print("[pf-B] S1b held-target consumption is charged but telescoped",
          flush=True)
    # Force the residual trap case: register frees while wood sits at 1,
    # so the NOVEL frontier wood>=2 commits — then a placement under that
    # hold is charged as real measured regress, and telescoping bounds it.
    gmb = GoalMachine(cfg)
    gmb.reset(_m())                          # wood>=1 held
    gmb.step(_m(wood=1))                     # arrive; sapling>=1 commits
    gmb.step(_m(wood=1, sapling=1))          # sapling arrives; wood>=2
    #   commits (wood=1 -> t=2 is a novel cell; phi_commit = 1)
    r5, _ = gmb.step(_m(wood=0, sapling=1))  # place table: CHARGED -w*1
    r6, _ = gmb.step(_m(wood=1, sapling=1))  # re-chop +w*1
    r7, _ = gmb.step(_m(wood=2, sapling=1))  # +w*1, arrive, +bonus
    led = [h for h in gmb.ledger
           if h["channel"] == "wood" and h["target"] == 2.0]
    assert len(led) == 1 and abs(led[0]["paid"] - 2.0 * 1.0) < 1e-6, led
    assert abs(r5 + 2.0) < 1e-6, r5          # the dip, exactly -w*1
    assert abs(r7 - 3.0) < 1e-6, r7          # climb + one-shot bonus
    assert led[0]["max_prefix"] <= 2.0 * led[0]["phi_commit"] + 1e-6

    print("[pf-B] S2 oscillation nets zero", flush=True)
    gm2 = GoalMachine(cfg)
    gm2.reset(_m(drink=5.0))             # commit drink>=8
    osc = []
    for i in range(6):                   # 3 full 5->7->5 cycles, hold=6
        osc.append(gm2.step(_m(drink=7.0 if i % 2 == 0 else 5.0))[0])
    assert abs(sum(osc)) < 1e-6, osc     # closes by timeout at phi parity
    tl = [h for h in gm2.ledger if h["channel"] == "drink"]
    assert tl and tl[-1]["closed_by"] == "timeout" \
        and abs(tl[-1]["paid"]) < 1e-6

    print("[pf-B] S3 death settles at terminal state", flush=True)
    gm3 = GoalMachine(cfg)
    gm3.reset(_m(drink=5.0))
    rd, _ = gm3.step(_m(drink=4.0), done=True)
    hd = [h for h in gm3.ledger if h["channel"] == "drink"][-1]
    assert abs(rd + 1.0) < 1e-6 and hd["closed_by"] == "death" \
        and abs(hd["paid"] + 1.0) < 1e-6
    assert all(h["closed_by"] == "death" for h in gm3.ledger)
    gm3.reset(_m())                      # fresh life
    assert gm3.arrived_cells == set() and gm3.holds[1] is not None

    print("[pf-B] S4 chop-place farming pays once", flush=True)
    gm4 = GoalMachine(cfg)
    gm4.reset(_m())
    total = gm4.step(_m(wood=1))[0]      # arrive (4,1): +2+1; sapling commits
    farm = 0.0
    for i in range(10):
        farm += gm4.step(_m(wood=0.0 if i % 2 == 0 else 1.0))[0]
    assert abs(total - 3.0) < 1e-6 and abs(farm) < 1e-6, (total, farm)

    print("[pf-B] S5 vitals maintenance re-pays honestly", flush=True)
    gm5 = GoalMachine(cfg)
    gm5.reset(_m(drink=5.0))
    pays = []
    for _ in range(3):
        pays.append(gm5.step(_m(drink=8.0))[0])
        gm5.step(_m(drink=5.0))          # env-clocked sag; recommit
    assert all(abs(p - 3.0) < 1e-6 for p in pays), pays
    print(f"[pf-B] synthetic suite PASS ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- calibration + env walks (cached: deterministic)
    import tempfile
    from pathlib import Path
    cache_dir = Path(tempfile.gettempdir()) / "iga_v40_walks"
    cache_dir.mkdir(exist_ok=True)

    def _walk(steps, seed, place_p, plant_p):
        p = cache_dir / f"walk_{steps}_{seed}_{place_p}_{plant_p}.pt"
        if p.exists():
            d = torch.load(p, weights_only=False)
            return d["wins"], d["truth"], d["dones"], d["consume"]
        out = collect_harness_walk(steps, seed=seed,
                                   stock_windows=stock_windows,
                                   place_p=place_p, plant_p=plant_p)
        torch.save({"wins": out[0], "truth": out[1], "dones": out[2],
                    "consume": out[3]}, p)
        return out

    # Calibration walk = DISCOVERY protocol (collection, no consumption:
    # planting away the saplings starved the head solve — corr swung
    # 0.91 -> 0.34 across collections). Audit walk = DEPLOYMENT-like
    # (consumption on). Same split as phase-1: the discoverer chooses
    # its coverage; the frozen instrument is then audited under field
    # dynamics.
    print("[pf-B] collecting calib walk 50k + audit walk 25k", flush=True)
    wins_c, truth_c, dones_c, _ = _walk(50_000, 0, 0.0, 0.0)
    ep_c = torch.cumsum(torch.cat([torch.tensor([0]), dones_c[:-1].long()]), 0)
    wins_a, truth_a, dones_a, consume_a = _walk(25_000, 101, 0.20, 0.15)
    print(f"[pf-B] walks done ({time.time()-t0:.0f}s); "
          f"audit consume events: {len(consume_a)}", flush=True)

    inst = calibrate_instrument(wins_c, truth_c, ep_c, wins_a, truth_a)
    for k in CHANNELS:
        ch = inst["channels"][k]
        print(f"[pf-B] instrument {k:8s} corr {ch['corr_holdout']:.3f} "
              f"MAE {ch['mae_holdout']:.3f} tau {ch['tau']:.0f}", flush=True)
    torch.save({"inst": inst, "stock_windows": stock_windows},
               RESULTS / "v40_instrument.pt")

    stds = tuple(inst["channels"][k]["std"] for k in CHANNELS)
    # w ∝ hold length (spec form; walk-rho tau is invalid on counters —
    # first harness run measured stocks at 0.1x vitals, backwards)
    cfg_real = GMConfig(stds=stds)
    cfg_real = GMConfig(stds=stds, w_bands=(
        1.0, cfg_real.holds[1] / cfg_real.holds[0]))
    print(f"[pf-B] w_bands (hold ratio): {cfg_real.w_bands}  "
          f"eps {cfg_real.arrive_eps}", flush=True)

    # ---------------- env audit, truth-driven
    M_true = torch.stack([truth_a[k] for k in CHANNELS], dim=1)
    gmt = GoalMachine(cfg_real)
    _drive(gmt, M_true, dones_a)
    exact_t, bound_t = _audit_ledger(gmt.ledger)
    arr_frontier = [h for h in gmt.ledger
                    if h["frontier"] and h["closed_by"] == "arrive"]
    arr_vitals = [h for h in gmt.ledger
                  if h["band"] == 0 and h["closed_by"] == "arrive"]
    cells = {}
    for h in arr_frontier:
        cells[(h["channel"], h["target"])] = \
            cells.get((h["channel"], h["target"]), 0) + 1
    print(f"[pf-B] env(truth): holds {len(gmt.ledger)}, exact {exact_t}, "
          f"bound {bound_t}", flush=True)
    print(f"[pf-B] env(truth): frontier arrivals {len(arr_frontier)} "
          f"{dict(list(cells.items())[:8])}, vitals restores "
          f"{len(arr_vitals)}", flush=True)

    # ---------------- instrument in the loop
    M_hat = torch.zeros_like(M_true)
    for i, k in enumerate(CHANNELS):
        ch = inst["channels"][k]
        M_hat[:, i] = ch["a"] * (wins_a[k] @ ch["w"] + ch["b"]) + ch["c"]
    gmi = GoalMachine(cfg_real)
    _drive(gmi, M_hat, dones_a)
    exact_i, bound_i = _audit_ledger(gmi.ledger)

    # Deployment-faithful verification: ONE machine runs closed-loop on
    # the heads (as the fleet will); every EVENT it declares is checked
    # against ground truth at that timestep. Corruption (paying for
    # things that didn't happen) must be ~0; blindness (missing things
    # that did) is inefficiency, quantified and tolerated in bounds.
    i_arr = [h for h in gmi.ledger if h["closed_by"] == "arrive"]
    i_frontier_arr = [h for h in i_arr if h["frontier"]]
    phantom = [h for h in i_frontier_arr
               if float(truth_a[h["channel"]][min(h["t_close"],
                        len(dones_a) - 1)]) < h["target"] - 0.5]
    shortfalls = [max(0.0, h["target"] - 0.75
                      - float(truth_a[h["channel"]][min(h["t_close"],
                              len(dones_a) - 1)]))
                  for h in i_arr if not h["frontier"]]
    p95_short = float(np.percentile(shortfalls, 95)) if shortfalls else 0.0
    missed = []
    for h in gmi.ledger:
        if h["closed_by"] != "timeout":
            continue
        h0, h1 = h["t_commit"] + 1, min(h["t_close"], len(dones_a) - 1)
        eps_b = cfg_real.arrive_eps[h["band"]]
        seg = truth_a[h["channel"]][h0:h1 + 1]
        if seg.numel() and float(seg.max()) >= h["target"] - eps_b:
            missed.append(h)
    n_timeout = sum(1 for h in gmi.ledger if h["closed_by"] == "timeout")
    phantom_rate = len(phantom) / max(len(i_frontier_arr), 1)
    missed_rate = len(missed) / max(len(gmi.ledger), 1)
    print(f"[pf-B] env(inst): holds {len(gmi.ledger)}, exact {exact_i}, "
          f"bound {bound_i}; arrivals {len(i_arr)} "
          f"(frontier {len(i_frontier_arr)}, phantom {len(phantom)} "
          f"rate {phantom_rate:.3f}); vitals-arrival p95 shortfall "
          f"{p95_short:.2f}; timeouts {n_timeout} blind-missed "
          f"{len(missed)} rate {missed_rate:.3f}", flush=True)

    # ---------------- trace excerpt (first life, truth machine)
    first_death = int(dones_a.nonzero()[0]) if dones_a.any() else 400
    print("[pf-B] --- proposal trace, first life ---", flush=True)
    for (t, evn, band, ch, lvl) in gmt.trace:
        if t > first_death:
            break
        print(f"[pf-B]   t={t:4d} {'STOCK' if band else 'VITAL'} "
              f"{evn:7s} {ch}>={lvl:.0f}", flush=True)

    gates = {
        "B1_exact": bool(exact_t and exact_i),
        "B2_bound": bool(bound_t and bound_i),
        "B3_antifarm": True,      # asserted above (S2/S4)
        "B4_death": True,         # asserted above (S3)
        "B5_agenda": bool(len(arr_frontier) >= 1 and len(arr_vitals) >= 1),
        "B6_no_phantom": bool(phantom_rate <= 0.05 and p95_short <= 1.0),
        "B7_blindness": bool(missed_rate <= 0.20),
    }
    (RESULTS / "v40_preflight_B.json").write_text(json.dumps({
        "instrument": {k: {kk: vv for kk, vv in inst["channels"][k].items()
                           if kk not in ("w",)} for k in CHANNELS},
        "w_bands": list(cfg_real.w_bands),
        "arrive_eps": list(cfg_real.arrive_eps),
        "env_truth": {"holds": len(gmt.ledger),
                      "frontier_arrivals": len(arr_frontier),
                      "vitals_restores": len(arr_vitals),
                      "cells": {f"{c}>={g:.0f}": n
                                for (c, g), n in cells.items()},
                      "consume_events": len(consume_a)},
        "env_inst": {"holds": len(gmi.ledger),
                     "arrivals": len(i_arr),
                     "frontier_arrivals": len(i_frontier_arr),
                     "phantom_rate": phantom_rate,
                     "vitals_p95_shortfall": p95_short,
                     "timeouts": n_timeout,
                     "blind_missed_rate": missed_rate},
        "gates": gates}, indent=2))
    ok = all(gates.values())
    print(f"[pf-B] GATES: " + "  ".join(
        f"{k} {'PASS' if v else 'FAIL'}" for k, v in gates.items()),
        flush=True)
    print(f"[pf-B] {'ALL PASS' if ok else 'FAILURES PRESENT'} "
          f"({time.time()-t0:.0f}s)", flush=True)


# ====================================================================
# Step C: proposal-agenda statistics ('audit' mode). The machine is
# correct (B); C asks whether its AGENDA has the shape a learner can
# live inside: how often registers are held, how dense the reward
# stream is, which goals commit/arrive/time out, what the proposer
# rejects and why. Runs on the cached B walks; seconds, $0.
# ====================================================================

def audit() -> None:
    import tempfile
    from pathlib import Path
    t0 = time.time()
    blob = torch.load(RESULTS / "v40_instrument.pt", weights_only=False)
    inst = blob["inst"]
    cache = (Path(tempfile.gettempdir()) / "iga_v40_walks"
             / "walk_25000_101_0.2_0.15.pt")
    d = torch.load(cache, weights_only=False)
    wins_a, truth_a, dones_a = d["wins"], d["truth"], d["dones"]
    N = int(dones_a.shape[0])
    M_hat = torch.zeros(N, len(CHANNELS))
    for i, k in enumerate(CHANNELS):
        ch = inst["channels"][k]
        M_hat[:, i] = ch["a"] * (wins_a[k] @ ch["w"] + ch["b"]) + ch["c"]

    stds = tuple(inst["channels"][k]["std"] for k in CHANNELS)
    cfg = GMConfig(stds=stds)
    cfg = GMConfig(stds=stds, w_bands=(1.0, cfg.holds[1] / cfg.holds[0]))
    gm = GoalMachine(cfg)
    rewards = _drive(gm, M_hat, dones_a)

    # hold coverage per band (fraction of steps with an open hold)
    cov = [0, 0]
    for h in gm.ledger:
        cov[h["band"]] += h["t_close"] - h["t_commit"]
    cov = [c / N for c in cov]
    nz = np.abs(rewards) > 1e-9
    density = float(nz.mean())
    pos = rewards[rewards > 1e-9]
    neg = rewards[rewards < -1e-9]

    table = {}
    for h in gm.ledger:
        key = f"{h['channel']}>={h['target']:.0f}"
        row = table.setdefault(key, {"commits": 0, "arrive": 0,
                                     "timeout": 0, "death": 0, "lens": []})
        row["commits"] += 1
        row[h["closed_by"]] += 1
        row["lens"].append(h["t_close"] - h["t_commit"])
    for row in table.values():
        row["median_len"] = int(np.median(row.pop("lens")))

    # richest life (most arrivals) — the trace preview
    lives, start = [], 0
    for t in range(N):
        if bool(dones_a[t]):
            lives.append((start, t))
            start = t + 1
    arr_times = [h["t_close"] for h in gm.ledger
                 if h["closed_by"] == "arrive"]
    best_life = max(lives, key=lambda ab: sum(1 for t in arr_times
                                              if ab[0] <= t <= ab[1]))
    print(f"[pf-C] lives {len(lives)}, stock-hold coverage {cov[1]:.2f}, "
          f"vitals coverage {cov[0]:.2f}", flush=True)
    print(f"[pf-C] reward stream: density {density:.3f}, +mean "
          f"{pos.mean() if pos.size else 0:.3f} (n {pos.size}), -mean "
          f"{neg.mean() if neg.size else 0:.3f} (n {neg.size}), bonus "
          f"total {gm.total_bonus:.0f}", flush=True)
    print(f"[pf-C] rejects: {gm.rejects}", flush=True)
    for k, row in sorted(table.items()):
        print(f"[pf-C]   {k:12s} {row}", flush=True)
    print(f"[pf-C] --- richest life t=[{best_life[0]},{best_life[1]}] ---",
          flush=True)
    for (t, evn, band, chn, lvl) in gm.trace:
        if best_life[0] <= t <= best_life[1]:
            print(f"[pf-C]   t={t:5d} {'STOCK' if band else 'VITAL'} "
                  f"{evn:7s} {chn}>={lvl:.0f}", flush=True)

    frontier_lives = sum(1 for (a, b) in lives
                         if any(a <= t <= b for t in arr_times))
    gates = {"C1_coverage": bool(cov[1] >= 0.5),
             "C2_density": bool(density >= 0.01)}
    (RESULTS / "v40_preflight_C.json").write_text(json.dumps({
        "lives": len(lives), "coverage": cov, "density": density,
        "pos_n": int(pos.size), "neg_n": int(neg.size),
        "bonus_total": gm.total_bonus, "rejects": gm.rejects,
        "frontier_life_frac": frontier_lives / max(len(lives), 1),
        "table": table, "gates": gates}, indent=2))
    print(f"[pf-C] GATES: " + "  ".join(
        f"{k} {'PASS' if v else 'FAIL'}" for k, v in gates.items())
        + f"  ({time.time()-t0:.0f}s)", flush=True)


# ====================================================================
# Step F: the flinch forward model ('forward' mode). C4 needs a frozen
# one-step predictor; this trains, freezes, and AUDITS it. Two gates:
#   F1 drop-AUC       predicted next-step health delta must rank real
#                     health drops (>= 0.70 holdout AUC)
#   F2 discrimination flinch only matters if the prediction MOVES with
#                     the action: starvation is action-blind, combat is
#                     not. On holdout drop events, the executed action's
#                     predicted drop must exceed the alternatives' mean
#                     (fraction >= 0.55).
# FAIL -> flinch deferred per SPEC C4 (documented), fleet runs 3 arms.
# ====================================================================

def collect_forward_walk(steps: int, seed: int, place_p: float = 0.20,
                         plant_p: float = 0.15, sleep_p: float = 0.30):
    """Frames + executed actions + vitals truth (same walk rule as B/C)."""
    import crafter
    rng = np.random.default_rng(seed)
    env = crafter.Env(seed=seed)
    obs = env.reset()
    env._step = int(rng.integers(0, 300))
    A = {n: i for i, n in enumerate(env.action_names)}
    pick = _walk_rule(rng, A, place_p, plant_p, sleep_p)
    frames, acts, dones = [], [], []
    truth = {k: [] for k in ("health", "food", "drink", "energy")}
    prev = {"wood": 0.0, "sapling": 0.0, "energy": 9.0}
    for t in range(steps):
        a = pick(t, prev["wood"], prev["sapling"], prev["energy"])
        obs, r, done, info = env.step(a)
        inv = info["inventory"]
        frames.append(torch.from_numpy(obs.copy()).permute(2, 0, 1))
        acts.append(a)
        for k in truth:
            truth[k].append(float(inv.get(k, 0)))
        prev = {k: float(inv.get(k, 0)) for k in ("wood", "sapling", "energy")}
        dones.append(bool(done))
        if done:
            obs = env.reset()
            env._step = int(rng.integers(0, 300))
            prev = {"wood": 0.0, "sapling": 0.0, "energy": 9.0}
    return (torch.stack(frames), {k: torch.tensor(v) for k, v in truth.items()},
            torch.tensor(acts), torch.tensor(dones))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels > 0].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def forward_model() -> None:
    from .crafter_support import EncoderCrafter, ema_slow_stats
    t0 = time.time()
    pa = json.loads((RESULTS / "v40_preflight_A.json").read_text())
    stock_windows = {k: tuple(map(tuple, v)) for k, v in pa["windows"].items()}
    inst = torch.load(RESULTS / "v40_instrument.pt",
                      weights_only=False)["inst"]
    spec = channel_windows(stock_windows)

    fr, tr, ac, dn = collect_forward_walk(40_000, seed=7)
    fr_h, tr_h, ac_h, dn_h = collect_forward_walk(10_000, seed=207)
    print(f"[pf-F] walks done ({time.time()-t0:.0f}s)", flush=True)
    n_act = 17
    assert int(ac.max()) < n_act and int(ac_h.max()) < n_act

    enc = EncoderCrafter(band_dims=(16, 4, 2))
    enc.load_state_dict(torch.load(RESULTS / "v09_encoder.pt",
                                   map_location="cpu"))
    enc.eval()

    def feats(frames, dones):
        ep = torch.cumsum(torch.cat([torch.tensor([0]),
                                     dones[:-1].long()]), 0)
        es = ema_slow_stats(frames, ep)
        zs = []
        with torch.no_grad():
            for i in range(0, frames.shape[0], 512):
                zs.append(enc(frames[i:i + 512].float().div(255.0),
                              slow_feats=es[i:i + 512]))
        z = torch.cat(zs)
        m = torch.zeros(frames.shape[0], len(CHANNELS))
        for i, k in enumerate(CHANNELS):
            kind, (r0, r1), (c0, c1) = spec[k]
            x = frames[:, :, HUD_ROWS, :] if kind == "hud" else frames
            X = x[:, :, r0:r1, c0:c1].float().div(255.0).flatten(1)
            ch = inst["channels"][k]
            m[:, i] = ch["a"] * (X @ ch["w"] + ch["b"]) + ch["c"]
        return torch.cat([z, m], dim=-1)

    X, Xh = feats(fr, dn), feats(fr_h, dn_h)
    print(f"[pf-F] features {X.shape} ({time.time()-t0:.0f}s)", flush=True)
    vit = torch.stack([tr[k] for k in ("health", "food", "drink",
                                       "energy")], 1)
    vit_h = torch.stack([tr_h[k] for k in ("health", "food", "drink",
                                           "energy")], 1)

    def pairs(Xf, vitf, acf, dnf):
        keep = (~dnf[:-1]).nonzero().squeeze(1)
        oh = torch.nn.functional.one_hot(acf[keep + 1], n_act).float()
        return (torch.cat([Xf[keep], oh], -1),
                vitf[keep + 1] - vitf[keep], keep)

    Xi, dy, _ = pairs(X, vit, ac, dn)
    Xi_h, dy_h, keep_h = pairs(Xh, vit_h, ac_h, dn_h)

    torch.manual_seed(4000)
    model = torch.nn.Sequential(
        torch.nn.Linear(Xi.shape[1], 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for ep in range(40):
        perm = torch.randperm(Xi.shape[0])
        for i in range(0, len(perm), 2048):
            idx = perm[i:i + 2048]
            loss = ((model(Xi[idx]) - dy[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred_h = model(Xi_h)
    drops = (dy_h[:, 0] <= -0.9).numpy()
    auc = _auc((-pred_h[:, 0]).numpy(), drops.astype(float))

    # F2: action discrimination on drop events
    margins = []
    drop_idx = np.nonzero(drops)[0]
    with torch.no_grad():
        for j in drop_idx:
            base = Xh[keep_h[j]]
            allx = torch.cat([base.unsqueeze(0).expand(n_act, -1),
                              torch.eye(n_act)], -1)
            ph = model(allx)[:, 0]
            a_exec = int(ac_h[keep_h[j] + 1])
            others = torch.cat([ph[:a_exec], ph[a_exec + 1:]])
            margins.append(float(others.mean() - ph[a_exec]))
    margins = np.array(margins) if margins else np.array([0.0])
    frac_pos = float((margins > 0).mean())
    sens = []
    with torch.no_grad():
        for j in range(0, Xh.shape[0] - 1, 37):
            allx = torch.cat([Xh[j].unsqueeze(0).expand(n_act, -1),
                              torch.eye(n_act)], -1)
            ph = model(allx)[:, 0]
            sens.append(float(ph.max() - ph.min()))
    metrics = {"drop_auc": auc, "n_drops": int(drops.sum()),
               "margin_frac_pos": frac_pos,
               "margin_mean": float(margins.mean()),
               "action_sens_mean": float(np.mean(sens)),
               "holdout_mse_health": float(((pred_h[:, 0] - dy_h[:, 0]) ** 2)
                                           .mean())}
    gates = {"F1_drop_auc": bool(auc >= 0.70),
             "F2_discrimination": bool(frac_pos >= 0.55)}
    torch.save({"state_dict": model.state_dict(), "in_dim": Xi.shape[1],
                "n_act": n_act, "metrics": metrics, "gates": gates},
               RESULTS / "v40_forward.pt")
    print(f"[pf-F] drop AUC {auc:.3f} (n={int(drops.sum())}); "
          f"exec-action margin frac {frac_pos:.2f} mean "
          f"{float(margins.mean()):.3f}; action sensitivity "
          f"{float(np.mean(sens)):.3f}", flush=True)
    print(f"[pf-F] GATES: " + "  ".join(
        f"{k} {'PASS' if v else 'FAIL'}" for k, v in gates.items())
        + f"  ({time.time()-t0:.0f}s)", flush=True)
    if not all(gates.values()):
        print("[pf-F] -> flinch DEFERRED per SPEC C4 (no trustworthy "
              "action-discriminating forward model); fleet runs 3 arms",
              flush=True)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "counters"
    if mode == "counters":
        step_a()
    elif mode == "harness":
        harness()
    elif mode == "audit":
        audit()
    elif mode == "forward":
        forward_model()
    else:
        raise SystemExit(f"unknown pre-flight mode: {mode}")


if __name__ == "__main__":
    main()
