"""v5.0 drive layer on the language conveyor — full architecture, v0.

Laws carried over intact:
  labels select, the world pays — <ok> scenes mint templates; only
    measured readings (graded against text the model cannot edit) pay.
  no hold outlives its referent — turn holds settle at <eot_model>,
    scene holds at </scene> (arrival or expiry-at-zero), run holds
    live on the training-run clock.
  telescoping — pay = w*(phi_open - phi_settle); a hold that closes
    where it opened pays exactly zero; the ledger stores every hold
    and the audit reconstructs every pay.
  imagination ranks and vetoes, never pays — the band predictors'
    demonstrated fidelity is the forward model's competence estimate:
    a frontier target whose carrying band cannot currently predict its
    own next window is vetoed as unreachable; low fast-band fidelity
    vetoes on health (C4 transposed). Nothing here is trained.

Channels (all graded against the stream's future, by construction):
  recall:<type>:b<bin>  — model probability of the planted answer
    token at an annotated probe, binned by gap (the held-meaning
    instruments). Differentiable through the softmax.
  fid:<k>               — band-k next-window cosine fidelity at ticks.

Gradient note: phi_open is stored detached (it may predate the chunk);
pay's gradient flows through the settlement reading only — the
pressure is "arrive", which is the correct direction.
"""

import math
import random
import torch

from .lm_bands import N_BANDS

GAP_BINS = [(0, 256), (256, 2048), (2048, 16384), (16384, 10 ** 9)]
BIN_BAND = {0: 2, 1: 3, 2: 4, 3: 5}  # which band must carry each gap bin
FID_FLOOR = 0.05     # imagination veto floors (v0 constants, card-fixable)
FID_HEALTHY = (0.15, 0.60)
FRONTIER_DELTA = 0.05
EMA = 0.02


def gap_bin(gap):
    for i, (lo, hi) in enumerate(GAP_BINS):
        if lo <= gap < hi:
            return i
    return len(GAP_BINS) - 1


class Drive:
    def __init__(self, n_lanes, lam=0.1, seed=0):
        self.n_lanes = n_lanes
        self.lam = lam
        self.rng = random.Random(seed)
        self.ema = {}                  # channel -> float EMA of readings
        self.records = {}              # channel -> best run-EMA (frontier)
        self.levels_paid = set()       # (channel, level) one-shot
        self.minted = set()            # (scene_type, channel) templates
        self.holds = [[] for _ in range(n_lanes)]
        self.ledger = []               # dicts: lane band key phi0 phi1 pay w t0 t1 scope
        self.progress = {"saga": 0.0, "episode": 0.0}
        self._prev_ema = {}
        self.vetoes = 0
        self.proposed = 0
        self.scene_probes = [dict() for _ in range(n_lanes)]  # key -> [tensor readings]
        self.scene_type = [None] * n_lanes
        self.step_t = 0

    # ---------- proposer + imagination ----------
    def _propose(self, lane, scene_type):
        cands = []
        for k in range(1, N_BANDS):
            key = f"fid:{k}"
            v = self.ema.get(key, 0.0)
            if v < FID_HEALTHY[0]:      # maintain: restore into range
                cands.append(("maintain", key, k, FID_HEALTHY[0] + 0.05))
        for b in range(len(GAP_BINS)):
            key = f"recall:{scene_type}:b{b}"
            if (scene_type, key) not in self.minted:
                continue                # labels select: unminted never proposed
            rec = self.records.get(key, self.ema.get(key, 0.0))
            level = int(rec / FRONTIER_DELTA)
            if (key, level) in self.levels_paid:
                continue                # one-shot per level
            cands.append(("frontier", key, BIN_BAND[b], rec + FRONTIER_DELTA))
        committed = []
        scored = []
        for kind, key, band, target in cands:
            self.proposed += 1
            carry_fid = self.ema.get(f"fid:{band}", 0.0)
            fast_fid = min(self.ema.get(f"fid:{k}", 1.0) for k in (1, 2))
            if kind == "frontier" and carry_fid < FID_FLOOR:
                self.vetoes += 1        # unreachable: carrying band can't hold
                continue
            if kind == "frontier" and fast_fid < FID_FLOOR:
                self.vetoes += 1        # health veto (C4 transposed)
                continue
            scored.append((carry_fid, kind, key, band, target))
        scored.sort(reverse=True)       # imagination ranks...
        for carry_fid, kind, key, band, target in scored[:3]:
            phi0 = max(0.0, target - self.ema.get(key, 0.0)) / max(target, 1e-6)
            committed.append({"lane": lane, "band": band, "key": key,
                              "target": target, "phi0": phi0, "w": 1.0,
                              "scope": "scene", "t0": self.step_t,
                              "kind": kind})
        return committed                # ...and never pays

    # ---------- event handling ----------
    def scene_start(self, lane, scene_type):
        self.scene_type[lane] = scene_type
        self.scene_probes[lane] = {}
        self.holds[lane] = [h for h in self.holds[lane] if h["scope"] == "run"]
        self.holds[lane] += self._propose(lane, scene_type)

    def probe(self, lane, prob_tensor, gap, scene_type):
        key = f"recall:{scene_type}:b{gap_bin(gap)}"
        self.scene_probes[lane].setdefault(key, []).append(prob_tensor)
        v = float(prob_tensor.detach())
        self.ema[key] = (1 - EMA) * self.ema.get(key, v) + EMA * v

    def tick_fid(self, k, fid_tensor):
        v = float(fid_tensor.detach().mean())
        key = f"fid:{k}"
        self.ema[key] = (1 - EMA) * self.ema.get(key, v) + EMA * v

    def scene_end(self, lane, ok, scene_type, losses):
        # minting: labels select what may pay, they never pay
        if ok:
            for key in self.scene_probes[lane]:
                self.minted.add((scene_type, key))
        pays = 0.0
        for h in self.holds[lane]:
            if h["scope"] != "scene":
                continue
            readings = self.scene_probes[lane].get(h["key"])
            if h["key"].startswith("fid:"):
                k = int(h["key"].split(":")[1])
                phi1 = max(0.0, h["target"] - self.ema.get(h["key"], 0.0)) \
                    / max(h["target"], 1e-6)
                pay = h["w"] * (h["phi0"] - phi1)
                self._settle(h, phi1, pay)
                pays += pay
            elif readings:
                mean_r = torch.stack(readings).mean()
                phi1_t = torch.clamp(h["target"] - mean_r, min=0.0) \
                    / max(h["target"], 1e-6)
                if phi1_t.requires_grad:
                    losses.append(-self.lam * h["w"] * (h["phi0"] - phi1_t))
                # ledger bookkeeping in float64 — exact by construction;
                # the loss term above is its differentiable twin
                phi1 = float(phi1_t.detach())
                pay = h["w"] * (h["phi0"] - phi1)
                self._settle(h, phi1, pay)
                pays += pay
                if h["kind"] == "frontier" and phi1 == 0.0:
                    level = int(h["target"] / FRONTIER_DELTA)
                    self.levels_paid.add((h["key"], level))
            else:
                self._settle(h, h["phi0"], 0.0)       # expiry: zero, printed
        self.holds[lane] = [h for h in self.holds[lane] if h["scope"] == "run"]
        # run band: records + scheduler progress
        for key, val in list(self.ema.items()):
            if val > self.records.get(key, 0.0):
                self.records[key] = val
        prev = self._prev_ema.get(scene_type, None)
        cur = sum(v for k, v in self.ema.items()
                  if k.startswith(f"recall:{scene_type}")) or 0.0
        if prev is not None:
            self.progress[scene_type] = (1 - 0.1) * self.progress[scene_type] \
                + 0.1 * abs(cur - prev)
        self._prev_ema[scene_type] = cur

    def _settle(self, h, phi1, pay):
        self.ledger.append({**{k: h[k] for k in
                               ("lane", "band", "key", "phi0", "w", "scope", "t0")},
                            "phi1": phi1, "pay": pay, "t1": self.step_t})

    # ---------- scheduler (arm-3 proposer choosing the belt) ----------
    def next_kind(self, lane):
        if self.rng.random() < 0.3:
            return self.rng.choice(list(self.progress))
        return max(self.progress, key=self.progress.get)

    # ---------- audit ----------
    def audit(self):
        exact = all(abs(e["pay"] - e["w"] * (e["phi0"] - e["phi1"])) < 1e-9
                    for e in self.ledger)
        scoped = all(e["t1"] >= e["t0"] for e in self.ledger)
        return {"holds": len(self.ledger), "telescoping_exact": exact,
                "scoped": scoped, "vetoes": self.vetoes,
                "proposed": self.proposed}

    # ---------- the readable agenda ----------
    def panel(self):
        lines = []
        for lane in range(self.n_lanes):
            for h in self.holds[lane]:
                lines.append(f"lane{lane} {h['kind']:8s} {h['key']:22s} "
                             f"target {h['target']:.2f} phi0 {h['phi0']:.2f} "
                             f"band {h['band']}")
        for key in sorted(self.records):
            lines.append(f"record   {key:22s} {self.records[key]:.3f} "
                         f"(ema {self.ema.get(key, 0):.3f})")
        return "\n".join(lines) if lines else "(no open holds yet)"
