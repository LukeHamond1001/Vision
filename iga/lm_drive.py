"""v5.0 drive layer on the endless dialogue — full architecture, no scenes.

Laws, unchanged in substance (card A6):
  labels select, the world pays — a human "thanks" (the earned event)
    MINTS the channel it followed; pay comes only from measured
    readings graded against text the model cannot edit.
  no hold outlives its referent — every hold carries a horizon set by
    its band's clock; at the horizon it settles on the readings that
    arrived, or expires at exactly zero. No brackets needed.
  telescoping — pay = w*(phi_open - phi_settle), float64 bookkeeping,
    exact by construction; a hold that ends where it opened pays zero.
  imagination ranks and vetoes, never pays — a frontier target whose
    carrying band cannot currently predict its own next window is
    vetoed as unreachable; low fast-band fidelity vetoes on health.

Channels:
  recall:b<bin> — p(answer token) at annotated probes, binned by gap.
  fid:<k>       — band-k next-window cosine fidelity at ticks.

The scheduler: bin_weights() biases the weaver toward gap bins showing
learning progress — the drive layer choosing what rides the belt.
"""

import random
import torch

from .lm_bands import N_BANDS, CLOCKS

GAP_BINS = [(0, 256), (256, 2048), (2048, 16384), (16384, 10 ** 9)]
BIN_BAND = {0: 2, 1: 3, 2: 4, 3: 5}
FID_FLOOR = 0.05
FID_HEALTHY = (0.15, 0.60)
FRONTIER_DELTA = 0.05
EMA = 0.02
MAX_HOLDS_PER_LANE = 4


def gap_bin(gap):
    for i, (lo, hi) in enumerate(GAP_BINS):
        if lo <= gap < hi:
            return i
    return len(GAP_BINS) - 1


def horizon(band):
    return max(4 * CLOCKS[band], 512)


class Drive:
    def __init__(self, n_lanes, lam=0.25, seed=0, constants=None,
                 imagination_absent=False, absent_bands=(),
                 hold_cap=None):
        """constants: dict (or path to results/lm_constants.json) from
        iga.lm_calibrate — horizons and fid floors become data-set;
        absent, the scaffold defaults apply (debug only).
        imagination_absent (v5.2): the substrate carries no band
        predictors, so the imagination gate has no instrument — and an
        instrument that is absent holds no authority (F2). Fid-maintain
        proposals and fid-floor vetoes are skipped, ledgered.
        absent_bands (v5.4): bands the substrate does not implement
        (hybrid: 1, 2) — no organ, no maintain register; run 4 held
        perpetual zero-pay fid:1/2 maintains on organs that never
        existed. lam: 0.1 -> 0.25 (A24 L3) — run 4's in-window margin
        decayed all run while CE improved; pay pressure was losing."""
        self.imagination_absent = imagination_absent
        self.absent_bands = set(absent_bands)
        self.bin_band = dict(BIN_BAND)
        if isinstance(constants, str):
            import json as _json
            with open(constants) as f:
                constants = _json.load(f)
        c = constants or {}
        self._horizons = {int(k): int(v)
                          for k, v in c.get("horizons", {}).items()}
        self._fid_floors = {int(k): float(v)
                            for k, v in c.get("fid_floor", {}).items()}
        self.n_lanes = n_lanes
        self.lam = lam
        # A60 (v9.3): global cap on NEW holds per sweep. The A59b
        # onset-invariance analysis convicted event DENSITY: v-shards
        # mint ~13 holds/step vs r-shards' ~0.85, and every settled
        # hold's pay term injects retained-logp gradient into the
        # trunk (line ~190). The cap throttles the mint->settle->
        # re-propose loop to the measured-safe r-scale rate without
        # amputating any drive function. None = uncapped (parity).
        self.hold_cap = hold_cap
        self.rng = random.Random(seed)
        self.ema = {}
        self.records = {}
        self.levels_paid = set()
        self.minted = set()            # channel keys ratified by "thanks"
        self.holds = [[] for _ in range(n_lanes)]
        self.readings = {}             # (lane, key) -> [(t, tensor)]
        self.last_key = [None] * n_lanes
        self.ledger = []
        self.progress = {}             # key -> |d ema| EMA
        self.vetoes = 0
        self.proposed = 0
        self.step_t = 0

    # ---------- measurements ----------
    def probe(self, lane, prob_tensor, gap):
        key = f"recall:b{gap_bin(gap)}"
        self.readings.setdefault((lane, key), []).append(
            (self.step_t, prob_tensor))
        self.last_key[lane] = key
        v = float(prob_tensor.detach())
        old = self.ema.get(key, v)
        self.ema[key] = (1 - EMA) * old + EMA * v
        self.progress[key] = 0.9 * self.progress.get(key, 0.0) \
            + 0.1 * abs(self.ema[key] - old)

    def earned(self, lane, ok):
        # labels select: "thanks" mints the channel it followed; it
        # never pays, and "not right" mints nothing
        if ok and self.last_key[lane] is not None:
            self.minted.add(self.last_key[lane])

    def tick_fid(self, k, fid_tensor):
        v = float(fid_tensor.detach().mean())
        key = f"fid:{k}"
        old = self.ema.get(key, v)
        self.ema[key] = (1 - EMA) * old + EMA * v

    # ---------- proposer + imagination ----------
    def _propose(self, lane):
        open_keys = {h["key"] for h in self.holds[lane]}
        cands = []
        if not self.imagination_absent:
            for k in range(1, N_BANDS):
                if k in self.absent_bands:
                    continue          # no organ, no maintain register
                key = f"fid:{k}"
                if self.ema.get(key, 0.0) < FID_HEALTHY[0]:
                    cands.append(("maintain", key, k, FID_HEALTHY[0] + 0.05))
        for b in range(len(GAP_BINS)):
            key = f"recall:b{b}"
            if key not in self.minted:
                continue
            rec = self.records.get(key, self.ema.get(key, 0.0))
            level = int(rec / FRONTIER_DELTA)
            if (key, level) in self.levels_paid:
                continue
            cands.append(("frontier", key, self.bin_band[b],
                          rec + FRONTIER_DELTA))
        scored = []
        for kind, key, band, target in cands:
            if key in open_keys:      # one register per channel per lane
                continue
            self.proposed += 1
            carry = self.ema.get(f"fid:{band}", 0.0)
            if not self.imagination_absent:
                fast = min((self.ema.get(f"fid:{k}", 1.0) for k in (1, 2)
                            if k not in self.absent_bands), default=1.0)
                floor = self._fid_floors.get(band, FID_FLOOR)
                fast_floor = max((self._fid_floors.get(k, FID_FLOOR)
                                  for k in (1, 2)
                                  if k not in self.absent_bands),
                                 default=0.0)
                if kind == "frontier" and (carry < floor
                                           or fast < fast_floor):
                    self.vetoes += 1
                    continue
            scored.append((carry, kind, key, band, target))
        scored.sort(reverse=True)
        out = []
        room = MAX_HOLDS_PER_LANE - len(self.holds[lane])
        for carry, kind, key, band, target in scored[:max(room, 0)]:
            phi0 = max(0.0, target - self.ema.get(key, 0.0)) / max(target, 1e-6)
            out.append({"lane": lane, "band": band, "key": key,
                        "target": target, "phi0": phi0, "w": 1.0,
                        "t0": self.step_t,
                        "due": self.step_t + self.horizon_for(band),
                        "kind": kind})
        return out

    def horizon_for(self, band):
        return self._horizons.get(band, horizon(band))

    # ---------- the sweep: settle due holds, then propose ----------
    def sweep(self, losses):
        self._minted_this_sweep = 0
        for lane in range(self.n_lanes):
            keep = []
            for h in self.holds[lane]:
                if self.step_t < h["due"]:
                    keep.append(h)
                    continue
                rs = [r for (t, r) in self.readings.get((lane, h["key"]), [])
                      if t > h["t0"]]  # only progress made DURING the hold
                if h["key"].startswith("fid:"):
                    phi1 = max(0.0, h["target"] - self.ema.get(h["key"], 0.0)) \
                        / max(h["target"], 1e-6)
                    self._settle(h, phi1, h["w"] * (h["phi0"] - phi1))
                elif rs:
                    mean_r = torch.stack(rs).mean()
                    phi1_t = torch.clamp(h["target"] - mean_r, min=0.0) \
                        / max(h["target"], 1e-6)
                    if phi1_t.requires_grad:
                        losses.append(-self.lam * h["w"] * (h["phi0"] - phi1_t))
                    phi1 = float(phi1_t.detach())
                    pay = h["w"] * (h["phi0"] - phi1)
                    self._settle(h, phi1, pay)
                    if h["kind"] == "frontier" and phi1 == 0.0:
                        self.levels_paid.add(
                            (h["key"], int(h["target"] / FRONTIER_DELTA)))
                else:
                    self._settle(h, h["phi0"], 0.0)   # expiry: exactly zero
            self.holds[lane] = keep
            new = self._propose(lane)
            if self.hold_cap is not None:
                take = max(0, self.hold_cap - self._minted_this_sweep)
                new = new[:take]
                self._minted_this_sweep += len(new)
            self.holds[lane] += new
        for key, val in self.ema.items():
            if val > self.records.get(key, 0.0):
                self.records[key] = val
        cutoff = self.step_t - 8 * max(CLOCKS)
        self.readings = {k: [(t, r) for (t, r) in v if t > cutoff]
                         for k, v in self.readings.items()}

    def _settle(self, h, phi1, pay):
        self.ledger.append({**{k: h[k] for k in
                               ("lane", "band", "key", "phi0", "w", "t0")},
                            "phi1": phi1, "pay": pay, "t1": self.step_t})

    def detach_readings(self):
        self.readings = {k: [(t, r.detach()) for (t, r) in v]
                         for k, v in self.readings.items()}

    # ---------- scheduler ----------
    def bin_weights(self):
        w = [self.progress.get(f"recall:b{b}", 0.0) + 1e-4 for b in range(4)]
        s = sum(w)
        return [x / s for x in w]

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
                lines.append(f"lane{lane} {h['kind']:8s} {h['key']:14s} "
                             f"target {h['target']:.2f} phi0 {h['phi0']:.2f} "
                             f"band {h['band']} due {h['due']}")
        for key in sorted(self.records):
            lines.append(f"record   {key:14s} {self.records[key]:.3f} "
                         f"(ema {self.ema.get(key, 0):.3f})")
        return "\n".join(lines) if lines else "(no open holds yet)"
