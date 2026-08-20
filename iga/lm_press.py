"""A64 secondary rewards — per-band press-prophets, SPECTATORS.

The ratified design: higher rewards are BUILT by the timebands —
band organs learn to predict the primary (the graded press) at
their own horizons. Phase 2 keeps them measurement-only (law B5):
each hybrid band k gets a tiny head predicting the press value
that will arrive within the band's horizon, trained ONLY on real
presses (aged-out no-press snapshots supply zero targets), on
DETACHED band states, with its own optimizer. Zero influence on
the substrate: the model trains bit-identically with the prophet
on or off. Graduation to paying is a later, separate decision on
these heads' measured fidelity.

Construction draws from the global torch RNG (nn.Linear init) —
construct the prophet BEFORE train(), which reseeds; observe()
and its training steps consume no RNG at all.
"""

import torch
import torch.nn as nn

from .lm_hybrid import HYBRID_CLOCKS

BATCH = 64          # pending pairs per head step
CAP = 4096          # metric/report cap guard


class PressProphet:
    def __init__(self, d, lr=1e-3, device="cpu", clocks=None):
        # A70: clocks=None keeps the certified 3-band prophet exact
        self.clocks = dict(HYBRID_CLOCKS if clocks is None else clocks)
        self.bands = sorted(self.clocks)
        self.heads = nn.ModuleDict(
            {str(k): nn.Linear(d, 1) for k in self.bands}).to(device)
        self.opt = torch.optim.Adam(self.heads.parameters(), lr=lr)
        self.rings = {k: [] for k in self.bands}   # (t, state [B,d])
        self.covered = set()                        # (band, t, lane)
        self.pending = {k: [] for k in self.bands}  # (vec, target)
        self._press_i = 0
        self.stats = {k: {"n_pos": 0, "n_zero": 0, "n_neg": 0,
                          "sum_pos": 0.0, "sum_zero": 0.0,
                          "sum_neg": 0.0, "steps": 0, "loss": 0.0}
                      for k in self.bands}

    def observe(self, model, drive):
        st = getattr(model, "_st", None)
        if st is None or not hasattr(model, "bands"):
            return
        now = drive.step_t
        for k in self.bands:
            if st["chunk"] % self.clocks[k] == 0:
                self.rings[k].append((now, st["h"][k].detach().clone()))
        for p in drive.presses[self._press_i:]:
            for k in self.bands:
                hor = drive.horizon_for(k)
                for t, h in self.rings[k]:
                    mark = (k, t, p["lane"])
                    if p["t"] - hor < t < p["t"] \
                            and mark not in self.covered:
                        self.covered.add(mark)
                        self.pending[k].append(
                            (h[p["lane"]], float(p["v"])))
        self._press_i = len(drive.presses)
        for k in self.bands:
            hor = drive.horizon_for(k)
            keep = []
            for t, h in self.rings[k]:
                if t <= now - hor:      # aged out: uncovered lanes = 0
                    for lane in range(h.shape[0]):
                        if (k, t, lane) in self.covered:
                            self.covered.discard((k, t, lane))
                        else:
                            self.pending[k].append((h[lane], 0.0))
                else:
                    keep.append((t, h))
            self.rings[k] = keep
            if len(self.pending[k]) >= BATCH:
                self._step(k)

    def _step(self, k):
        pairs = self.pending[k][:CAP]
        self.pending[k] = []
        X = torch.stack([p[0] for p in pairs])
        y = torch.tensor([p[1] for p in pairs],
                         device=X.device).unsqueeze(1)
        pred = self.heads[str(k)](X)
        loss = torch.nn.functional.mse_loss(pred, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        s = self.stats[k]
        with torch.no_grad():
            pr = pred.squeeze(1)
            for grp, mask in (("pos", y.squeeze(1) > 0),
                              ("zero", y.squeeze(1) == 0),
                              ("neg", y.squeeze(1) < 0)):
                n = int(mask.sum())
                if n:
                    s[f"n_{grp}"] += n
                    s[f"sum_{grp}"] += float(pr[mask].sum())
        s["steps"] += 1
        s["loss"] = 0.9 * s["loss"] + 0.1 * float(loss.detach()) \
            if s["steps"] > 1 else float(loss.detach())

    def report(self):
        out = {}
        for k, s in self.stats.items():
            mp = s["sum_pos"] / s["n_pos"] if s["n_pos"] else 0.0
            mz = s["sum_zero"] / s["n_zero"] if s["n_zero"] else 0.0
            mn = s["sum_neg"] / s["n_neg"] if s["n_neg"] else 0.0
            out[k] = {"n_pos": s["n_pos"], "n_zero": s["n_zero"],
                      "n_neg": s["n_neg"],
                      "mean_pos": round(mp, 4),
                      "mean_zero": round(mz, 4),
                      "mean_neg": round(mn, 4),
                      "sep": round(mp - mz, 4),
                      "steps": s["steps"],
                      "loss": round(s["loss"], 5)}
        return out
