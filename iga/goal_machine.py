"""v4.0 goal machine: the toy-world proposer ladder ported to Crafter's
measured register space (SPEC L1-L4, C1-C7, G5, G7 — the complete
architecture's drive layer, parameter-free).

Register space = six CALIBRATED instrument channels in truth units
(health, food, drink, energy on 0-9; wood, sapling as counts):
closed-form heads (round-10 recipe) + affine walk calibration, frozen
before any RL. The policy still sees pixels only; the instrument is the
drive layer's sensor — same legitimacy class as phase-1
(instrument-located, walk-calibrated, frozen).

Port decisions (each disclosed, none silent):
  proposer   PARAMETER-FREE. Stock/vital spaces are small and
             enumerable, so the toy's learned candidate generator is
             replaced by exhaustive menus + the SAME fixed prospective
             ranking (f+ - f- + one-shot novelty), C4 prospective veto,
             C5 horizon, C7 value bar with the G7 novelty lane. G5
             (progress pays the policy, never the proposer) holds by
             construction: there is nothing to pay.
  goals      RAMPS, not points: goal = (channel c, level t), potential
             phi = max(0, t - m_c)/s_c. "At least t" is the native
             semantics of stocks and homeostasis; a point target would
             punish overshoot.
  curiosity  PER-LIFE one-shot (C6 within an episode). Death resets
             stocks to zero, so cross-life extinguishing would
             permanently veto the frontier via C5. Native Crafter
             reward re-pays achievements per episode — parity kept.
             Within a life no cell's bonus pays twice.
  C1 cap     arrive-count cap per stock cell per life (2) — defense in
             depth behind the C7 gain bar. NOTE (measured, v4.0 fleet):
             with C7 in improvement form the bar alone blocks
             re-commits of arrived cells, so the cap never fired in
             any fleet seed (cap rejects = 0). It stays wired as the
             backstop C1 specifies, not as a load-bearing constraint.
             Vitals are uncapped: sag is env-clocked (decay timers), so
             restore pay is rate-bounded by physiology and cannot be
             accelerated by the policy — the v1.2 maintenance lane.
             Bonus scope, stated plainly: the one-shot frontier bonus
             resets per LIFE (like native Crafter's per-episode
             achievements); the telescoping theorem covers the
             PROGRESS stream, and per-life bonus totals are bounded by
             the cell count — across lives the bonus re-pays at
             episode-turnover rate, by design and disclosed.
  vitals     food/drink/energy restore-ramps, proposed only when the
             vital sags below restore_lo. No instant-arrival commits
             (the toy's round-10 arrive-eps lesson, made explicit:
             already-satisfied candidates are skipped). Health has no
             goal — not directly actionable; it lives in f-.
  reward     band-weighted ramp progress (potential-based, telescoping,
             exact-claim ledger) + a one-shot bonus on FRONTIER arrival
             (G7: novelty dies on arrival, unfarmable). No
             realized-value stream — no hand-crafted dense reward; f+/f-
             are used PROSPECTIVELY only (ranking and veto).
  C7 bar     IMPROVEMENT form: candidates are scored by the prospective
             GAIN f(imagined) - f(now), not absolute f. The toy's
             absolute bar worked because its evaluators sat near zero
             baseline; here satiation keeps f+ far above any bar, which
             would let arrived stock cells re-commit forever. With the
             gain form, stock goals (gain 0 — stocks are not in f) pass
             ONLY through one-shot novelty, which is exactly the G7
             design; re-collection after consumption is paid through the
             NEXT frontier cell's climb, never by re-arriving a spent
             cell. Chop-place farming is structurally unpaid.
  tie-break  breadth-first frontier: among equal scores, the channel
             with the fewest arrivals this life wins. After wood>=1
             arrives, the stock register rotates to sapling while wood
             gets SPENT (table placement) uncharged — the consume-trap
             dissolves for the common case by ordering, not by a patch.
             A held higher wood target during placement is still charged
             as real measured regress (bounded, telescoped, measured in
             the T3 audit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .crafter_support import HUD_ROWS, MID_WINDOWS, closed_form_head, measure_rho

CHANNELS = ("health", "food", "drink", "energy", "wood", "sapling")
VITALS = (0, 1, 2, 3)
STOCKS = (4, 5)
GOAL_VITALS = (1, 2, 3)          # food, drink, energy — health is f- only


# ---------------------------------------------------------------- instrument
def channel_windows(stock_windows: dict) -> dict:
    """Per-channel pixel window: ('hud'|'abs', (r0,r1), (c0,c1)).
    Vitals come from the phase-1 digit windows (HUD-strip relative);
    stocks from the pre-flight-A exclusive-change windows (absolute)."""
    win = {}
    for k, ((r0, r1), (c0, c1)) in zip(("health", "food", "drink", "energy"),
                                       MID_WINDOWS):
        win[k] = ("hud", (r0, r1), (c0, c1))
    for k in ("wood", "sapling"):
        (r0, r1), (c0, c1) = stock_windows[k]
        win[k] = ("abs", (r0, r1), (c0, c1))
    return win


def extract_window(frame_u8: torch.Tensor, spec) -> torch.Tensor:
    """frame (3,64,64) uint8 -> flat float window in [0,1]."""
    kind, (r0, r1), (c0, c1) = spec
    x = frame_u8[:, HUD_ROWS, :] if kind == "hud" else frame_u8
    return x[:, r0:r1, c0:c1].float().div(255.0).flatten()


def calibrate_instrument(wins: dict, truth: dict, ep_ids: torch.Tensor,
                         wins_t: dict, truth_t: dict) -> dict:
    """Solve one closed-form head per channel on the calibration walk,
    then affine-map its unit-variance output to TRUTH UNITS (a*z + c by
    least squares on the same walk). Held-out corr/MAE reported from the
    audit walk. Deterministic, frozen; uses truth ONCE at build time —
    the phase-1 legitimacy class."""
    inst = {"channels": {}, "order": CHANNELS}
    taus = {}
    for k in CHANNELS:
        y = truth[k]
        rho = measure_rho(y, ep_ids, 20)
        if math.isnan(rho):
            rho = 0.9
        tau = -20 / math.log(max(min(float(rho), 0.995), 0.05))
        taus[k] = tau
        w, b = closed_form_head(wins[k], ep_ids, 20, tau)
        z = wins[k] @ w + b
        zv = float(z.var())
        a = float(((z - z.mean()) * (y - y.mean())).mean() / max(zv, 1e-8))
        if k in ("wood", "sapling") and bool((y == 0).any()):
            # Counters have a TRUE ZERO: anchor the intercept at the
            # empty state (digit-glyph -> z is nonlinear at the low end;
            # least-squares intercepts read ~1 at truth 0, shifting the
            # whole frontier ladder's labels by one).
            c = -a * float(z[y == 0].median())
        else:
            c = float(y.mean() - a * z.mean())
        zt = wins_t[k] @ w + b
        mt = a * zt + c
        yt = truth_t[k]
        corr = abs(float(torch.corrcoef(torch.stack([mt, yt]))[0, 1])) \
            if float(yt.std()) > 1e-6 else float("nan")
        mae = float((mt - yt).abs().mean())
        inst["channels"][k] = {
            "w": w, "b": float(b), "a": a, "c": c,
            "std": max(float(y.std()), 1e-3), "tau": tau,
            "corr_holdout": corr, "mae_holdout": mae}
    # Per-band progress weights are NOT estimated from walk
    # autocorrelation: counters are event-driven step functions, and
    # rho-tau reads them as FAST (the v2.0 tau_half lesson in HUD form —
    # the harness measured stocks at 0.1x vitals, exactly backwards).
    # The spec's own form is the HOLD-LENGTH ratio (a design constant):
    # w_bands = (1, holds[1]/holds[0]), set at machine config.
    return inst


class Readout:
    """Frozen instrument: frames -> m in truth units. No parameters —
    fixed tensors only (W1/W2)."""

    def __init__(self, inst: dict, stock_windows: dict):
        self.inst = inst
        self.win = channel_windows(stock_windows)

    def measure(self, frame_u8: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(len(CHANNELS))
        for i, k in enumerate(CHANNELS):
            ch = self.inst["channels"][k]
            x = extract_window(frame_u8, self.win[k])
            m[i] = ch["a"] * (x @ ch["w"] + ch["b"]) + ch["c"]
        return m


# ---------------------------------------------------------------- evaluators
def f_pos(m: torch.Tensor) -> float:
    """Fixed prospective evaluator: being fed/watered/rested is good.
    Used for RANKING and the C7 bar only — never paid."""
    return float((m[1] + m[2] + m[3]) / 27.0)


def f_neg(m: torch.Tensor) -> float:
    """Fixed danger evaluator: low health. Prospective veto (C4) only."""
    return float(max(0.0, 4.5 - float(m[0])) / 9.0)


# ---------------------------------------------------------------- machine
@dataclass
class GMConfig:
    # -------- channel spec (defaults = the Crafter instrument; the
    # DriveWrapper passes its own — same audited machine, any world)
    channels: tuple = CHANNELS
    vitals: tuple = VITALS           # band-0 channel indices
    stocks: tuple = STOCKS           # band-1 channel indices
    goal_vitals: tuple = GOAL_VITALS # maintain-lane menu (subset of vitals)
    restore: dict | None = None      # per-channel {idx: (lo, hi)};
    #   None -> the scalar restore_lo/restore_hi below for every vital
    f_pos_fn: object = None          # callable(m)->float; None -> Crafter f+
    f_neg_fn: object = None          # callable(m)->float; None -> Crafter f-
    holds: tuple = (60, 180)         # vitals, stocks (steps; ~episode-scale)
    arrive_eps_per_channel: tuple | None = None   # per-CHANNEL tolerances
    #   in the channel's own units (the DriveWrapper derives these from
    #   calibrated stds, making arrivals unit-free); None -> the
    #   per-band truth-unit constants below (the Crafter instrument).
    arrive_eps: tuple = (0.75, 0.4)  # per band, truth units. Vitals are
    #   release-TOLERANT (no bonus at stake on a vitals arrival; an early
    #   release just re-proposes if the vital is still low) so eps absorbs
    #   head noise (MAE ~0.5). Stocks are STRICT: the one-shot frontier
    #   bonus is the corruption channel, and the stock heads are tight
    #   (wood MAE 0.044) — the B6 audit verifies phantom rate directly.
    restore_lo: float = 5.5          # propose a restore when vital < lo
    restore_hi: float = 8.0          # ...targeting >= hi
    horizon_stock: float = 2.0       # C5: at most +2 counts per hold
    cap_arrivals: int = 2            # C1: per stock cell per life
    frontier_bonus: float = 1.0      # G7 one-shot arrival bonus
    value_bar: float = 0.02          # C7 (f units)
    veto_threshold: float = 0.5      # C4 prospective (f- units)
    w_bands: tuple = (1.0, 3.0)      # hold-length ratio (spec law)
    stds: tuple = (1.0,) * 6         # per-channel walk stds (frozen)
    use_proposer: bool = True        # False = fixed-target ablation arm
    fixed_targets: tuple = ()        # ((channel, level), ...) when ablated


@dataclass
class Hold:
    band: int
    channel: int
    target: float
    phi_commit: float
    t_commit: int
    remaining: int
    paid: float = 0.0
    max_prefix: float = 0.0
    frontier: bool = False


class GoalMachine:
    """Per-env drive layer. Parameter-free (G5 structural). One register
    per band: 0 = vitals maintenance, 1 = stock frontier."""

    def __init__(self, cfg: GMConfig):
        self.cfg = cfg
        self._fp = cfg.f_pos_fn or f_pos
        self._fn = cfg.f_neg_fn or f_neg
        self.ledger: list[dict] = []      # closed holds (audit surface)
        self.trace: list[tuple] = []      # (t, event, band, channel, level)
        self.t = 0                        # GLOBAL step clock (not per-life):
        # ledger/trace times index the caller's stream, so audits can look
        # up ground truth at event times.
        self.rejects = {"satisfied": 0, "veto": 0, "horizon": 0,
                        "bar": 0, "cap": 0}   # proposer diagnostics (C)
        self.total_bonus = 0.0
        self.reset(None)

    # -------------------------------------------------------------- life
    def reset(self, m0: torch.Tensor | None) -> None:
        self.m = None
        self.holds: list[Hold | None] = [None, None]
        self.arrived_cells: set = set()   # C6 within-life (bonus paid)
        self.arrive_count: dict = {}      # C1 cap
        self.arrivals_by_channel: dict = {}   # breadth-first tie-break
        if m0 is not None:
            self.m = m0.detach().clone()
            self._propose_all()

    # -------------------------------------------------------------- goals
    def _band_of(self, c: int) -> int:
        return 0 if c in self.cfg.vitals else 1

    def _phi(self, m: torch.Tensor, c: int, t: float) -> float:
        return max(0.0, t - float(m[c])) / self.cfg.stds[c]

    def _eps(self, c: int) -> float:
        if self.cfg.arrive_eps_per_channel is not None:
            return self.cfg.arrive_eps_per_channel[c]
        return self.cfg.arrive_eps[self._band_of(c)]

    def _satisfied(self, m: torch.Tensor, c: int, t: float) -> bool:
        return float(m[c]) >= t - self._eps(c)

    def _imagine(self, c: int, t: float) -> torch.Tensor:
        g = self.m.clone()
        g[c] = max(float(g[c]), t)
        return g

    def _candidates(self, band: int) -> list[tuple[int, float, bool]]:
        """(channel, level, is_frontier) menu. Enumeration is complete —
        the proposer's whole search space, visible in one screen."""
        out = []
        if band == 0:
            for c in self.cfg.goal_vitals:
                lo, hi = (self.cfg.restore or {}).get(
                    c, (self.cfg.restore_lo, self.cfg.restore_hi))
                if float(self.m[c]) < lo:
                    out.append((c, hi, False))
        else:
            for c in self.cfg.stocks:
                t = float(int(round(float(self.m[c]))) + 1)
                cell = (c, int(t))
                if self.arrive_count.get(cell, 0) >= self.cfg.cap_arrivals:
                    self.rejects["cap"] += 1                   # C1
                    continue
                if t - float(self.m[c]) > self.cfg.horizon_stock:
                    self.rejects["horizon"] += 1               # C5
                    continue
                out.append((c, t, True))
        return out

    def _propose(self, band: int) -> None:
        if self.holds[band] is not None:
            return
        if not self.cfg.use_proposer:
            for c, t in self.cfg.fixed_targets:
                b = 0 if c in VITALS else 1
                if b == band and not self._satisfied(self.m, c, t):
                    self._commit(band, c, float(t), frontier=False)
                    return
            return
        base = self._fp(self.m) - self._fn(self.m)
        best, best_key, best_frontier = None, None, False
        for c, t, frontier in self._candidates(band):
            if self._satisfied(self.m, c, t):
                self.rejects["satisfied"] += 1
                continue                    # no instant-arrival commits
            g = self._imagine(c, t)
            if self._fn(g) > self.cfg.veto_threshold:
                self.rejects["veto"] += 1                      # C4
                continue
            novelty = (self.cfg.frontier_bonus
                       if frontier and (c, int(t)) not in self.arrived_cells
                       else 0.0)
            gain = self._fp(g) - self._fn(g) - base
            if gain < self.cfg.value_bar and novelty <= 0:
                self.rejects["bar"] += 1    # C7 improvement bar / G7 lane
                continue
            key = (gain + novelty, -self.arrivals_by_channel.get(c, 0), -c)
            if best_key is None or key > best_key:
                best, best_key, best_frontier = (c, t), key, frontier
        if best is not None:
            self._commit(band, best[0], best[1], frontier=best_frontier)

    def _propose_all(self) -> None:
        for band in (1, 0):               # slow band proposes first
            self._propose(band)

    def _commit(self, band: int, c: int, t: float, frontier: bool) -> None:
        self.holds[band] = Hold(
            band=band, channel=c, target=t,
            phi_commit=self._phi(self.m, c, t), t_commit=self.t,
            remaining=self.cfg.holds[band], frontier=frontier)
        self.trace.append((self.t, "commit", band, self.cfg.channels[c], t))

    # -------------------------------------------------------------- step
    def step(self, m_now: torch.Tensor, done: bool = False) -> tuple[float, dict]:
        """Advance one env transition (self.m -> m_now). Returns the
        drive reward for this transition and an event dict. On done,
        pass the LAST-SEEN measurement (self.m.clone()) so the dying
        transition pays nothing and claims settle at the last-seen
        state — the v1.2 boundary rule; DriveWrapper does exactly this.
        (Passing the true terminal measurement instead charges the
        death transition — a design choice, but not the audited one.)
        Then reset(m0_of_next_life)."""
        assert self.m is not None, "reset(m0) before stepping"
        self.t += 1
        r = 0.0
        ev = {"arrivals": [], "timeouts": [], "bonus": 0.0}
        for h in self.holds:
            if h is None:
                continue
            w = self.cfg.w_bands[h.band]
            pay = w * (self._phi(self.m, h.channel, h.target)
                       - self._phi(m_now, h.channel, h.target))
            h.paid += pay
            h.max_prefix = max(h.max_prefix, h.paid)
            r += pay
            h.remaining -= 1
        self.m = m_now.detach().clone()
        for band, h in enumerate(self.holds):
            if h is None:
                continue
            arrived = self._satisfied(self.m, h.channel, h.target)
            if arrived or h.remaining <= 0 or done:
                how = "arrive" if arrived else ("death" if done else "timeout")
                if arrived:
                    cell = (h.channel, int(h.target))
                    self.arrivals_by_channel[h.channel] = \
                        self.arrivals_by_channel.get(h.channel, 0) + 1
                    if h.frontier:
                        self.arrive_count[cell] = \
                            self.arrive_count.get(cell, 0) + 1
                        if cell not in self.arrived_cells:      # G7 one-shot
                            self.arrived_cells.add(cell)
                            r += self.cfg.frontier_bonus
                            ev["bonus"] += self.cfg.frontier_bonus
                            self.total_bonus += self.cfg.frontier_bonus
                    ev["arrivals"].append((self.cfg.channels[h.channel], h.target))
                else:
                    ev["timeouts"].append((self.cfg.channels[h.channel], h.target))
                self.ledger.append({
                    "band": band, "channel": self.cfg.channels[h.channel],
                    "target": h.target, "t_commit": h.t_commit,
                    "t_close": self.t, "closed_by": how,
                    "phi_commit": h.phi_commit,
                    "phi_close": self._phi(self.m, h.channel, h.target),
                    "w": self.cfg.w_bands[h.band],
                    "paid": h.paid, "max_prefix": h.max_prefix,
                    "frontier": h.frontier})
                self.trace.append((self.t, how, band,
                                   self.cfg.channels[h.channel], h.target))
                self.holds[band] = None
        if not done:
            self._propose_all()
        return r, ev
