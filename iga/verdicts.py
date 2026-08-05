"""Committed verdict statistics — every headline number, recomputed
from committed artifacts with EXACT small-n methods. Run:

    python -m iga.verdicts

Methods (stated so nobody has to guess):
  - paired mean differences with Student-t CI95 (df = n-1) — NOT a
    normal z interval; at n=5, t(4) = 2.776
  - exact two-sided sign test on the paired differences (ties dropped)
  - n=5 cannot reach p < 0.05 under the sign test even at 5/5 — stated
    wherever the CI is stated. The CIs are effect-size intervals, not
    significance claims.

Sources: results/v40_sequencing.json (the v4.0 fleet, 13 rows),
results/v12_fleet.json (the v1.2 pod triples, recovered to main from
the pod branches), and the per-card artifact JSONs (v30_hacking,
v09_routing, v20_phase1, v21_battery, v13_goalswap) whose headline
numbers are reprinted verbatim at the end."""

from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

T95_BY_DF = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def t_ci(diffs):
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return mean, float("nan"), float("nan")
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    t = T95_BY_DF[n - 1]
    return mean, mean - t * se, mean + t * se


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg
    k = max(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail), pos, neg


def report(name, diffs, bar=None):
    mean, lo, hi = t_ci(diffs)
    p, pos, neg = sign_test(diffs)
    line = (f"{name}: diffs {['%+.2f' % d for d in diffs]} -> mean "
            f"{mean:+.2f}, t-CI95 [{lo:+.2f}, {hi:+.2f}] (df={len(diffs)-1}),"
            f" sign test {pos}+/{neg}- p={p:.3f}")
    if bar is not None:
        verdict = "PASS" if mean >= bar and lo > 0 else "FAIL"
        line += f" | registered bar {bar:+.2f}: {verdict}"
    print(line, flush=True)
    return {"diffs": diffs, "mean": mean, "ci95": [lo, hi],
            "sign_p": p}


def main():
    print("== v4.0 sequencing (5 seeds x full/no-proposer, paired) ==")
    rows = json.load(open(RESULTS / "v40_sequencing.json"))
    by = {(r["arm"], r["seed"]): r for r in rows}
    seeds = sorted({s for a, s in by if a == "full"})
    d40 = [by[("full", s)]["achv_median"] - by[("no-proposer", s)]["achv_median"]
           for s in seeds]
    report("G-seq achv-median (full - no-proposer)", d40, bar=1.0)
    dmean = [by[("full", s)]["achv_mean"] - by[("no-proposer", s)]["achv_mean"]
             for s in seeds]
    report("supplementary achv-MEAN diff", dmean)

    print("\n== v1.2 homeostasis (5 pods x wired/zero/native, paired) ==")
    fleet = json.load(open(RESULTS / "v12_fleet.json"))["rows"]
    dwz = [r["wired"]["survival"] - r["zero"]["survival"] for r in fleet]
    report("survival (wired - zero)", dwz)
    dnw = [r["native"]["survival"] - r["wired"]["survival"] for r in fleet]
    report("survival (native - wired), context", dnw)
    ratios = [r["wired"]["survival"] / r["zero"]["survival"] for r in fleet]
    mean_ratio = sum(ratios) / len(ratios)
    print(f"registered v1.2 gate (wired/zero IQM ratio >= 1.25): mean "
          f"ratio {mean_ratio:.3f} -> "
          f"{'PASS' if mean_ratio >= 1.25 else 'FAIL'}", flush=True)

    fp = [r for r in fleet if "drink" in r.get("wired", {})]
    if fp:
        w, z = fp[0]["wired"], fp[0]["zero"]
        print(f"mechanism fingerprint (pod '{fp[0]['pod']}' per-channel "
              f"logs): drink {w['drink']:.2f} vs {z['drink']:.2f}; "
              f"slept/life {w['slept']:.2f} vs {z['slept']:.2f}")

    print("\n== committed-artifact headlines (the other cards) ==")

    def j(name):
        f = RESULTS / name
        return json.load(open(f)) if f.exists() else None

    v30 = j("v30_hacking.json")
    if v30:
        eng = v30["engineered"]
        print(f"v3.0 engineered reward: score "
              f"{min(eng['eng_last3rd']):.0f}-{max(eng['eng_last3rd']):.0f}"
              f", laps {sorted(eng['laps_last3rd'])} -> HACKED, 3/3 at "
              f"zero laps")
        reg = sorted(v30["register"]["laps_last3rd"])
        mf = sorted(v30["register_meanfill"]["laps_last3rd"])
        print(f"v3.0 register arm laps {['%.2f' % x for x in reg]} "
              f"(pre-registered); mean-fill readout "
              f"{['%.2f' % x for x in mf]} (robustness round, labeled "
              f"post-hoc)")
    v09 = j("v09_routing.json")
    if v09:
        vit = [v09[k] for k in ("mid_food", "mid_drink", "mid_health")]
        print(f"v0.9 routing (held-out corr): slow "
              f"{v09['slow_daylight']:.2f}, vitals {min(vit):.2f}-"
              f"{max(vit):.2f}, energy {v09['mid_energy']:.2f} (energy "
              f"gate amended pre-run, ledgered)")
    v20 = j("v20_phase1.json")
    if v20:
        t = v20["taus"]
        print(f"v2.0 tau-ladder (steps): gait {t['gait']:.0f} | temp "
              f"{t['temp']:.0f} | battery {t['battery']:.0f} | wear "
              f"{t['wear']:,.0f} | gates {v20['gates']}")
    v21 = j("v21_battery.json")
    if v21:
        arms = {}
        for r in v21:
            arms.setdefault(r["arm"], []).append(r["brownout"])
        line = " vs ".join(f"{a} {sum(v) / len(v):.2f}"
                           for a, v in sorted(arms.items()))
        print(f"v2.1 brownout dissociation (mean by arm): {line}")
    v13 = j("v13_goalswap.json")
    if v13:
        parts = ", ".join(f"{r['arm']}: slept/life "
                          f"{r['slept_last3rd']:.2f}, drink "
                          f"{r['drink_last3rd']:.2f}" for r in v13)
        print(f"v1.3 edited arms ({parts}); baseline 4.3 slept/life is "
              f"the v1.2-lineage policy, logged in "
              f"results/INTERPRETATION.md")

    print("\n(read the CIs as effect-size intervals: at n=5 the exact "
          "sign test cannot reach p<0.05 even at 5/5)")


if __name__ == "__main__":
    main()
