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

Sources: results/v40_sequencing.json (the v4.0 fleet, 13 rows) and
results/v12_fleet.json (the v1.2 pod triples, recovered to main from
the pod branches)."""

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

    print("\n(read the CIs as effect-size intervals: at n=5 the exact "
          "sign test cannot reach p<0.05 even at 5/5)")


if __name__ == "__main__":
    main()
