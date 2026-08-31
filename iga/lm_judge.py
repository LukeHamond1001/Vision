"""v10 — the frozen PUBLIC-instrument judge (A64 frozen-instrument
law, spec 2b judge-grounding clause).

Two instruments, both auditable by anyone:
  - documents (reading/study material): the source's own public
    quality column where it ships one (FineWeb-Edu int_score 0-5,
    Magpie-Ultra ArmoRM quality) -> grade_doc / grade_upstream.
  - dialogue (the spine): a small frozen grader CALIBRATED ONCE on
    HelpSteer2's public human helpfulness ratings (0-4) over
    deterministic text features -> grade_dialogue. Coefficients are
    frozen constants in THIS file; `python -m iga.lm_judge calibrate`
    re-derives them from public data so anyone can verify.

The judge SELECTS twice: passes_floor() is admission (below-floor
exchanges are DROPPED — no bad data ever enters, never negatively
rewarded), press_for() is selection (+2/+1/silence via per-stage
thresholds — density staging lives in the THRESHOLDS; the scorer
never changes across the flash).

Freeze protocol: JUDGE_VERSION + the full constants block are
copied into every shard manifest; a fixture test locks featurize()
and grade_dialogue() to 6 decimals; the heartbeat's tail audit
re-grades sampled text against the manifest copy.
"""

import gzip
import json
import math
import re

JUDGE_VERSION = "v10.0"

# press thresholds by stage: q >= q2 -> +2, q >= q1 -> +1, else
# silence. Quantile-calibrated on 4,004 UltraChat exchanges
# (2026-08-19) to the stage press-density targets — infancy 35%
# total (+2 15/+1 20; caregiver-dense), childhood 23% (8/15),
# adolescence 12% (4/8; prophets must predict), tail 18% (8/10;
# the highest-pedigree material earns MORE selection again, so the
# anneal is dense -> sparse -> tail-rich, not monotone). The step-8
# full-corpus calibration re-derives these per real stage mixes;
# every shard manifest embeds the values it was built with.
JUDGE = {
    "floor": 0.30,
    "q1": {"infancy": 0.773, "childhood": 0.795,
           "adolescence": 0.815, "tail": 0.804},
    "q2": {"infancy": 0.810, "childhood": 0.823,
           "adolescence": 0.835, "tail": 0.823},
    "doc_score_div": 5.0,          # FineWeb-Edu int_score 0-5 -> q
    "audit_every": 997,            # primes: no day-cadence aliasing
    "audit_every_tail": 101,
}

# The pre-registered stage press-density TARGETS the thresholds
# above were calibrated to: {stage: (frac_+2, frac_+1)} of KEPT
# (post-floor) exchanges. The pod-side freeze (freeze-judge, step
# 8) re-derives q1/q2 from the REAL per-stage source mixes against
# these same targets — the targets never move, only the quantiles.
DENSITY = {"infancy": (0.15, 0.20), "childhood": (0.08, 0.15),
           "adolescence": (0.04, 0.08), "tail": (0.08, 0.10)}


def stage_thresholds(qs_by_stage, density=None, min_n=200):
    """Quantile q1/q2 from real per-stage q samples. Returns a
    freeze dict ({"q1": .., "q2": .., "n": ..}) — write it to json,
    commit it, and pass it to the builder via --judge-thresholds."""
    density = density or DENSITY
    out = {"q1": {}, "q2": {}, "n": {}}
    for st, qs in qs_by_stage.items():
        p2, p1 = density[st]
        kept = sorted(q for q in qs if q >= JUDGE["floor"])
        assert len(kept) >= min_n, \
            f"{st}: {len(kept)} kept samples < {min_n}"
        def _q(frac):
            k = max(0, min(len(kept) - 1,
                           int(len(kept) * (1 - frac))))
            return round(kept[k], 4)
        out["q2"][st] = _q(p2)
        out["q1"][st] = min(_q(p2 + p1), out["q2"][st])
        out["n"][st] = len(kept)
    return out


def freeze_stage_thresholds(t):
    """Load a stage-threshold freeze (path or dict with q1/q2) into
    the live JUDGE so the builder grades — and its manifest copies —
    the frozen values. The SCORER (COEF/BIAS/featurize) never
    changes; only the density staging quantiles do (the split this
    module's freeze protocol pre-registers)."""
    if not isinstance(t, dict):
        with open(t) as f:
            t = json.load(f)
    for key in ("q1", "q2"):
        assert set(t[key]) == set(JUDGE[key]), \
            f"stage set mismatch in freeze: {sorted(t[key])}"
        JUDGE[key] = {k: float(v) for k, v in t[key].items()}
    return JUDGE

STOPWORDS = frozenset(
    ("the a an and or but if then than that this these those is are "
     "was were be been being have has had do does did will would can "
     "could should may might must not no nor so to of in on at by "
     "for with about as into from up down out over under again there "
     "here when where why how all any both each few more most other "
     "some such only own same very just it its he she they them his "
     "her their we you i me my your our us what which who whom"
     ).split())

REFUSAL_RE = re.compile(
    r"\b(i can(?:no|')t|i cannot|i'm sorry,? but|as an ai\b|"
    r"i am not able to|i won't be able)\b", re.I)
BULLET_RE = re.compile(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+")
WORD_RE = re.compile(r"[a-z0-9']+")

FEATURE_NAMES = (
    "log_len_m", "too_short", "too_long", "ttr", "stop_rate",
    "rep4", "overlap", "interrog_answered", "struct", "ascii_ratio",
    "refusal", "verb_ratio", "digit_rate", "code",
    "log_sents", "avg_sent_len", "newline_rate", "first_person",
    "hedging", "log_len_h")

SENT_RE = re.compile(r"[.!?]+")
FP_RE = re.compile(r"\b(i|i'm|i'll|my|me)\b")
HEDGE_RE = re.compile(r"\b(maybe|perhaps|might|possibly|i think|"
                      r"not sure)\b")


def _words(t):
    return WORD_RE.findall(t.lower())


def featurize(h_text, m_text):
    """Deterministic, cheap (pure string ops) — order = FEATURE_NAMES."""
    mw = _words(m_text)
    hw = _words(h_text)
    n = len(mw)
    ttr = (len(set(mw)) / n) if n else 0.0
    stop = (sum(1 for w in mw if w in STOPWORDS) / n) if n else 0.0
    if n >= 4:
        grams = [tuple(mw[i:i + 4]) for i in range(n - 3)]
        rep4 = 1.0 - len(set(grams)) / len(grams)
    else:
        rep4 = 0.0
    hc = {w for w in hw if w not in STOPWORDS}
    mc = {w for w in mw if w not in STOPWORDS}
    overlap = (len(hc & mc) / len(hc)) if hc else 0.0
    ascii_n = sum(1 for c in m_text if ord(c) < 128)
    ascii_ratio = (ascii_n / len(m_text)) if m_text else 1.0
    return [
        math.log1p(n),
        1.0 if n < 8 else 0.0,
        1.0 if n > 900 else 0.0,
        ttr,
        stop,
        rep4,
        overlap,
        1.0 if ("?" in h_text and n >= 8) else 0.0,
        min(len(BULLET_RE.findall(m_text)), 5) / 5.0,
        ascii_ratio,
        1.0 if REFUSAL_RE.search(m_text[:160]) else 0.0,
        math.log1p(n / max(1, len(hw))),
        (sum(1 for c in m_text if c.isdigit()) / len(m_text))
        if m_text else 0.0,
        1.0 if "```" in m_text else 0.0,
        math.log1p(max(1, len(SENT_RE.findall(m_text)))),
        (n / max(1, len(SENT_RE.findall(m_text)))),
        m_text.count("\n") / max(1, len(m_text)) * 100,
        (len(FP_RE.findall(m_text.lower())) / n) if n else 0.0,
        (len(HEDGE_RE.findall(m_text.lower())) / n) if n else 0.0,
        math.log1p(len(hw)),
    ]


# ---- frozen dialogue grader (ridge on HelpSteer2, raw-space) ----
# Derived by `calibrate` on nvidia/HelpSteer2 train.jsonl.gz
# (lam=1.0) and hand-frozen 2026-08-19. Honest instrument card:
# val r(helpfulness) 0.248, MAE 0.977 — a WEAK ranker of expert
# quality; val top-10% precision(help>=3) 0.835 vs 0.717 base — a
# USABLE selector. Its jobs are the floor and the coarse top-slice,
# on sources already curated upstream; fine ranking rides public
# upstream columns (grade_doc/grade_upstream) where they exist.
# Prediction is helpfulness-hat in [0,4]-space; q = clip(pred/4).
COEF = [0.3119351006,
        0.4926008621,
        -0.2309325212,
        -0.2144300657,
        -0.6083498387,
        -1.9294522862,
        0.0997822644,
        0.1084618398,
        0.2671483038,
        1.9454096011,
        -0.5900941259,
        0.0207730851,
        -3.8791156550,
        0.0606072179,
        -0.2213118933,
        -0.0020167615,
        -0.0561700630,
        0.7251321575,
        -10.2614838035,
        -0.0459160527]
BIAS = 0.4214634712


def grade_dialogue(h_text, m_text):
    if COEF is None:
        raise RuntimeError("judge unfrozen: run calibrate and freeze "
                           "COEF/BIAS constants first")
    x = featurize(h_text, m_text)
    pred = BIAS + sum(c * v for c, v in zip(COEF, x))
    return max(0.0, min(1.0, pred / 4.0))


def grade_doc(int_score):
    """Public upstream score column (FineWeb-Edu 0-5 etc.) -> q."""
    return max(0.0, min(1.0, float(int_score) / JUDGE["doc_score_div"]))


def grade_upstream(q_raw):
    """Already-normalized public upstream quality in [0,1] (e.g.
    Magpie-Ultra ArmoRM score rescaled by its adapter)."""
    return max(0.0, min(1.0, float(q_raw)))


def passes_floor(q):
    return q >= JUDGE["floor"]


def press_for(q, stage):
    if q >= JUDGE["q2"][stage]:
        return 2
    if q >= JUDGE["q1"][stage]:
        return 1
    return 0


class AuditWriter:
    """Sampled human-readable audit rows -> judge_audit.jsonl.
    Densified in each life's final 10% and on every correction."""

    def __init__(self, path):
        self.f = open(path, "w")
        self.i = 0

    def maybe(self, stage, life, pos, q, press, h, m,
              tail=False, force=False):
        self.i += 1
        every = (JUDGE["audit_every_tail"] if tail
                 else JUDGE["audit_every"])
        if not force and self.i % every:
            return
        self.f.write(json.dumps({
            "i": self.i, "stage": stage, "life": life, "pos": pos,
            "q": round(q, 4), "press": press,
            "h": h[:200], "m": m[:300]}) + "\n")

    def close(self):
        self.f.close()


# ---------------- calibration (public, re-runnable) ----------------

def _rows(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            yield r["prompt"], r["response"], float(r["helpfulness"])


def calibrate(train_path, val_path, lam=1.0):
    import numpy as np
    X, y = [], []
    for h, m, help_ in _rows(train_path):
        X.append(featurize(h, m))
        y.append(help_)
    X, y = np.array(X), np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs.T @ (y - y.mean()))
    coef = w / sd                       # fold back to raw space
    bias = float(y.mean() - (w * mu / sd).sum())

    def score(path):
        Xv, yv = [], []
        for h, m, help_ in _rows(path):
            Xv.append(featurize(h, m))
            yv.append(help_)
        Xv, yv = np.array(Xv), np.array(yv)
        pred = Xv @ coef + bias
        r = float(np.corrcoef(pred, yv)[0, 1])
        mae = float(np.abs(pred - yv).mean())
        # top-decile precision: of the grader's top 10%, what
        # fraction of true helpfulness >= 3 (the selection job)
        k = max(1, len(yv) // 10)
        top = np.argsort(-pred)[:k]
        prec = float((yv[top] >= 3).mean())
        base = float((yv >= 3).mean())
        return r, mae, prec, base

    tr = score(train_path)
    va = score(val_path)
    print(f"train  r {tr[0]:.3f}  mae {tr[1]:.3f}  "
          f"top10% precision(help>=3) {tr[2]:.3f} vs base {tr[3]:.3f}")
    print(f"val    r {va[0]:.3f}  mae {va[1]:.3f}  "
          f"top10% precision(help>=3) {va[2]:.3f} vs base {va[3]:.3f}")
    print("\n# paste into lm_judge.py to freeze:")
    print("COEF = [" + ",\n        ".join(
        f"{c:.10f}" for c in coef) + "]")
    print(f"BIAS = {bias:.10f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--train", default="data/helpsteer2_train.jsonl.gz")
    c.add_argument("--val", default="data/helpsteer2_val.jsonl.gz")
    c.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    if a.cmd == "calibrate":
        calibrate(a.train, a.val, a.lam)
