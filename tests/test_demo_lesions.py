"""Laws of scripts/demo_lesions.py (the centerpiece demos).

  D1  The probe set is seeded and frozen: the same facts every run,
      n per bin, disjoint (name, object) pairs.
  D2  Gap invariant: every fact is asked at its bin — in-ctx facts are
      inside the pending window at the ask; short/b4 facts are at least
      their bin's chunks back in the committed stream.
  D3  Readings are paired and silent: the three conditions are read on
      the same committed state; the life is not advanced by a read; the
      model's switches are off afterwards.
  D4  The sign test is exact: all-discordant-for-base gives 1/2^n; no
      discordant pairs gives p = 1; ties never count.
  D5  The summary carries n, speech recall and p_true per condition and
      bin, and the two paired tests per bin.
"""

import argparse
import importlib.util
import os
import types

import torch

from iga.lm_hybrid import HybridLM
from iga.lm_vocab import PRESS_TOKENS

_spec = importlib.util.spec_from_file_location(
    "demo_lesions", os.path.join(os.path.dirname(__file__), "..",
                                 "scripts", "demo_lesions.py"))
DL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DL)

KW = dict(store="matrix", keyed="logit", norm_mix=True, aux_trunk=0.2,
          use_xl=False, gate_init=-2.0, clocks={3: 1, 4: 8, 5: 64, 6: 512})
V = 512
SPECIALS = {"<pad>": 500, "<eot_human>": 501, "<eot_model>": 502,
            **{t: 503 + i for i, t in enumerate(PRESS_TOKENS)}}
WORDS = {}


class _Tok:
    """words get stable ids below 500 (first-seen order); decode
    returns the words back so 'said' can be checked."""
    def token_to_id(self, t):
        return SPECIALS[t]

    def encode(self, text):
        ids = []
        for w in text.split():
            if w in SPECIALS:
                ids.append(SPECIALS[w])
            else:
                ids.append(WORDS.setdefault(w, len(WORDS) + 1))
        return types.SimpleNamespace(ids=ids)

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in WORDS.items()}
        return " ".join(inv.get(i, f"<{i}>") for i in ids)

    def get_vocab_size(self):
        return V


def _args(n=3, bins="in-ctx,short,b4", T=64):
    return argparse.Namespace(n=n, bins=bins, T=T, filler="", max_new=4,
                              device="cpu", reply=False)


def _model():
    torch.manual_seed(0)
    m = HybridLM(V, d=64, n_layers=2, n_heads=4, max_T=64, **KW)
    with torch.no_grad():
        for k in m.alpha:
            m.alpha[k].fill_(0.5)
    return m


def _run(tmp_path=None, **kw):
    a = _args(**kw)
    return DL.run(a, m=_model(), tok=_Tok(),
                  out_dir=str(tmp_path) if tmp_path else None)


def test_D1_probe_set_seeded_frozen_disjoint():
    import random
    f1 = DL.build_facts(random.Random(DL.SEED), 4, ["in-ctx", "b4"])
    f2 = DL.build_facts(random.Random(DL.SEED), 4, ["in-ctx", "b4"])
    assert f1 == f2 and len(f1) == 8
    assert len({(f["name"], f["obj"]) for f in f1}) == 8
    assert [f["bin"] for f in f1] == ["in-ctx"] * 4 + ["b4"] * 4


def test_D2_gap_invariant_and_D3_paired_silent_reads(tmp_path):
    res = _run(tmp_path)
    T = res["probe_set"]["T"]
    asked = res["asked_at"]
    for r in res["reads"]:
        gap_chunks = (asked["committed"] - r["planted_at"]) / T
        if r["bin"] == "in-ctx":
            assert r["planted_at"] >= asked["committed"], r
        else:
            assert gap_chunks >= DL.BIN_CHUNKS[r["bin"]], (r["bin"], gap_chunks)
        for cond in DL.CONDITIONS:
            assert set(r[cond]) == {"said", "text", "p_true"}
            assert 0.0 <= r[cond]["p_true"] <= 1.0
    # the reads did not advance the life
    assert res["pos"] == asked["pos"]
    assert res["committed"] == asked["committed"]
    # evidence files exist and the probe set was written first
    assert os.path.exists(tmp_path / "probe_set.json")
    assert os.path.exists(tmp_path / "demo_lesions.json")
    assert os.path.exists(tmp_path / "transcripts.md")


def test_D3_switches_off_after_reading():
    m = _model()
    a = _args()
    DL.run(a, m=m, tok=_Tok(), out_dir=None)
    assert m.lesioned == set() and m.store_read_off is False


def test_D4_sign_test_exact():
    p, plus, minus = DL.sign_test_one_sided([(True, False)] * 5)
    assert (plus, minus) == (5, 0) and abs(p - 1 / 32) < 1e-12
    p, plus, minus = DL.sign_test_one_sided([(True, True), (False, False)])
    assert p == 1.0 and (plus, minus) == (0, 0)
    p, _, _ = DL.sign_test_one_sided([(True, False)] * 3 + [(False, True)] * 3)
    assert abs(p - 0.65625) < 1e-12     # P(X >= 3), X ~ Bin(6, .5)


def test_D5_summary_shape():
    res = _run(n=2, bins="in-ctx,short")
    s = res["summary"]
    assert list(s) == ["in-ctx", "short"]
    for b, row in s.items():
        assert row["n"] == 2
        for cond in DL.CONDITIONS:
            assert set(row[cond]) == {"speech_recall", "p_true_mean"}
        for cond in ("bands", "store", "both"):
            assert set(row[f"sign_base_gt_{cond}"]) == {"p", "plus", "minus"}
    th = res["thread"]
    assert set(th) == set(DL.CONDITIONS) and th["none"]["of"] == 4
