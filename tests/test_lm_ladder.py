"""v5.0 law tests — the language ladder keeps the campaign's laws."""

import random
import unittest

import torch

from iga import lm_gen
from iga.lm_bands import BandLM, FAST_BANDS, N_BANDS
from iga.lm_conveyor import Vocab, Conveyor
from iga.lm_drive import Drive
from iga.lm_train import train


class TestGeneratorLaws(unittest.TestCase):
    def test_probe_positions_exact(self):
        rng = random.Random(3)
        for _ in range(5):
            toks, meta = lm_gen.saga(rng)
            for p in meta["probes"]:
                self.assertEqual(toks[p["pos"]], p["answer"])
            toks, meta = lm_gen.episode(rng)
            for p in meta["probes"]:
                self.assertEqual(toks[p["pos"]], p["answer"])

    def test_ok_iff_verified_success(self):
        rng = random.Random(4)
        seen = set()
        for _ in range(50):
            toks, meta = lm_gen.episode(rng)
            self.assertEqual("<ok>" in toks, meta["ok"])
            seen.add(meta["ok"])
        self.assertEqual(seen, {True, False})  # the label varies


class TestDriveLaws(unittest.TestCase):
    def test_closed_loop_pays_zero(self):
        d = Drive(n_lanes=1)
        d.ema["fid:1"] = 0.10  # below healthy -> maintain proposed
        d.scene_start(0, "saga")
        self.assertTrue(any(h["key"] == "fid:1" for h in d.holds[0]))
        d.scene_end(0, ok=False, scene_type="saga", losses=[])
        for e in d.ledger:
            if e["key"] == "fid:1":
                self.assertEqual(e["pay"], 0.0)  # ended where it opened

    def test_telescoping_reconstructs(self):
        d = Drive(n_lanes=1)
        d.ema["fid:1"] = 0.10
        d.scene_start(0, "saga")
        d.ema["fid:1"] = 0.30  # progress happened during the scene
        d.scene_end(0, ok=False, scene_type="saga", losses=[])
        for e in d.ledger:
            self.assertAlmostEqual(e["pay"], e["w"] * (e["phi0"] - e["phi1"]),
                                   places=12)

    def test_unminted_never_proposed_and_ok_mints(self):
        d = Drive(n_lanes=1)
        for k in range(1, N_BANDS):
            d.ema[f"fid:{k}"] = 0.5  # healthy: no maintains, no vetoes
        d.scene_start(0, "saga")
        self.assertFalse(any(h["kind"] == "frontier" for h in d.holds[0]))
        d.probe(0, torch.tensor(0.4), gap=100, scene_type="saga")
        d.scene_end(0, ok=True, scene_type="saga", losses=[])
        self.assertIn(("saga", "recall:saga:b0"), d.minted)
        d.scene_start(0, "saga")
        self.assertTrue(any(h["kind"] == "frontier" for h in d.holds[0]))

    def test_no_hold_outlives_scene(self):
        d = Drive(n_lanes=1)
        d.ema["fid:1"] = 0.10
        d.scene_start(0, "saga")
        self.assertTrue(d.holds[0])
        d.scene_end(0, ok=False, scene_type="saga", losses=[])
        self.assertFalse([h for h in d.holds[0] if h["scope"] == "scene"])


class TestBandLaws(unittest.TestCase):
    def test_scene_mask_resets_fast_persists_slow(self):
        v = Vocab()
        m = BandLM(len(v), d=32)
        torch.manual_seed(0)
        st = m.init_state(1, "cpu")
        toks = torch.tensor([v.encode("mira kept a silver key .".split())])
        _, st, _ = m(toks, st, None)
        st["h"][5] = torch.ones_like(st["h"][5])  # slow band carries something
        fast_before = [st["h"][k].clone() for k in FAST_BANDS]
        tok = torch.tensor([[v.idx["<scene>"]]])
        _, st, _ = m(tok, st, {0: [0]})
        fresh = m.init_state(1, "cpu")
        _, fresh, _ = m(tok, fresh, None)
        for k in FAST_BANDS:
            self.assertTrue(torch.allclose(st["h"][k], fresh["h"][k]),
                            f"band {k} not reset to fresh at scene start")
        self.assertTrue(bool((st["h"][5] == 1.0).all()),
                        "slow band must persist across scene starts")


class TestEndToEnd(unittest.TestCase):
    def test_smoke_train_audit(self):
        model, drive, vocab, ce0, ce1 = train(d=48, lanes=2, T=192, steps=12,
                                              device="cpu", log_every=100)
        self.assertLess(ce1, ce0)
        audit = drive.audit()
        self.assertTrue(audit["telescoping_exact"])
        self.assertTrue(audit["scoped"])
        self.assertGreater(audit["holds"], 0)


if __name__ == "__main__":
    unittest.main()
