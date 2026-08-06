"""v5.0 law tests — the endless-dialogue ladder keeps the campaign's laws."""

import random
import unittest

import torch

from iga import lm_gen
from iga.lm_bands import BandLM, N_BANDS
from iga.lm_conveyor import Vocab, Lane
from iga.lm_drive import Drive, horizon
from iga.lm_train import train


class TestWeaverLaws(unittest.TestCase):
    def test_probe_positions_exact_and_all_thanked(self):
        v = Vocab()
        lane = Lane(v, random.Random(3))
        toks, _, evs = lane.take(6000)
        probes = [(p, d) for p, k, d in evs if k == "probe"]
        earned = [d["ok"] for p, k, d in evs if k == "earned"]
        self.assertGreater(len(probes), 0)
        for p, d in probes:
            self.assertEqual(toks[p], d["answer"],
                             "probe position must be the answer token id")
            self.assertGreater(d["gap"], 0)
        self.assertGreater(len(earned), 0)
        self.assertTrue(all(earned),
                        "A7: all-good data — every exchange is thanked")

    def test_failure_branches_survive_for_real_data_rounds(self):
        v = Vocab()
        lane = Lane(v, random.Random(5))
        lane.weaver.correct_rate = 0.5
        lane.weaver.success_rate = 0.5
        _, _, evs = lane.take(8000)
        earned = [d["ok"] for p, k, d in evs if k == "earned"]
        self.assertIn(False, earned)  # the machinery still selects

    def test_only_two_special_tokens_no_scene(self):
        self.assertNotIn("<scene>", lm_gen.LEXICON)
        self.assertNotIn("</scene>", lm_gen.LEXICON)
        self.assertNotIn("<ok>", lm_gen.LEXICON)
        self.assertIn("<eot_human>", lm_gen.LEXICON)
        self.assertIn("<eot_model>", lm_gen.LEXICON)


class TestDriveLaws(unittest.TestCase):
    def test_closed_loop_pays_zero(self):
        d = Drive(n_lanes=1)
        d.ema["fid:1"] = 0.10  # below healthy -> maintain proposed
        d.sweep(losses=[])
        self.assertTrue(any(h["key"] == "fid:1" for h in d.holds[0]))
        d.step_t += horizon(1) + 1
        d.sweep(losses=[])     # ema unchanged: ends where it opened
        settled = [e for e in d.ledger if e["key"] == "fid:1"]
        self.assertTrue(settled)
        self.assertEqual(settled[0]["pay"], 0.0)

    def test_telescoping_reconstructs(self):
        d = Drive(n_lanes=1)
        d.ema["fid:1"] = 0.10
        d.sweep(losses=[])
        d.ema["fid:1"] = 0.19  # progress during the hold
        d.step_t += horizon(1) + 1
        d.sweep(losses=[])
        for e in d.ledger:
            self.assertAlmostEqual(e["pay"], e["w"] * (e["phi0"] - e["phi1"]),
                                   places=12)

    def test_thanks_mints_and_unminted_never_proposed(self):
        d = Drive(n_lanes=1)
        for k in range(1, N_BANDS):
            d.ema[f"fid:{k}"] = 0.5  # healthy: no maintains, no vetoes
        d.sweep(losses=[])
        self.assertFalse(any(h["kind"] == "frontier" for h in d.holds[0]))
        d.probe(0, torch.tensor(0.4), gap=100)
        d.earned(0, ok=False)     # "not right" mints nothing
        d.sweep(losses=[])
        self.assertFalse(any(h["kind"] == "frontier" for h in d.holds[0]))
        d.earned(0, ok=True)      # "thanks" mints the channel it followed
        d.sweep(losses=[])
        self.assertTrue(any(h["kind"] == "frontier" for h in d.holds[0]))

    def test_expiry_pays_exactly_zero_and_hold_dies(self):
        d = Drive(n_lanes=1)
        for k in range(1, N_BANDS):
            d.ema[f"fid:{k}"] = 0.5
        d.probe(0, torch.tensor(0.4), gap=100)
        d.earned(0, ok=True)
        d.sweep(losses=[])
        h = [h for h in d.holds[0] if h["kind"] == "frontier"][0]
        d.step_t = h["due"] + 1   # horizon passes, no readings in-window
        d.sweep(losses=[])
        exp = [e for e in d.ledger if e["key"] == h["key"]]
        self.assertTrue(exp)
        self.assertEqual(exp[0]["pay"], 0.0)
        self.assertNotIn(h, d.holds[0])


class TestBandLaws(unittest.TestCase):
    def test_state_persists_and_lesion_zeroes(self):
        v = Vocab()
        m = BandLM(len(v), d=32)
        st = m.init_state(1, "cpu")
        toks = torch.tensor([v.encode("by the way mira kept a red key .".split())])
        _, st, _ = m(toks, st, None)
        self.assertGreater(float(st["h"][0].abs().sum()), 0.0)
        st = m.detach_state(st)
        _, st2, _ = m(toks, st, None)   # continuity: nothing resets, ever
        self.assertGreater(float(st2["h"][0].abs().sum()), 0.0)
        m.lesioned = {5}
        reads = m._read(st2)
        self.assertEqual(float(reads[5].abs().sum()), 0.0)
        m.lesioned = set()


class TestCalibration(unittest.TestCase):
    def test_calibrate_produces_dataset_constants(self):
        from iga.lm_calibrate import run
        import os
        out = "results/lm_constants_test.json"
        c = run(chunks=4, T=256, lanes=2, out=out)
        self.assertEqual(set(int(k) for k in c["horizons"]),
                         set(range(N_BANDS)))
        self.assertTrue(c["chance_floors"])   # at least one bin measured
        os.remove(out)

    def test_drive_consumes_constants(self):
        d = Drive(n_lanes=1, constants={"horizons": {"2": 99999},
                                        "fid_floor": {"2": 0.9}})
        self.assertEqual(d.horizon_for(2), 99999)
        self.assertEqual(d.horizon_for(1), horizon(1))  # default fallback


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
