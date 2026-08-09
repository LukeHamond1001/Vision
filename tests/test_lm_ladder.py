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

    def test_roster_never_exhausts_on_long_runs(self):
        # regression: the 64-combo retry spin that froze the first pod
        from iga.lm_data_ultrachat import Instruments
        import random as _r
        inst = Instruments(_r.Random(0))
        pos, convos, probed = 0, 0, 0
        for i in range(3000):
            got = inst.maybe_convo(pos)
            pos += 1500  # ~one conversation of stream per slot
            if got:
                convos += 1
                turns, probes = got
                probed += len(probes)
                for pr in probes:
                    self.assertGreaterEqual(len(pr["distractors"]), 3)
                if not probes and inst.pending \
                        and inst.pending[-1].get("plant") is None:
                    inst.pending[-1]["plant"] = pos
                    inst.pending[-1]["due"] = pos \
                        + inst.pending[-1].pop("due_gap")
        self.assertGreater(convos, 400)  # kept producing to the end
        self.assertGreater(probed, 100)  # asks keep flowing too
        w_lane = Lane(Vocab(), random.Random(9))
        toks, _, evs = w_lane.take(120_000)  # weaver deep run
        late_probes = [p for p, k, d in evs if k == "probe" and p > 60_000]
        self.assertGreater(len(late_probes), 0)

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


class TestWidthShaping(unittest.TestCase):
    def test_slowheavy_forward_backward_and_shapes(self):
        from iga.lm_bands import shape_widths
        v = Vocab()
        widths = shape_widths(32, "slowheavy")
        m = BandLM(len(v), d=32, widths=widths)
        self.assertEqual([h.shape[1] for h in
                          m.init_state(2, "cpu")["h"]], widths)
        lane = Lane(v, random.Random(1))
        toks, tgts, _ = lane.take(600)
        x = torch.tensor([toks[:300], toks[300:600]])
        st = m.init_state(2, "cpu")
        logits, st, _ = m(x, st, None)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, len(v)),
            torch.tensor([tgts[:300], tgts[300:600]]).reshape(-1))
        loss.backward()
        self.assertGreater(m.n_params(),
                           BandLM(len(v), d=32).n_params())


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


class TestTransformerSubstrate(unittest.TestCase):
    def test_smoke_train_and_absent_imagination(self):
        # v5.2: standard transformer under the drive layer
        model, drive, vocab, ce0, ce1 = train(
            d=48, lanes=2, T=192, steps=10, device="cpu",
            log_every=100, arch="transformer")
        self.assertLess(ce1, ce0)
        self.assertTrue(drive.audit()["telescoping_exact"])
        # absent instrument holds no authority: frontier proposes
        # without fid readings once a channel is minted
        d = Drive(n_lanes=1, imagination_absent=True)
        d.probe(0, torch.tensor(0.4), gap=100)
        d.earned(0, ok=True)
        d.sweep(losses=[])
        self.assertTrue(any(h["kind"] == "frontier" for h in d.holds[0]))
        self.assertEqual(d.vetoes, 0)


class TestHybridComplete(unittest.TestCase):
    def test_hybrid_trains_ticks_and_lesions(self):
        # v5.3: the complete architecture — attention + slow latents +
        # live imagination instrument + drive
        model, drive, vocab, ce0, ce1 = train(
            d=48, lanes=2, T=128, steps=16, device="cpu",
            log_every=100, arch="hybrid")
        self.assertLess(ce1, ce0)
        self.assertTrue(drive.audit()["telescoping_exact"])
        # slow-band predictors ticked (imagination instrument LIVE)
        self.assertTrue(any(k.startswith("fid:") for k in drive.ema))
        self.assertEqual(drive.bin_band[0], 3)  # carry-band remap
        st = model._st
        self.assertGreater(float(st["h"][3].abs().sum()), 0.0)
        model.lesioned = {3, 4, 5}
        mem = model._mem_tokens(st, 2)
        self.assertEqual(float(mem.abs().sum()), 0.0)  # lesion works
        model.lesioned = set()
        # A24: bands 1/2 have no organs in the hybrid — no maintains
        for lane_holds in drive.holds:
            for h in lane_holds:
                self.assertNotIn(h["key"], ("fid:1", "fid:2"))


class TestSlowWrites(unittest.TestCase):
    def test_gate_closed_at_init_and_write_cost_flows(self):
        # A24 L1: the state barely moves at init (the medium the
        # forward model must predict is stable by construction)
        import torch as t
        from iga.lm_hybrid import SlowCell, HybridLM
        cell = SlowCell(32)
        h = t.randn(4, 32)
        x = t.randn(4, 32)
        h2, z = cell(x, h)
        self.assertLess(float(z), 0.2)   # sigmoid(-2) ~ 0.12
        drift = float((h2 - h).norm() / h.norm())
        self.assertLess(drift, 0.5)
        m = HybridLM(64, d=32, n_layers=1, n_heads=2, max_T=64)
        st = m.init_state(2, "cpu")
        _, st, _ = m(t.randint(0, 64, (2, 64)), st, None)
        wc = m.pop_write_cost()
        self.assertIsNotNone(wc)         # band 3 ticked this chunk
        self.assertTrue(wc.requires_grad)
        self.assertIsNone(m.pop_write_cost())  # popped clean

    def test_absent_bands_never_maintained(self):
        d = Drive(n_lanes=1, absent_bands={1, 2})
        for k in range(1, N_BANDS):
            d.ema[f"fid:{k}"] = 0.01     # everything unhealthy
        d.sweep(losses=[])
        keys = {h["key"] for h in d.holds[0]}
        self.assertNotIn("fid:1", keys)
        self.assertNotIn("fid:2", keys)
        self.assertIn("fid:3", keys)     # present organs still kept

    def test_resume_continues_state_and_steps(self):
        # A26: continuation — model/opt/drive EMAs carry, step
        # numbering continues, trace keeps appending
        import json
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            ck = os.path.join(td, "r.pt")
            m1, d1, _, _, _ = train(d=32, lanes=2, T=96, steps=4,
                                    device="cpu", log_every=1,
                                    arch="hybrid", ckpt=ck)
            # ckpt writes every 500 steps only — save one by hand the
            # way the trainer does, at step 4
            torch.save({"model": m1.state_dict(),
                        "opt": torch.optim.AdamW(
                            m1.parameters()).state_dict(),
                        "step": 4,
                        "drive": {"ema": d1.ema, "records": d1.records,
                                  "minted": sorted(d1.minted),
                                  "holds_settled": len(d1.ledger),
                                  "vetoes": d1.vetoes}}, ck)
            m2, d2, _, _, _ = train(d=32, lanes=2, T=96, steps=3,
                                    device="cpu", log_every=1,
                                    arch="hybrid", ckpt=ck, resume=ck)
            self.assertEqual(d2.step_t, (4 + 3) * 96)  # steps continued
            for k, v in d1.ema.items():   # ema seeded from snapshot
                self.assertIn(k, d2.ema)
            lines = [json.loads(l) for l in open(ck + ".trace.jsonl")]
            self.assertEqual(lines[-1]["step"], 7)

    def test_xl_carry_and_gated_reads(self):
        # A30: information flows across the chunk boundary through
        # attention (XL carry), and matrix reads start gated shut
        import torch as t
        from iga.lm_hybrid import HybridLM
        t.manual_seed(0)
        m = HybridLM(64, d=32, n_layers=2, n_heads=2, max_T=64,
                     store="matrix")
        m.eval()
        for k in m.bands:   # gates shut at init
            self.assertLess(float(t.sigmoid(m.read_gate[str(k)])),
                            0.05)
        a = t.randint(0, 64, (1, 64))
        b = t.randint(0, 64, (1, 64))
        with t.no_grad():
            st = m.init_state(1, "cpu")
            _, st, _ = m(a, st, None)
            self.assertIsNotNone(st["xl"])
            self.assertEqual(st["xl"][0].shape, (1, 64, 32))
            logits_carry, _, _ = m(b, {**st, "xl": [h.clone()
                                   for h in st["xl"]]}, None)
            st_blank = {**st, "xl": [t.zeros_like(h)
                                     for h in st["xl"]]}
            logits_blank, _, _ = m(b, st_blank, None)
        delta = float((logits_carry - logits_blank).abs().mean())
        self.assertGreater(delta, 1e-4)  # last chunk reaches this one

    def test_holdout_probe_runs_and_accumulates(self):
        import os
        shard = "data/uc_lite_smoke"
        if not os.path.exists(os.path.join(shard, "tokens.bin")):
            self.skipTest("local smoke shard absent")
        from iga.lm_data_ultrachat import UltraConveyor, load_tokenizer
        from iga.lm_train import holdout_probe
        from iga.lm_hybrid import HybridLM
        tok = load_tokenizer(os.path.join(shard, "tokenizer.json"))
        m = HybridLM(tok.get_vocab_size(), d=32, n_layers=2,
                     n_heads=2, max_T=256)
        pe = {"conv": UltraConveyor(shard, n_lanes=2),
              "st": m.init_state(2, "cpu"), "agg": {}}
        out1 = holdout_probe(m, pe, 256, "cpu")
        out2 = holdout_probe(m, pe, 256, "cpu")
        n1 = sum(v[2] for v in out1.values()) if out1 else 0
        n2 = sum(v[2] for v in out2.values())
        self.assertGreater(n2, n1)       # cumulative
        self.assertTrue(m.training)      # mode restored

    def test_matrix_store_laws(self):
        # A28: write recovers, capacity holds 8 pairs, decay
        # half-life matches the clock
        import torch as t
        from iga.lm_hybrid import BandMatrix
        t.manual_seed(0)
        bm = BandMatrix(64, decay=0.0)
        with t.no_grad():
            bm.beta.fill_(10.0)          # sigmoid ~ 1: full writes
            M = t.zeros(1, 64, 64)
            xs = [t.randn(1, 64) for _ in range(8)]
            for x in xs:
                M, _ = bm.write(M, x)
            coss = []
            for x in xs:
                k = t.nn.functional.normalize(bm.wk(x), dim=-1)
                v = bm.wv(x)
                back = t.einsum("bij,bj->bi", M, k)
                coss.append(float(t.nn.functional.cosine_similarity(
                    back, v, dim=-1)))
            self.assertGreater(sum(coss) / len(coss), 0.7)  # capacity
            self.assertGreater(coss[-1], 0.95)   # delta rule exact-ish
            # decay half-life: clock-8 band loses half in 8 chunks
            bm8 = BandMatrix(64, decay=1 - 0.5 ** (1 / 8))
            M2 = t.ones(1, 64, 64)
            for _ in range(8):
                M2 = (1 - bm8.decay) * M2
            self.assertAlmostEqual(float(M2[0, 0, 0]), 0.5, places=5)

    def test_hybrid_matrix_trains_and_lesions(self):
        model, drive, vocab, ce0, ce1 = train(
            d=48, lanes=2, T=128, steps=16, device="cpu",
            log_every=100, arch="hybrid", store="matrix")
        self.assertLess(ce1, ce0)
        self.assertTrue(drive.audit()["telescoping_exact"])
        st = model._st
        self.assertIn("M", st)
        self.assertGreater(float(st["M"][3].abs().sum()), 0.0)
        # lesioned bands are skipped in the read loop: with all
        # lesioned, no band contributes (sum over empty = int 0)
        import torch as t
        model.lesioned = {3, 4, 5}
        text = t.randn(2, 8, 48)
        r = sum(model.mats[str(k)].read(st["M"][k], text)
                for k in model.bands if k not in model.lesioned)
        self.assertEqual(r, 0)
        model.lesioned = set()
        self.assertIsNone(model.pop_recon())  # popped by the trainer

    def test_trace_file_written(self):
        import json
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            ck = os.path.join(td, "t.pt")
            train(d=32, lanes=2, T=96, steps=3, device="cpu",
                  log_every=1, arch="hybrid", ckpt=ck)
            trace = ck + ".trace.jsonl"
            self.assertTrue(os.path.exists(trace))
            lines = [json.loads(l) for l in open(trace)]
            self.assertEqual(len(lines), 3)
            self.assertIn("ema", lines[0])
            self.assertIn("vetoes", lines[0])


class TestPrepareStreams(unittest.TestCase):
    def test_spill_equivalence(self):
        # regression: v5.3 runs 1-2 — prep held the whole corpus as a
        # Python list and the OOM killer ended it silently. The sink
        # spills to disk; spilling must not change one byte.
        import hashlib
        import os
        import shutil
        raw = "data/ultrachat_raw.jsonl"
        tokref = "data/uc_tokref/tokenizer.json"
        if not (os.path.exists(raw) and os.path.exists(tokref)):
            self.skipTest("local raw corpus / tokref absent")
        from iga import lm_data_ultrachat as u
        prev = os.environ.get("ULTRACHAT_JSONL")
        os.environ["ULTRACHAT_JSONL"] = raw
        try:
            digests = []
            for name, spill in [("nospill", 10 ** 12), ("spill", 5000)]:
                out = f"results/_prep_{name}"
                shutil.rmtree(out, ignore_errors=True)
                u.prepare(out, n_convos=300, seed=0, instrument_every=1,
                          tokenizer_path=tokref, spill=spill)
                with open(out + "/tokens.bin", "rb") as f:
                    digests.append(hashlib.sha256(f.read()).hexdigest())
            self.assertEqual(digests[0], digests[1])
            self.assertEqual(open("results/_prep_nospill/events.jsonl").read(),
                             open("results/_prep_spill/events.jsonl").read())
        finally:
            for name in ("nospill", "spill"):
                shutil.rmtree(f"results/_prep_{name}", ignore_errors=True)
            if prev is None:
                os.environ.pop("ULTRACHAT_JSONL", None)
            else:
                os.environ["ULTRACHAT_JSONL"] = prev


class TestConveyorEventLookup(unittest.TestCase):
    def test_binary_search_matches_linear_scan(self):
        # regression: v5.3 run 3 — the per-chunk linear scan over ALL
        # lane events was ~30 CPU-hours at full corpus. The
        # searchsorted window must select the identical events.
        import os
        shard = "data/uc_lite_smoke"
        if not os.path.exists(os.path.join(shard, "tokens.bin")):
            self.skipTest("local smoke shard absent")
        from iga.lm_data_ultrachat import UltraConveyor
        conv = UltraConveyor(shard, n_lanes=4)
        seen = 0
        for _ in range(200):
            cursors = list(conv.cursor)
            _, _, events = conv.chunk(256)
            for lane, evs in enumerate(events):
                c = cursors[lane]
                lo, hi = lane * conv.seg, (lane + 1) * conv.seg
                if c + 256 + 1 > hi:
                    c = lo
                linear = [(e["pos"] - c, e["kind"], e)
                          for e in conv.lane_events[lane]
                          if c <= e["pos"] < c + 256]
                self.assertEqual(evs, linear)
                seen += len(evs)
        self.assertGreater(seen, 50)  # the comparison actually bit


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
