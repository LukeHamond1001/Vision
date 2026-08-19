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

    def test_xl_dropout_trains_blind_sometimes_evals_always(self):
        # A34: in train mode the carry is dropped ~half the time;
        # in eval mode it is always used
        import torch as t
        from iga.lm_hybrid import HybridLM
        t.manual_seed(0)
        m = HybridLM(64, d=32, n_layers=2, n_heads=2, max_T=64)
        st = m.init_state(1, "cpu")
        x = t.randint(0, 64, (1, 64))
        _, st, _ = m(x, st, None)          # prime the cache
        m.train()
        used = []
        for _ in range(40):
            _, st, _ = m(x, st, None)
            used.append(m._xl_used)
        self.assertIn(True, used)
        self.assertIn(False, used)
        m.eval()
        for _ in range(5):
            _, st, _ = m(x, st, None)
            self.assertTrue(m._xl_used)

    def test_v60_machine_no_xl_gated_dropout_reads(self):
        # A36: THE machine — xl off (cache stays None), matrix reads
        # gated AND dropped-out in training, always on in eval
        import torch as t
        from iga.lm_hybrid import HybridLM
        t.manual_seed(0)
        m = HybridLM(64, d=32, n_layers=2, n_heads=2, max_T=64,
                     store="matrix", use_xl=False)
        st = m.init_state(1, "cpu")
        x = t.randint(0, 64, (1, 64))
        _, st, _ = m(x, st, None)
        self.assertIsNone(st["xl"])          # no carry cached
        m.train()
        used = []
        for _ in range(40):
            _, st, _ = m(x, st, None)
            used.append(m._reads_used)
        self.assertIn(True, used)            # reads sometimes on
        self.assertIn(False, used)           # and sometimes dropped
        m.eval()
        for _ in range(5):
            _, st, _ = m(x, st, None)
            self.assertTrue(m._reads_used)   # eval always reads

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
        pe = {"conv": UltraConveyor(shard, n_lanes=2), "agg": {}}
        out1 = holdout_probe(m, pe, 256, "cpu", warm=1, score=12)
        out2 = holdout_probe(m, pe, 256, "cpu", warm=1, score=12)
        n1 = sum(v[2] for v in out1.values()) if out1 else 0
        n2 = sum(v[2] for v in out2.values())
        self.assertGreater(n2, 0)
        self.assertGreaterEqual(n2, n1)  # cumulative, never resets
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

    def test_bootstrap_knobs_default_identical(self):
        # A39: gate_init/read_drop knobs — defaults reproduce the
        # v6.0/v6.1 machine exactly; non-default values take effect
        import torch as t
        from iga.lm_hybrid import HybridLM
        t.manual_seed(0)
        m = HybridLM(64, d=32, n_layers=2, n_heads=2, max_T=64,
                     store="matrix", use_xl=False)
        self.assertAlmostEqual(float(m.read_gate["3"]), -4.0)
        self.assertEqual(m.read_drop, 0.5)
        m2 = HybridLM(64, d=32, n_layers=2, n_heads=2, max_T=64,
                      store="matrix", use_xl=False,
                      gate_init=-1.0, read_drop=0.2)
        self.assertAlmostEqual(float(m2.read_gate["4"]), -1.0)
        m2.train()
        m2.read_drop = 0.0        # anneal endpoint: reads always on
        st = m2.init_state(1, "cpu")
        x = t.randint(0, 64, (1, 64))
        for _ in range(6):
            _, st, _ = m2(x, st, None)
            self.assertTrue(m2._reads_used)
        m2.read_drop = 1.0        # full dropout: reads never on
        for _ in range(6):
            _, st, _ = m2(x, st, None)
            self.assertFalse(m2._reads_used)

    def test_long_boost_density_end_to_end(self):
        # A43 (corrected): the cap alone was a NO-OP (steady-state
        # in-flight ~1.7 << 8 — caught by identical probes/convo in
        # the first v6.4 prep). long_boost plants several facts per
        # spawn slot; this test runs REAL prepare() on a synthetic
        # corpus and counts long-gap probes — the check that would
        # have caught the no-op before launch.
        import json
        import os
        import random
        import shutil
        import tempfile
        from iga.lm_data_ultrachat import Instruments, prepare
        self.assertEqual(Instruments(random.Random(0)).long_boost, 1)
        tmp = tempfile.mkdtemp()
        raw = os.path.join(tmp, "raw.jsonl")
        rng = random.Random(0)
        words = ("the sky turned grey over the harbor and the boats "
                 "came in early carrying nets full of silver fish "
                 "while the market stayed busy until dark").split()
        with open(raw, "w") as f:
            for _ in range(150):
                turns = [" ".join(rng.choices(words, k=40)) + " ."
                         for _ in range(6)]
                f.write(json.dumps({"data": turns}) + "\n")
        prev = os.environ.get("ULTRACHAT_JSONL")
        os.environ["ULTRACHAT_JSONL"] = raw
        try:
            counts = {}
            for tag, kw in (("std", {}),
                            ("dense", {"long_pending": 32,
                                       "long_boost": 3,
                                       "short_rate": 0.3})):
                out = os.path.join(tmp, tag)
                prepare(out, n_convos=150, seed=0, vocab=512,
                        instrument_every=1, tok_sample=50, **kw)
                evs = [json.loads(l) for l in
                       open(os.path.join(out, "events.jsonl"))]
                counts[tag] = sum(
                    1 for e in evs
                    if e.get("kind") == "probe"
                    and e.get("gap", 0) > 2000)
            self.assertGreater(counts["std"], 0)
            self.assertGreater(counts["dense"], counts["std"] * 2)
        finally:
            if prev is None:
                os.environ.pop("ULTRACHAT_JSONL", None)
            else:
                os.environ["ULTRACHAT_JSONL"] = prev
            shutil.rmtree(tmp, ignore_errors=True)

    def test_gate_norm_cap(self):
        # A42: gate-head weight norms are capped at 1.0 (the v6.2
        # late-collapse suspect grew unbounded 0->1.72); cap leaves
        # under-norm weights untouched and rescales over-norm ones
        import torch as t
        from iga.lm_hybrid import HybridLM
        from iga.lm_train import cap_gate_norms
        t.manual_seed(0)
        m = HybridLM(64, d=32, n_layers=2, n_heads=2, max_T=64,
                     store="matrix", use_xl=False, gate_mode="position")
        with t.no_grad():
            m.read_gate_pos["4"].weight.fill_(1.0)   # norm sqrt(32)
            m.read_gate_pos["3"].weight.fill_(0.01)  # tiny norm
        small = m.read_gate_pos["3"].weight.clone()
        cap_gate_norms(m)
        self.assertAlmostEqual(
            float(m.read_gate_pos["4"].weight.norm()), 1.0, places=5)
        self.assertTrue(t.equal(m.read_gate_pos["3"].weight, small))
        cap_gate_norms(HybridLM(64, d=32, n_layers=2, n_heads=2,
                                max_T=64, store="matrix",
                                use_xl=False))  # scalar mode: no-op

    def test_position_gate_init_equivalent_and_trainable(self):
        # A41 candidate: per-position read gates — at init (zero
        # weights, bias=gate_init) every position's gate equals the
        # scalar gate's value; the head is trainable (gradient
        # reaches it through a read); scalar mode stays the default
        import torch as t
        from iga.lm_hybrid import HybridLM
        t.manual_seed(0)
        m = HybridLM(64, d=32, n_layers=2, n_heads=2, max_T=64,
                     store="matrix", use_xl=False, gate_mode="position")
        lin = m.read_gate_pos["4"]
        x = t.randn(2, 7, 32)
        g = t.sigmoid(lin(x))
        self.assertTrue(t.allclose(
            g, t.full_like(g, float(t.sigmoid(t.tensor(-4.0))))))
        m.train()
        st = m.init_state(1, "cpu")
        toks = t.randint(0, 64, (1, 64))
        with t.no_grad():
            _, st, _ = m(toks, st, None)   # populate M (chunk-1 store
        st = m.detach_state(st)            # is all zeros: reads are 0)
        s = None
        for cand in range(50):
            t.manual_seed(cand)
            if float(t.rand(())) >= 0.5:
                s = cand
                break
        t.manual_seed(s)
        logits, st, _ = m(toks, st, None)
        self.assertTrue(m._reads_used)
        m.zero_grad()
        logits.mean().backward()
        self.assertIsNotNone(lin.bias.grad)
        self.assertGreater(float(lin.bias.grad.abs().sum()), 0.0)
        m_def = HybridLM(64, d=32, n_layers=2, n_heads=2, max_T=64,
                         store="matrix", use_xl=False)
        self.assertEqual(getattr(m_def, "gate_mode", "scalar"), "scalar")
        self.assertFalse(hasattr(m_def, "read_gate_pos"))

    def test_write_credit_reaches_selector_next_chunk(self):
        # A38: the store pass carries ONE write-op of graph across the
        # boundary — a read at chunk t+1 must send gradient back to
        # write_q/wk/wv/beta of the write at chunk t; recon backward at
        # chunk t must not free anything chunk t+1 needs; and graph
        # depth must not grow (three boundaries, one backward each).
        import torch as t
        from iga.lm_hybrid import HybridLM
        t.manual_seed(0)
        m = HybridLM(64, d=32, n_layers=2, n_heads=2, max_T=64,
                     store="matrix", use_xl=False)
        m.train()
        st = m.init_state(1, "cpu")
        x = t.randint(0, 64, (1, 64))
        for boundary in range(3):
            # force the read path on: read_ok draws one rand — find a
            # seed whose first draw is >= 0.5, then replay it
            s = None
            for cand in range(50):
                t.manual_seed(cand)
                if float(t.rand(())) >= 0.5:
                    s = cand
                    break
            self.assertIsNotNone(s)
            t.manual_seed(s)
            logits, st, _ = m(x, st, None)
            self.assertTrue(m._reads_used)
            loss = logits.mean() + 0.05 * m.pop_recon()
            m.zero_grad()
            loss.backward()          # legal every chunk, no retain
            if boundary > 0:         # reads touched the PREVIOUS write
                for name in ("write_q", "read_gate"):
                    g = getattr(m, name)["3"].grad
                    self.assertIsNotNone(g, name)
                self.assertGreater(
                    float(m.write_q["3"].grad.abs().sum()), 0.0)
                self.assertIsNotNone(m.mats["3"].wk.weight.grad)
                self.assertIsNotNone(m.mats["3"].beta.grad)
            st = m.detach_state(st)  # must NOT sever the credit path

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


class TestTokenKeyed(unittest.TestCase):
    def test_keyed_addressing_is_exact(self):
        # A52 (R4): the address IS the token identity — a value
        # written under token t's embedding key comes back for a
        # query at t and not for a query at a different token. This
        # is the property the learned-soup keys provably lacked
        # (TM-null x3: v8.0, R1, R2).
        torch.manual_seed(0)
        from iga.lm_hybrid import HybridLM
        m = HybridLM(64, d=32, max_T=16, store="matrix",
                     use_xl=False, gate_init=-2.0, keyed="token")
        m.eval()
        E = torch.nn.functional.normalize(
            m.embed.weight, dim=-1).detach()
        mat = m.mats["3"]
        M0 = torch.zeros(1, 32, 32)
        K = E[torch.tensor([[5]])]
        V = torch.randn(1, 1, 32)
        s = torch.ones(1, 1)
        M1, recon = mat.write_keyed(M0, K, V, s)
        v = torch.nn.functional.linear(V, mat.wv.weight)
        r_hit = torch.einsum("bij,btj->bti", M1, E[torch.tensor([[5]])])
        r_miss = torch.einsum("bij,btj->bti", M1, E[torch.tensor([[9]])])
        cos = torch.nn.functional.cosine_similarity(r_hit, v, dim=-1)
        self.assertGreater(float(cos), 0.99)     # exact-address recall
        self.assertGreater(float(r_hit.norm()),
                           3 * float(r_miss.norm()))
        self.assertLess(float(recon), 1.0)

    def test_keyed_repeated_key_cannot_overshoot(self):
        # A52 NaN law: a token occurring n>>1 times in one chunk must
        # CONVERGE toward its value, never overshoot — the unnormal-
        # ized sum applied the correction n times over (whitespace:
        # n in the hundreds), oscillated divergently, and NaN'd the
        # first r4 pod by step ~1.5k. Repeated writes must shrink
        # the residual monotonically and keep M bounded.
        torch.manual_seed(2)
        from iga.lm_hybrid import HybridLM
        m = HybridLM(64, d=32, max_T=16, store="matrix",
                     use_xl=False, keyed="token")
        E = torch.nn.functional.normalize(
            m.embed.weight, dim=-1).detach()
        mat = m.mats["3"]
        K = E[torch.full((1, 200), 5)]           # one token, 200 times
        V = torch.randn(1, 1, 32).repeat(1, 200, 1)
        s = torch.full((1, 200), 0.5)
        v = torch.nn.functional.linear(V[:, :1], mat.wv.weight)
        M = torch.zeros(1, 32, 32)
        prev = float(v.norm())
        for _ in range(6):
            M, _ = mat.write_keyed(M, K, V, s)
            res = float((v - torch.einsum(
                "bij,btj->bti", M, K[:, :1])).norm())
            self.assertLess(res, prev + 1e-4)    # never grows
            prev = res
        self.assertLess(float(M.norm()), 10 * float(v.norm()))
        self.assertTrue(M.isfinite().all())

    def test_keyed_write_strength_gates_storage(self):
        # tok_u prices storage per token TYPE: a pair written with
        # s=0 must leave no trace (glue tokens shouldn't consume
        # the store's update budget)
        torch.manual_seed(1)
        from iga.lm_hybrid import HybridLM
        m = HybridLM(64, d=32, max_T=16, store="matrix",
                     use_xl=False, keyed="token")
        E = torch.nn.functional.normalize(
            m.embed.weight, dim=-1).detach()
        mat = m.mats["3"]
        M0 = torch.zeros(1, 32, 32)
        K = E[torch.tensor([[7]])]
        V = torch.randn(1, 1, 32)
        M1, _ = mat.write_keyed(M0, K, V, torch.zeros(1, 1))
        self.assertLess(float(M1.abs().max()), 1e-7)

    def test_keyed_recon_cannot_train_selectivity(self):
        # A52b anti-gaming law: the recon loss must carry NO
        # gradient into tok_u through the recon WEIGHTS — R4's
        # tok_u minimized s-weighted fidelity by storing glue
        # ('the', '=') and suppressing identifiers. tok_u may
        # learn only from read-usefulness. The strength path into
        # the WRITE itself (s scaling the update) keeps gradient:
        # that one is priced by next-chunk read credit (A38).
        torch.manual_seed(3)
        from iga.lm_hybrid import HybridLM
        m = HybridLM(64, d=32, max_T=16, store="matrix",
                     use_xl=False, keyed="token")
        E = torch.nn.functional.normalize(
            m.embed.weight, dim=-1).detach()
        toks = torch.tensor([[5, 9, 5, 11]])
        K = E[toks]
        V = torch.randn(1, 4, 32)
        s = torch.sigmoid(m.tok_u[toks])
        M0 = torch.zeros(1, 32, 32)
        _, recon = m.mats["3"].write_keyed(M0, K, V, s)
        recon.backward()
        g = m.tok_u.grad
        self.assertIsNotNone(g)        # write-path gradient flows...
        # ...but re-run with the write path cut: recon-weight-only
        # gradient must be exactly zero
        m.tok_u.grad = None
        s2 = torch.sigmoid(m.tok_u[toks])
        M1, recon2 = m.mats["3"].write_keyed(
            M0, K, V, s2 * 0 + s2.detach())   # weights see s2 only
        recon2.backward()
        self.assertTrue(m.tok_u.grad is None or
                        float(m.tok_u.grad.abs().max()) == 0.0)

    def test_keyed_trains_end_to_end_and_ledger_exact(self):
        # 6 CPU steps through the full trainer: exact ledger, one
        # band tick per chunk, and the selectivity params exist,
        # require grad, and MOVE (credit reaches tok_u through
        # recon this chunk + read-success next chunk, A38)
        model, drive, vocab, ce0, ce1 = train(
            d=32, lanes=2, T=128, steps=6, device="cpu",
            log_every=100, arch="hybrid", store="matrix",
            use_xl=False, gate_init=-2.0, keyed="token", lr=1e-3)
        self.assertTrue(drive.audit()["telescoping_exact"])
        self.assertEqual(model._st["chunk"], 6)
        self.assertTrue(model.tok_u.requires_grad)
        self.assertTrue(model.qmix.requires_grad)
        self.assertGreater(float(model.tok_u.abs().max()), 0.0)
        self.assertTrue(ce1 == ce1 and ce1 < 20)


class TestLogitStore(unittest.TestCase):
    def test_store_to_logit_lift_end_to_end(self):
        # A53 law — THE test the campaign lacked until the decode
        # bench: a stored item must lift the answer's LOGIT through
        # the full forward, by construction. Plant (context-key ->
        # identity of token B) and forward a sequence whose tail
        # reproduces the context: p(B) must rise vs an empty store.
        torch.manual_seed(4)
        from iga.lm_hybrid import HybridLM
        m = HybridLM(64, d=32, max_T=32, store="matrix",
                     use_xl=False, keyed="logit")
        m.eval()
        with torch.no_grad():
            for k in m.bands:
                m.alpha[str(k)].fill_(1.0)
        E = torch.nn.functional.normalize(
            m.embed.weight, dim=-1).detach()
        x = torch.randint(0, 64, (1, 32))
        B_tok = 7
        ctx = E[x[0, -m.QR:]].mean(0, keepdim=True)[None]  # [1,1,32]
        with torch.no_grad():
            st0 = m.init_state(1, "cpu")
            l0, _, _ = m(x, st0, None)
            st1 = m.init_state(1, "cpu")
            for k in m.bands:
                stn = m.stores[str(k)]
                M2, _ = stn.write(st1["M"][k], stn.lift(ctx),
                                  E[B_tok][None, None],
                                  torch.ones(1, 1))
                st1["M"][k] = M2
            l1, _, _ = m(x, st1, None)
        d0 = float(l1[0, -1, B_tok] - l0[0, -1, B_tok])
        others = (l1[0, -1] - l0[0, -1]).clone()
        others[B_tok] = 0.0
        self.assertGreater(d0, 0.5)           # the lift is REAL
        self.assertGreater(d0, 2 * float(others.abs().max()))

    def test_capacity_many_pairs_still_retrievable(self):
        # A52b capacity law: with C pairs well under D, a specific
        # item must survive crosstalk (this is the arithmetic that
        # was never checked at d=256: load >> capacity in bands 4-5)
        torch.manual_seed(5)
        from iga.lm_hybrid import LogitStore
        stn = LogitStore(d=32, D=512, decay=0.0, seed=9)
        C = 64
        keys = torch.nn.functional.normalize(
            torch.randn(1, C, 32), dim=-1)
        vals = torch.nn.functional.normalize(
            torch.randn(1, C, 32), dim=-1)
        M = torch.zeros(1, 32, 512)
        with torch.no_grad():
            stn.beta.fill_(4.0)               # near-1 write rate
            for i in range(C):                # items arrive over time
                M, _ = stn.write(M, stn.lift(keys[:, i:i+1]),
                                 vals[:, i:i+1], torch.ones(1, 1))
            r = stn.read(M, stn.lift(keys[:, 3:4]))
        cos = torch.nn.functional.cosine_similarity(
            r[0, 0], vals[0, 3], dim=-1)
        self.assertGreater(float(cos), 0.7)

    def test_cosine_lr_decay_smoke(self):
        # A54 (audit C3): cosine decay runs end-to-end and CE stays
        # finite — the lr x duration guard for the 366k-step gate
        from iga.lm_train import train
        model, drive, vocab, ce0, ce1 = train(
            d=32, lanes=2, T=128, steps=6, device="cpu",
            log_every=100, arch="hybrid", store="matrix",
            use_xl=False, keyed="logit", lr=1e-3,
            lr_decay="cosine")
        self.assertTrue(drive.audit()["telescoping_exact"])
        self.assertTrue(ce1 == ce1 and ce1 < 20)

    def test_logit_mode_trains_and_ledger_exact(self):
        # e2e: 6 CPU steps, exact ledger, ticks once, finite CE,
        # alpha/tok_u trainable, mid-layer residual read OFF, and
        # the per-band store shapes are the capacity-sized ones
        from iga.lm_train import train
        model, drive, vocab, ce0, ce1 = train(
            d=32, lanes=2, T=128, steps=6, device="cpu",
            log_every=100, arch="hybrid", store="matrix",
            use_xl=False, keyed="logit", lr=1e-3)
        self.assertTrue(drive.audit()["telescoping_exact"])
        self.assertEqual(model._st["chunk"], 6)
        self.assertTrue(ce1 == ce1 and ce1 < 20)
        self.assertTrue(model.alpha["3"].requires_grad)
        self.assertTrue(model.tok_u.requires_grad)
        for k, D in ((3, 512), (4, 1024), (5, 2048)):
            self.assertEqual(tuple(model._st["M"][k].shape[1:]),
                             (32, D))


class TestEndToEnd(unittest.TestCase):
    def test_smoke_train_audit(self):
        model, drive, vocab, ce0, ce1 = train(d=48, lanes=2, T=192, steps=12,
                                              device="cpu", log_every=100)
        self.assertLess(ce1, ce0)
        audit = drive.audit()
        self.assertTrue(audit["telescoping_exact"])
        self.assertTrue(audit["scoped"])
        self.assertGreater(audit["holds"], 0)


class TestNormMix(unittest.TestCase):
    """A55 laws (R6 candidate, from the A54e F2 geometry finding).
    The defect: production keys are softmax MEANS of QR unit rows
    (norm ~ 1/sqrt(support)), but the RFF lift's gamma was
    calibrated for unit inputs — low-norm inputs land in the
    kernel's flat region and distinct contexts collide. The fix:
    unit-normalize the mix before lift. These tests pin BOTH the
    defect (raw mixes collide) and the fix (normalized separate),
    so R6 cannot launch on a lift that doesn't discriminate —
    the same trap the A53 law test set for the linear lift."""

    def _mixes(self, n, d, support=64, seed=11):
        g = torch.Generator().manual_seed(seed)
        E = torch.nn.functional.normalize(
            torch.randn(4 * support, d, generator=g), dim=-1)
        out = []
        for i in range(n):
            idx = torch.randperm(E.shape[0], generator=g)[:support]
            out.append(E[idx].mean(0))
        return torch.stack(out)                      # [n, d] low-norm

    def test_raw_mixes_collide_normalized_separate(self):
        from iga.lm_hybrid import LogitStore
        st = LogitStore(256, 512, 0.1, seed=1003)
        x = self._mixes(96, 256)
        self.assertLess(float(x.norm(dim=-1).mean()), 0.3)
        for inp, lo, hi in ((x, 0.7, None),
                            (torch.nn.functional.normalize(x, dim=-1),
                             None, 0.35)):
            K = st.lift(inp[None])[0]                # [96, D]
            C = K @ K.t()
            off = C[~torch.eye(96, dtype=torch.bool)]
            if lo is not None:
                self.assertGreater(float(off.mean()), lo)
            if hi is not None:
                self.assertLess(float(off.abs().mean()), hi)

    def test_normalized_keys_retrieve_raw_do_not(self):
        from iga.lm_hybrid import LogitStore
        st = LogitStore(256, 512, 0.1, seed=1004)
        g = torch.Generator().manual_seed(12)
        n = 48
        V = torch.nn.functional.normalize(
            torch.randn(n, 256, generator=g), dim=-1)
        x = self._mixes(n, 256, seed=13)
        hits = {}
        for name, inp in (("raw", x),
                          ("norm", torch.nn.functional.normalize(
                              x, dim=-1))):
            K = st.lift(inp[None])                   # [1, n, D]
            M, _ = st.write(torch.zeros(1, 256, 512),
                            K, V[None], torch.ones(1, n))
            r = torch.einsum("bij,btj->bti", M, K)[0]
            hits[name] = float((torch.argmax(r @ V.t(), dim=-1) ==
                                torch.arange(n)).float().mean())
        self.assertLess(hits["raw"], 0.25)
        self.assertGreater(hits["norm"], 0.6)

    def test_norm_mix_flag_end_to_end(self):
        from iga.lm_hybrid import HybridLM
        torch.manual_seed(7)
        m0 = HybridLM(64, d=32, max_T=32, store="matrix",
                      use_xl=False, keyed="logit")
        m1 = HybridLM(64, d=32, max_T=32, store="matrix",
                      use_xl=False, keyed="logit", norm_mix=True)
        # no parameters added: checkpoints portable both directions
        self.assertEqual(set(m0.state_dict().keys()),
                         set(m1.state_dict().keys()))
        m1.load_state_dict(m0.state_dict())
        with torch.no_grad():
            for k in m0.bands:
                m0.alpha[str(k)].fill_(1.0)
                m1.alpha[str(k)].fill_(1.0)
        x = torch.randint(0, 64, (1, 32))
        outs = []
        for m in (m0, m1):
            st = m.init_state(1, "cpu")
            l1, st, _ = m(x, st, None)
            l2, st, _ = m(x, st, None)
            outs.append(l2)
        # the flag is live: same weights, different key geometry
        diff = (outs[0] - outs[1]).abs().max().detach()
        self.assertGreater(float(diff), 1e-6)
        # trainable: backward through the normalized path is finite
        loss = outs[1].square().mean()
        loss.backward()
        gm = m1.stores["3"].beta.grad
        self.assertIsNotNone(gm)
        self.assertTrue(bool(torch.isfinite(gm)))


class TestAuxTrunk(unittest.TestCase):
    """A58 laws (R8): the pay-the-trunk auxiliary loss. A57c
    exonerated geometry, volume, and dose — the bleed is gradient
    starvation. These pin the plumbing: the pre-bonus logits are
    kept exactly when (training AND bonus applied AND aux on), the
    aux term is live in the gradient, and aux off is bit-parity."""

    def _model(self, aux):
        from iga.lm_hybrid import HybridLM
        torch.manual_seed(9)
        m = HybridLM(64, d=32, max_T=32, store="matrix",
                     use_xl=False, keyed="logit", norm_mix=True,
                     aux_trunk=aux)
        with torch.no_grad():
            for k in m.bands:
                m.alpha[str(k)].fill_(1.0)
        return m

    def _two_chunks(self, m):
        g = torch.Generator().manual_seed(123)
        x = torch.randint(0, 64, (1, 32), generator=g)
        st = m.init_state(1, "cpu")
        l1, st, _ = m(x, st, None)
        l2, st, _ = m(x, st, None)
        return l2

    def test_aux_hidden_kept_only_when_armed(self):
        m = self._model(0.2)
        m.train()
        m.read_drop = 0.0                 # bonus always applied
        self._two_chunks(m)
        self.assertIsNotNone(m._aux_hidden)
        m.eval()                          # never at eval
        self._two_chunks(m)
        self.assertIsNone(m._aux_hidden)
        m0 = self._model(0.0)             # never when aux off
        m0.train()
        m0.read_drop = 0.0
        self._two_chunks(m0)
        self.assertIsNone(m0._aux_hidden)
        self.assertFalse(hasattr(m0, "aux_head"))

    def test_aux_pays_blocks_not_production_head(self):
        # R8b law — the whole point: trunk BLOCKS get the aux
        # gradient, the PRODUCTION head gets none of it
        y = torch.randint(0, 64, (1, 32))
        m = self._model(1.0)
        m.train()
        m.read_drop = 0.0
        l2 = self._two_chunks(m)
        aux_lg = m.aux_head(m.lnf(m._aux_hidden))
        aux_loss = torch.nn.functional.cross_entropy(
            aux_lg.reshape(-1, 64), y.reshape(-1))
        m.zero_grad()
        aux_loss.backward(retain_graph=True)
        self.assertIsNone(m.head.weight.grad)     # head untouched
        blk = next(m.blocks[0].parameters())
        self.assertIsNotNone(blk.grad)            # blocks paid
        self.assertGreater(float(blk.grad.abs().max()), 0)
        self.assertTrue(bool(torch.isfinite(aux_loss.detach())))

    def test_aux_off_is_parity(self):
        # same shared weights -> identical production forward (the
        # aux head exists but is structurally outside the path);
        # load m0's weights into m1 because creating aux_head
        # consumes RNG draws and shifts every later init
        m0 = self._model(0.0)
        m1 = self._model(0.3)
        m1.load_state_dict(m0.state_dict(), strict=False)
        outs = []
        for m in (m0, m1):
            m.eval()
            outs.append(self._two_chunks(m).detach())
        self.assertLess(float((outs[0] - outs[1]).abs().max()), 1e-6)


class TestButtonLaws(unittest.TestCase):
    """A64 Phase 2 — graded-press primary reinforcer laws.
    B1 press-never-pays, B2 magnitude->w, B3 withhold-then-veto,
    B4 parity-off, B5 prophet-spectator."""

    def _drive_with_open_hold(self):
        d = Drive(n_lanes=1)
        for k in range(1, N_BANDS):
            d.ema[f"fid:{k}"] = 0.5      # healthy: no maintains/vetoes
        d.probe(0, torch.tensor(0.4), gap=100)   # recall:b0
        d.button(0, 1)                    # +press mints (labels select)
        d.sweep(losses=[])
        self.assertTrue(any(h["key"] == "recall:b0" for h in d.holds[0]))
        return d

    def test_B1_B2_press_weights_and_pay_is_w_scaled(self):
        d = self._drive_with_open_hold()
        h = [h for h in d.holds[0] if h["key"] == "recall:b0"][0]
        self.assertEqual(h["w"], 1.0)
        d.button(0, 2)                   # magnitude -> open hold weight
        self.assertEqual(h["w"], 2.0)
        d.step_t += 1
        d.probe(0, torch.tensor(0.9, requires_grad=True), gap=100)
        losses = []
        d.step_t = h["due"] + 1
        d.sweep(losses=losses)
        e = [e for e in d.ledger if e["key"] == "recall:b0"][0]
        self.assertEqual(e["w"], 2.0)
        self.assertAlmostEqual(e["pay"], 2.0 * (e["phi0"] - e["phi1"]),
                               places=12)
        self.assertEqual(len(losses), 1)   # settle pays through loss...
        a = d.audit()
        self.assertTrue(a["telescoping_exact"])
        self.assertTrue(a["voided_zero"])
        self.assertEqual(a["presses"], 2)

    def test_B1_negative_voids_settle_zero_no_loss(self):
        d = self._drive_with_open_hold()
        d.step_t += 1
        d.probe(0, torch.tensor(0.9, requires_grad=True), gap=100)
        d.button(0, -1)                  # withhold: void the open hold
        h = [h for h in d.holds[0] if h["key"] == "recall:b0"][0]
        self.assertEqual(h["w"], 0.0)
        losses = []
        d.step_t = h["due"] + 1
        d.sweep(losses=losses)
        self.assertEqual(losses, [])     # ...a voided one never does
        e = [e for e in d.ledger if e["key"] == "recall:b0"][0]
        self.assertEqual(e["pay"], 0.0)
        self.assertTrue(d.audit()["voided_zero"])

    def test_B3_repeated_negatives_veto_reset_on_fire(self):
        d = self._drive_with_open_hold()
        d.button(0, -1)
        d.button(0, -1)
        self.assertNotIn("recall:b0", d.veto_until)  # R2: 2 is not
        d.button(0, -1)                              # repeated enough
        self.assertIn("recall:b0", d.veto_until)
        self.assertEqual(d.neg_count["recall:b0"], 0)  # reset on fire
        self.assertLessEqual(d.veto_until["recall:b0"] - d.step_t,
                             2048)                   # B3'': capped
        v0 = d.vetoes
        d.step_t += 1
        d.sweep(losses=[])
        self.assertGreater(d.vetoes, v0)   # disapproval veto ledgered
        d.step_t = max(d.veto_until["recall:b0"],
                       max(h["due"] for h in d.holds[0])) + 1
        d.sweep(losses=[])                 # old holds expire in-sweep
        d.step_t += 1
        d.sweep(losses=[])                 # veto lapsed: proposes again
        self.assertTrue(any(h["key"] == "recall:b0" for h in d.holds[0]))
        d.probe(0, torch.tensor(0.4), gap=100)
        d.button(0, -1)                    # one negative post-lapse
        self.assertLessEqual(d.veto_until.get("recall:b0", 0),
                             d.step_t)     # does NOT re-veto
        d2 = self._drive_with_open_hold()
        d2.button(0, -1)
        d2.button(0, -1)
        d2.button(0, 1)                    # positive resets the count
        d2.button(0, -1)
        d2.button(0, -1)
        self.assertNotIn("recall:b0", d2.veto_until)

    def test_B4_certified_stream_emits_no_press_events(self):
        lane = Lane(Vocab(), random.Random(11))
        _, _, evs = lane.take(30000)
        self.assertFalse(any(k == "button" for _, k, _ in evs))
        self.assertTrue(any(k == "earned" for _, k, _ in evs))

    def test_parenting_mode_classes_presses_no_earned(self):
        log = []
        cfg = {"pos": 0.4, "neg": 0.3, "pos_v": 2, "neg_v": 1,
               "log": log}
        v = Vocab()
        lane = Lane(v, random.Random(7), buttons=cfg)
        toks, _, evs = lane.take(40000)
        presses = [(p, d["v"]) for p, k, d in evs if k == "button"]
        self.assertGreater(len(presses), 0)
        self.assertFalse(any(k == "earned" for _, k, _ in evs))
        for p, val in presses:           # the press is IN the stream
            self.assertEqual(v.words[toks[p]],
                             f"<{'+' if val > 0 else '-'}{abs(val)}>")
        self.assertEqual({i["cls"] for i in log}, {"pos", "none", "neg"})

    def test_vocab_surgery_output_parity(self):
        # A65: extending a trained ckpt with press tokens must leave
        # every old-token logit bit-near-exact (store paths included)
        # and the new rows dead (bias -20) until trained
        import torch as t
        from iga.lm_hybrid import HybridLM
        from iga.lm_vocab import extend_model_state
        t.manual_seed(3)
        m0 = HybridLM(64, d=32, n_layers=2, n_heads=2, max_T=64,
                      store="matrix", keyed="logit", norm_mix=True,
                      aux_trunk=0.2, use_xl=False)
        with t.no_grad():
            for a in m0.alpha.values():
                a.fill_(2.0)          # store bonus live during parity
        sd, v0, v1 = extend_model_state(m0.state_dict())
        self.assertEqual((v0, v1), (64, 68))
        m1 = HybridLM(68, d=32, n_layers=2, n_heads=2, max_T=64,
                      store="matrix", keyed="logit", norm_mix=True,
                      aux_trunk=0.2, use_xl=False)
        m1.load_state_dict(sd)
        m0.eval()
        m1.eval()
        x = t.randint(0, 64, (1, 64))
        st0, st1 = m0.init_state(1, "cpu"), m1.init_state(1, "cpu")
        with t.no_grad():
            for _ in range(2):        # chunk 2 reads chunk 1's writes
                l0, st0, _ = m0(x, st0, None)
                m0.pop_write_cost()
                m0.pop_recon()
                l1, st1, _ = m1(x, st1, None)
                m1.pop_write_cost()
                m1.pop_recon()
                st0 = m0.detach_state(st0)
                st1 = m1.detach_state(st1)
        self.assertLess(float((l0 - l1[..., :64]).abs().max()), 1e-5)
        self.assertLess(float(l1[..., 64:].max()), -10.0)

    def test_B5_prophet_is_a_spectator(self):
        import torch as t
        from iga.lm_press import PressProphet
        kw = dict(d=32, lanes=2, T=64, steps=8, device="cpu",
                  arch="hybrid", store="matrix", keyed="logit",
                  norm_mix=True, aux_trunk=0.2, log_every=100)
        cfg = {"pos": 0.5, "neg": 0.2, "pos_v": 2, "neg_v": 1}
        m1, d1, *_ = train(**kw, buttons=dict(cfg))
        prophet = PressProphet(d=32)     # RNG draw erased by train()'s
        m2, d2, *_ = train(**kw, buttons=dict(cfg), prophet=prophet)
        s1, s2 = m1.state_dict(), m2.state_dict()
        for k in s1:
            self.assertTrue(t.equal(s1[k], s2[k]), f"B5 broken at {k}")
        self.assertEqual(len(d1.presses), len(d2.presses))


class TestSleepLaws(unittest.TestCase):
    """A62 Phase 1 — the consolidation laws, pinned before any run.
    L1 only-paid-replays, L2 parity-off, L3 slow-weights-only,
    L4 no drive pay."""

    def _model(self, seed=0):
        import torch as t
        from iga.lm_hybrid import HybridLM
        t.manual_seed(seed)
        m = HybridLM(64, d=32, n_layers=2, n_heads=2, max_T=64,
                     store="matrix", keyed="logit", norm_mix=True,
                     aux_trunk=0.2, use_xl=False)
        with t.no_grad():
            for a in m.alpha.values():
                a.fill_(2.0)   # store bonus live (init 0 is silent)
        return m

    def _rig(self, arm, paid=True):
        import torch as t
        from iga.lm_drive import Drive
        from iga.lm_sleep import Sleeper
        m = self._model()
        opt = t.optim.AdamW(m.parameters(), lr=1e-3)
        drive = Drive(n_lanes=2)
        sl = Sleeper(arm=arm, every=4, block_chunks=2, seed=0)
        sl.bind(drive)
        t.manual_seed(100)
        for _ in range(8):
            sl.observe(t.randint(0, 64, (2, 64)))
        if paid:
            drive.ledger.append(
                {"lane": 0, "band": 3, "key": "recall:b0",
                 "phi0": 0.5, "w": 1.0, "t0": 64, "phi1": 0.1,
                 "pay": 0.4, "t1": 448})
        drive.ledger.append(
            {"lane": 1, "band": 3, "key": "recall:b0", "phi0": 0.5,
             "w": 1.0, "t0": 0, "phi1": 0.5, "pay": 0.0, "t1": 512})
        drive.step_t = 512
        return m, opt, drive, sl

    def test_L1_only_paid_replays(self):
        m, opt, drive, sl = self._rig("A")
        row = sl.maybe_sleep(m, opt, drive, step=4)
        self.assertIsNotNone(row)
        a = sl.audit()
        self.assertGreater(a["replayed"], 0)
        self.assertTrue(a["only_paid"])
        for r in sl.replayed:      # never the zero-pay lane-1 entry
            self.assertEqual(r["lane"], 0)
            self.assertEqual(r["ledger_i"], 0)

    def test_L1_no_paid_entries_no_sleep(self):
        m, opt, drive, sl = self._rig("A", paid=False)
        self.assertIsNone(sl.maybe_sleep(m, opt, drive, step=4))
        self.assertEqual(sl.audit()["replayed"], 0)

    def test_L3_slow_weights_only(self):
        import torch as t
        from iga.lm_sleep import frozen_param_names
        for arm in ("A", "B"):
            m, opt, drive, sl = self._rig(arm)
            frozen = set(frozen_param_names(m))
            self.assertIn("tok_u", frozen)
            self.assertIn("aux_head.weight", frozen)
            pre = {n: p.clone() for n, p in m.named_parameters()}
            row = sl.maybe_sleep(m, opt, drive, step=4)
            self.assertIsNotNone(row)
            self.assertTrue(bool(t.isfinite(t.tensor(row["loss"]))))
            moved = []
            for n, p in m.named_parameters():
                if n in frozen:
                    self.assertTrue(t.equal(p, pre[n]),
                                    f"{arm}: frozen {n} moved")
                    self.assertIsNone(p.grad,
                                      f"{arm}: frozen {n} got grad")
                    self.assertTrue(p.requires_grad,
                                    f"{arm}: {n} not restored")
                elif not t.equal(p, pre[n]):
                    moved.append(n)
            self.assertTrue(moved, f"{arm}: no slow weight moved")
            self.assertFalse(m.store_read_off)   # wake reads restored
            self.assertTrue(m.training)

    def test_L4_drive_untouched(self):
        m, opt, drive, sl = self._rig("B")
        before = (len(drive.ledger), drive.step_t, dict(drive.ema),
                  sorted(drive.minted),
                  [len(h) for h in drive.holds])
        self.assertIsNotNone(sl.maybe_sleep(m, opt, drive, step=4))
        after = (len(drive.ledger), drive.step_t, dict(drive.ema),
                 sorted(drive.minted),
                 [len(h) for h in drive.holds])
        self.assertEqual(before, after)

    def test_L2_parity_off_bit_exact(self):
        import torch as t
        from iga.lm_sleep import Sleeper
        kw = dict(d=32, lanes=2, T=64, steps=6, device="cpu",
                  arch="hybrid", store="matrix", keyed="logit",
                  norm_mix=True, aux_trunk=0.2, log_every=100)
        m1, d1, *_ = train(**kw)
        m2, d2, *_ = train(**kw, sleep=Sleeper(every=0))
        s1, s2 = m1.state_dict(), m2.state_dict()
        self.assertEqual(sorted(s1), sorted(s2))
        for k in s1:
            self.assertTrue(t.equal(s1[k], s2[k]),
                            f"parity broken at {k}")
        self.assertEqual(d1.ledger, d2.ledger)

    def test_arm_b_distills_the_store_bonus(self):
        # chunk 1 writes into a fresh store, so teacher == student
        # (KL exactly 0); chunk 2's teacher reads chunk 1's writes —
        # the KL that funds what LM loss defunds must be nonzero
        m, opt, drive, sl = self._rig("B")
        self.assertIsNotNone(sl.maybe_sleep(m, opt, drive, step=4))
        self.assertGreater(sl.stats[0]["chunks"], 1)
        self.assertGreater(sl.stats[0]["loss"], 0.0)

    def test_store_read_off_leaves_bands_alive(self):
        # the A62 switch severs ONLY the store bonus: with M still
        # empty the toggle changes nothing; once M holds a chunk the
        # toggled logits must differ while mem tokens stay live
        # (lesioned would zero those too)
        import torch as t
        from iga.lm_sleep import state_copy
        m = self._model()
        m.eval()
        t.manual_seed(7)
        x = t.randint(0, 64, (1, 64))
        st = m.init_state(1, "cpu")
        with t.no_grad():
            base, st1, _ = m(x, state_copy(st), None)
            m.store_read_off = True
            off, _, _ = m(x, state_copy(st), None)
            m.store_read_off = False
            self.assertLess(float((base - off).abs().max()), 1e-6)
            st1 = m.detach_state(st1)         # M now written
            on2, _, _ = m(x, state_copy(st1), None)
            m.store_read_off = True
            off2, _, _ = m(x, state_copy(st1), None)
            m.store_read_off = False
        self.assertGreater(float((on2 - off2).abs().max()), 1e-6)

    def test_in_vivo_wake_sleep_lawful(self):
        # the live loop: tap buffers wake tokens, blocks fire on the
        # real ledger, every replay stays inside a paid span
        from iga.lm_sleep import Sleeper
        sl = Sleeper(arm="B", every=8, block_chunks=2, seed=0)
        train(d=32, lanes=2, T=64, steps=24, device="cpu",
              arch="hybrid", store="matrix", keyed="logit",
              norm_mix=True, aux_trunk=0.2, log_every=100,
              constants={"horizons": {1: 64, 2: 64, 3: 128,
                                      4: 256, 5: 512}},
              sleep=sl)
        a = sl.audit()
        self.assertTrue(a["only_paid"])
        for row in sl.stats:
            self.assertGreater(row["chunks"], 0)


class TestServeLaws(unittest.TestCase):
    """A65 — serving-harness laws: commit parity (faithful regime),
    press spans + void, L1 at serve, wipe alignment, store frozen
    through serve-sleep."""

    class _Tok:
        ids = {"<pad>": 0, "<eot_human>": 1, "<eot_model>": 2,
               "<+1>": 60, "<+2>": 61, "<-1>": 62, "<-2>": 63}

        def token_to_id(self, s):
            return self.ids[s]

    def _session(self, sleeper=None, T=64):
        import torch as t
        from iga.lm_hybrid import HybridLM
        from iga.lm_serve import ServeSession
        t.manual_seed(0)
        m = HybridLM(64, d=32, n_layers=2, n_heads=2, max_T=T,
                     store="matrix", keyed="logit", norm_mix=True,
                     aux_trunk=0.2, use_xl=False)
        with t.no_grad():
            for a in m.alpha.values():
                a.fill_(2.0)
        return ServeSession(m, self._Tok(), T=T, device="cpu",
                            sleeper=sleeper, seed=0), m

    def test_commit_parity_with_training_chunk_path(self):
        # the session's committed state must bit-match the training
        # chunk path — generation on copies never perturbs it
        import torch as t
        from iga.lm_sleep import state_copy
        s, m = self._session()
        t.manual_seed(5)
        ids = t.randint(3, 60, (300,)).tolist()
        s._append(ids)                    # 4 commits + 44 pending
        st = m.init_state(1, "cpu")
        with t.no_grad():
            for off in range(0, 256, 64):
                _, st, _ = m(t.tensor([ids[off:off + 64]]), st, None)
                m.pop_write_cost()
                m.pop_recon()
                st = m.detach_state(st)
        for k in m.bands:
            self.assertTrue(t.equal(s.st["h"][k], st["h"][k]))
            self.assertTrue(t.equal(s.st["M"][k], st["M"][k]))
        with t.no_grad():
            lg, _, _ = m(t.tensor([ids[256:]]), state_copy(st), None)
            m.pop_write_cost()
            m.pop_recon()
        self.assertTrue(t.equal(s._next_logits(), lg[0, -1].float()))

    def test_press_token_and_ledger_position(self):
        s, m = self._session()
        s._append(list(range(3, 33)))
        p0 = s.pos
        s.press(2)
        self.assertEqual(s.pending[-1], 61)   # the act is perceivable
        self.assertEqual(s.drive.presses[0]["t"], p0)
        self.assertEqual(s.pos, p0 + 1)

    def test_press_spans_void_and_L1_at_serve(self):
        import torch as t
        from iga.lm_sleep import Sleeper, frozen_param_names
        sl = Sleeper(arm="B", every=0, block_chunks=2, seed=0)
        s, m = self._session(sleeper=sl)
        t.manual_seed(6)
        s._append(t.randint(3, 60, (500,)).tolist())
        s.drive.step_t = 300
        s.drive.button(0, 2)              # span [172, 300] at w=128
        s.drive.step_t = 400
        s.drive.button(0, 1)              # span [272, 400]
        s.drive.step_t = 450
        s.drive.button(0, -1)             # voids only the overlap
        s._flush()
        n = sl.harvest_presses(s.drive, span_w=128)
        self.assertEqual(n, 1)
        self.assertEqual((sl.spans[0]["t0"], sl.spans[0]["t1"]),
                         (172, 300))
        frozen = set(frozen_param_names(m))
        pre = {k: p.clone() for k, p in m.named_parameters()
               if k in frozen}
        opt = t.optim.AdamW(m.parameters(), lr=1e-3)
        self.assertIsNotNone(sl._block(m, opt, step=500))
        self.assertTrue(sl.audit()["only_paid"])
        for r in sl.replayed:             # L1 at serve: inside the
            self.assertGreaterEqual(r["lo"], 172)   # press span
            self.assertLessEqual(r["hi"], 300)
        for k, p in m.named_parameters():
            if k in frozen:
                self.assertTrue(t.equal(p, pre[k]), k)

    def test_wipe_flush_keeps_clock_and_buffer_aligned(self):
        from iga.lm_sleep import Sleeper
        sl = Sleeper(arm="B", every=0, block_chunks=2, seed=0)
        s, m = self._session(sleeper=sl)
        s._append(list(range(3, 33)))
        s.wipe()                          # short-commit flush (A65)
        self.assertEqual(s.pending, [])
        self.assertEqual(s.pos, 30)
        self.assertEqual(s.n_committed, 30)
        self.assertEqual(sl.end, s.n_committed)

    def test_sleep_now_end_to_end(self):
        import torch as t
        from iga.lm_sleep import Sleeper
        sl = Sleeper(arm="B", every=0, block_chunks=2, seed=0)
        s, m = self._session(sleeper=sl)
        t.manual_seed(8)
        s._append(t.randint(3, 60, (400,)).tolist())
        s.press(2)
        out = s.sleep_now(blocks=2, span_w=200)
        self.assertGreaterEqual(out["blocks"], 1)
        self.assertTrue(out["audit"]["only_paid"])

    def test_replay_twice_distills_where_single_pass_cannot(self):
        # A66-R2: a sub-chunk span replayed once reads an empty
        # store (KL = float noise ~1e-9, under the floor); replayed
        # twice, the teacher reads its own pass-one writes and the
        # KL carries real signal (>=1e-4 measured), so the step
        # lands. Floor 1e-6 sits between noise and signal.
        import torch as t
        from iga.lm_sleep import Sleeper
        for twice, expect_steps in ((False, 0), (True, 1)):
            sl = Sleeper(arm="B", every=0, block_chunks=2, seed=0,
                         min_step_loss=1e-6, replay_twice=twice)
            s, m = self._session(sleeper=sl)
            t.manual_seed(11)
            s._append(t.randint(3, 60, (100,)).tolist())
            s.press(2)
            out = s.sleep_now(blocks=1, span_w=40)   # span < one chunk
            self.assertEqual(out["blocks"], 1)
            self.assertTrue(out["audit"]["only_paid"])
            if expect_steps:
                self.assertGreaterEqual(
                    out["audit"]["steps_taken"], 1, "twice: no step")
            else:
                self.assertEqual(out["audit"]["steps_taken"], 0,
                                 "single pass stepped on empty store")

    def test_min_step_loss_gates_noise_updates(self):
        # A65: no disagreement, no update — with the floor above any
        # possible loss, blocks replay and record but the model must
        # not move at all
        import torch as t
        from iga.lm_sleep import Sleeper
        sl = Sleeper(arm="B", every=0, block_chunks=2, seed=0,
                     min_step_loss=1e9)
        s, m = self._session(sleeper=sl)
        t.manual_seed(9)
        s._append(t.randint(3, 60, (400,)).tolist())
        s.press(2)
        pre = {k: p.clone() for k, p in m.named_parameters()}
        out = s.sleep_now(blocks=3, span_w=200)
        self.assertGreaterEqual(out["blocks"], 1)   # replayed, recorded
        for k, p in m.named_parameters():
            self.assertTrue(t.equal(p, pre[k]), k)  # zero movement


if __name__ == "__main__":
    unittest.main()
