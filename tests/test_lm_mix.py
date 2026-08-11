"""A46 law tests — the v8 mixed corpus pipeline.

Pins: the identifier miner emits probes whose answer token really is
the token at the probe position (the property the whole natural-eval
battery rests on); the mixer honors budget/determinism; digest pairs
format as human=code/model=summary; and a T=2048 hybrid trains on a
prepare_mix shard end-to-end (the shakedown's local twin).
"""

import json
import os
import random
import shutil
import tempfile
import unittest

import numpy as np


SYNTH_CODE = '''
def compute_total(items):
    total = 0
    for it in items:
        total += it.price
    return total


def render_receipt(purchases):
    header = "RECEIPT"
    lines = [header]
''' + "\n".join(f"    lines.append('row {i}')" for i in range(220)) + '''
    amount = compute_total(purchases)
    lines.append(str(amount))
    return "\\n".join(lines)
'''


def tiny_tokenizer(tmp):
    from iga.lm_data_ultrachat import train_tokenizer
    texts = [SYNTH_CODE, "the market was busy again .",
             "thanks . good job ."] * 50
    return train_tokenizer(iter(texts),
                           os.path.join(tmp, "tok.json"), 512)


class TestMiner(unittest.TestCase):
    def test_probe_answer_is_token_at_pos(self):
        from iga.lm_data_mix import mine_identifier_probes
        tmp = tempfile.mkdtemp()
        try:
            tok = tiny_tokenizer(tmp)
            evs = mine_identifier_probes(SYNTH_CODE, tok, base_pos=0,
                                         min_gap_chars=500,
                                         rng=random.Random(0))
            self.assertGreater(len(evs), 0)
            ids = tok.encode(SYNTH_CODE).ids
            for e in evs:
                self.assertEqual(ids[e["pos"]], e["answer"])
                self.assertGreater(e["gap"], 0)
                self.assertTrue(e["nat"])
                self.assertGreaterEqual(len(e["distractors"]), 2)
                self.assertNotIn(e["answer"], e["distractors"])
            names = {tok.decode([e["answer"]]).strip() for e in evs}
            self.assertTrue(any("compute" in n or "total" in n
                                or "render" in n or "amount" in n
                                or "header" in n for n in names))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestPrepareMix(unittest.TestCase):
    def _sources(self, n_code=40, n_wiki=30, n_digest=20):
        code_docs = [SYNTH_CODE.replace("compute_total",
                                        f"compute_total_{i}")
                     for i in range(n_code)]
        wiki_docs = [("history section %d .\n\n" % i) * 40
                     for i in range(n_wiki)]
        pairs = [("digest",
                  SYNTH_CODE.replace("render_receipt", f"render_{i}"),
                  "renders a receipt for the given purchases .")
                 for i in range(n_digest)]
        return [("code", 0.55, lambda: iter(code_docs)),
                ("wiki", 0.25, lambda: iter(wiki_docs)),
                ("digest", 0.20, lambda: iter(pairs))]

    def test_budget_determinism_and_digest_format(self):
        from iga.lm_data_mix import prepare_mix
        from iga.lm_data_ultrachat import load_tokenizer
        tmp = tempfile.mkdtemp()
        try:
            stats = {}
            for tag in ("a", "b"):
                out = os.path.join(tmp, tag)
                stats[tag] = prepare_mix(
                    out, self._sources(), budget_tokens=60_000,
                    seed=7, vocab=512, tok_sample_docs=30)
            self.assertEqual(stats["a"], stats["b"])   # deterministic
            toks_a = np.fromfile(os.path.join(tmp, "a", "tokens.bin"),
                                 dtype=np.uint16)
            toks_b = np.fromfile(os.path.join(tmp, "b", "tokens.bin"),
                                 dtype=np.uint16)
            self.assertTrue((toks_a == toks_b).all())
            self.assertGreaterEqual(len(toks_a), 60_000)
            self.assertGreater(stats["a"]["code"], 0)
            self.assertGreater(stats["a"]["digest"], 0)
            tok = load_tokenizer(os.path.join(tmp, "a",
                                              "tokenizer.json"))
            eot_h = tok.token_to_id("<eot_human>")
            eot_m = tok.token_to_id("<eot_m" "odel>")
            self.assertIn(eot_h, toks_a.tolist())
            self.assertIn(eot_m, toks_a.tolist())
            evs = [json.loads(l) for l in
                   open(os.path.join(tmp, "a", "events.jsonl"))]
            self.assertTrue(any(e["kind"] == "probe" for e in evs))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_mined_eval_shard_and_t2048_training(self):
        from iga.lm_data_mix import prepare_mix
        from iga.lm_data_ultrachat import UltraConveyor
        from iga.lm_train import train
        tmp = tempfile.mkdtemp()
        try:
            out = os.path.join(tmp, "ev")
            prepare_mix(out, self._sources(), budget_tokens=80_000,
                        seed=3, vocab=512, tok_sample_docs=30,
                        mine_ids=True)
            evs = [json.loads(l) for l in
                   open(os.path.join(out, "events.jsonl"))]
            nat = [e for e in evs if e.get("nat")]
            self.assertGreater(len(nat), 0)
            toks = np.fromfile(os.path.join(out, "tokens.bin"),
                               dtype=np.uint16)
            for e in nat[:10]:      # miner alignment survives the mix
                self.assertEqual(int(toks[e["pos"]]), e["answer"])
            conv = UltraConveyor(out, n_lanes=2)
            x, y, events = conv.chunk(2048)      # T=2048 serving
            self.assertEqual(x.shape, (2, 2048))
            model, drive, vocab, ce0, ce1 = train(
                d=32, lanes=2, T=2048, steps=3, device="cpu",
                log_every=10, arch="hybrid", store="matrix",
                use_xl=False, data=out, lr=1e-4)
            self.assertTrue(drive.audit()["telescoping_exact"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestBf16(unittest.TestCase):
    def test_bf16_train_ledger_exact_and_states_fp32(self):
        # A49: bf16 autocast — the exact ledger must stay exact, CE
        # finite, and chunk-boundary states re-anchored to fp32 so
        # precision-sensitive accumulators never compound in bf16
        from iga.lm_train import train
        import torch as t
        model, drive, vocab, ce0, ce1 = train(
            d=32, lanes=2, T=128, steps=6, device="cpu",
            log_every=100, arch="hybrid", store="matrix",
            use_xl=False, bf16=True)
        self.assertTrue(drive.audit()["telescoping_exact"])
        self.assertTrue(ce1 == ce1 and ce1 < 20)     # finite
        st = model._st
        for k in st["h"]:
            self.assertEqual(st["h"][k].dtype, t.float32)
        for k in st["M"]:
            self.assertEqual(st["M"][k].dtype, t.float32)


class TestJsonlSource(unittest.TestCase):
    def test_jsonl_code_source_roundtrip(self):
        import json as _json
        from iga.lm_data_mix import iter_jsonl_texts
        tmp = tempfile.mkdtemp()
        try:
            p = os.path.join(tmp, "code.jsonl")
            with open(p, "w") as f:
                f.write(_json.dumps({"text": SYNTH_CODE}) + "\n")
                f.write("not json\n")
                f.write(_json.dumps({"text": "x"}) + "\n")   # too short
                f.write(_json.dumps({"text": SYNTH_CODE * 2}) + "\n")
            docs = list(iter_jsonl_texts(p))
            self.assertEqual(len(docs), 2)
            self.assertTrue(docs[0].startswith("\ndef compute_total"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
