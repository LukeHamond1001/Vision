"""Laws for the v10.1 EPISODIC CAST (2026-08-21). The v10 cast — 24
persistent facts per life, asked tens of thousands of times — was
memorized; the binder never armed. Episodic mode plants NOVEL facts
from large vocabularies disjoint from the roster, asks each a few
times at the stage's gaps, then retires it; the roster becomes
biography at 16x the longest gap. These tests pin the contract; the
bit-exact parity of episodic=None against the pre-change builder was
verified by md5 on three builds (ledgered)."""
import json
import os
import shutil
import pytest

SRC = "data/ultrachat_raw.jsonl"
pytestmark = pytest.mark.skipif(not os.path.exists(SRC),
                                reason="local UltraChat sample absent")


def _sources():
    from iga.lm_data_life import ultrachat_source, simple_only
    return {"default": ultrachat_source(SRC),
            "tokenizer": ultrachat_source(SRC),
            "infancy": simple_only(ultrachat_source(SRC)),
            "childhood": ultrachat_source(SRC, 40),
            "adolescence": ultrachat_source(SRC, 400),
            "tail": ultrachat_source(SRC, 900)}


def _build(tmp, episodic, budget=1_400_000, lives=2):
    from iga.lm_data_life import prepare_life
    out = os.path.join(tmp, "epi" if episodic else "roster")
    shutil.rmtree(out, ignore_errors=True)
    prepare_life(out, budget, lives, seed=7, world_seed=99, vocab=4096,
                 sources=_sources(), tok_sample=300, episodic=episodic)
    m = json.load(open(os.path.join(out, "manifest.json")))
    ev = [json.loads(l) for l in open(os.path.join(out, "events.jsonl"))]
    return m, ev


def test_roster_mode_unchanged_shape(tmp_path):
    m, ev = _build(str(tmp_path), None)
    assert m["cast_mode"] == "roster_v10" and m["episodic"] is None
    for life in m["lives"]:
        assert "episodic" not in life
        assert len(life["cast"]) == 24
        assert all(not str(f["cls"]).startswith("epi") for f in life["cast"])
    assert m["stats"]["plants"] == 24 * m["n_lives"]


def test_episodic_contract(tmp_path):
    from iga.lm_data_life import (EPISODIC_NAMES, EPISODIC_OBJECTS,
                                  NAMES, OBJECTS)
    assert not set(EPISODIC_NAMES) & set(NAMES)
    assert not set(EPISODIC_OBJECTS) & set(OBJECTS)
    assert len(EPISODIC_NAMES) * len(EPISODIC_OBJECTS) > 40_000
    m, ev = _build(str(tmp_path), {"n_asks": (2, 4)})
    assert m["cast_mode"] == "episodic_v1"
    st = m["stats"]
    # novel facts carry the load; the roster is biography
    assert st["epi_asks"] / st["cast_asks"] >= 0.6
    for life in m["lives"]:
        e = life["episodic"]
        assert e["n_facts"] > 100
        # quota law: retired facts were asked exactly their quota (2..4)
        sample = [f for f in life["cast"] if str(f["cls"]).startswith("epi:")]
        assert sample and all(2 <= f["asks"] <= 4 for f in sample)
        # retirement law: nearly every fact retires inside the life
        assert e["retired"] >= 0.8 * e["n_facts"]
        # the roster is no longer the drill: far fewer asks per fact than
        # the episodic stream delivers per token
        roster = [f for f in life["cast"] if not str(f["cls"]).startswith("epi")]
        assert len(roster) == 24
        assert sum(f["asks"] for f in roster) < 0.4 * e["n_asks"]
    # every probe is a real ask with a positive gap and 4 distractors
    probes = [x for x in ev if x["kind"] == "probe"]
    assert probes and all(p["gap"] >= 1 and len(p["distractors"]) == 4
                          for p in probes)
    # in-context and short gaps both realized (the binder's food)
    gaps = [p["gap"] for p in probes]
    assert sum(g <= 256 for g in gaps) > 50
    assert sum(256 < g <= 2048 for g in gaps) > 50


def test_st2_order_is_curriculum():
    from iga.lm_data_life import st2_ordered, ST2_ORDER
    names = ["x/LongAlign_64k_a.parquet", "x/table_gpt_b.parquet",
             "x/smoltalk_smollm3_everyday_conversations.parquet",
             "x/OpenHermes_2.5.parquet", "x/unknown_subset.parquet",
             "x/Mixture_of_Thoughts_science.parquet"]
    order = [os.path.basename(p) for p in st2_ordered(names)]
    assert order[0].startswith("smoltalk_smollm3_everyday")
    assert order[-1].startswith("LongAlign")
    assert order.index("OpenHermes_2.5.parquet") < \
        order.index("Mixture_of_Thoughts_science.parquet")
    assert ST2_ORDER[-1] == "LongAlign"
