"""The centerpiece's removal switches in the serve room (docs/CENTERPIECE.md).

Laws:
  L1  'none' is bit-exact: the read path with no removal equals the
      plain forward, and the model's flags are untouched after it.
  L2  The committed life is certified whatever the switch says: two
      sessions fed the same stream, one under 'bands', one under
      'store', end with the SAME band states and stores as the base.
  L3  The switches bite where they should: under 'bands' the memory
      tokens are zero and stores are unread; under 'store' the memory
      tokens are live and stores are unread; reply logits differ from
      the base once the life has written state.
  L4  The switch never outlives a forward — flags restore even when
      the read raises.
  L5  Unknown modes are refused; the switch is logged as an event.
"""

import types

import pytest
import torch

from iga.lm_hybrid import HybridLM
from iga.lm_serve import ServeSession, LESION_MODES
from iga.lm_sleep import state_copy
from iga.lm_vocab import PRESS_TOKENS

KW = dict(store="matrix", keyed="logit", norm_mix=True, aux_trunk=0.2,
          use_xl=False, gate_init=-2.0, clocks={3: 1, 4: 8, 5: 64, 6: 512})
V = 512
SPECIALS = {"<pad>": 500, "<eot_human>": 501, "<eot_model>": 502,
            **{t: 503 + i for i, t in enumerate(PRESS_TOKENS)}}


class _Tok:
    """A stand-in tokenizer: words hash below 500, specials fixed."""
    def token_to_id(self, t):
        return SPECIALS[t]

    def encode(self, text):
        ids = [SPECIALS.get(w, (abs(hash(w)) % 490) + 1)
               for w in text.split()]
        return types.SimpleNamespace(ids=ids)

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))


def _model(seed=0):
    torch.manual_seed(seed)
    m = HybridLM(V, d=64, n_layers=2, n_heads=4, max_T=64, **KW)
    # a fresh model reads its stores at alpha = 0 (the A30 bootstrap);
    # open the read so the 'store' switch has something to remove
    with torch.no_grad():
        for k in m.alpha:
            m.alpha[k].fill_(0.5)
    return m


def _session(seed=0, mode="none"):
    s = ServeSession(_model(seed), _Tok(), T=64, temperature=0.0,
                     max_reply=4, seed=seed)
    if mode != "none":
        s.lesion(mode)
    return s


# a long enough stream to force several commits and write state
STREAM = ["the red key was kept by mira in the attic room ."] * 12


def _feed(s):
    for line in STREAM:
        s.user(line)


def _flat(st):
    out = [st["h"][k] for k in sorted(st["h"])]
    out += [st["M"][k] for k in sorted(st["M"])]
    return torch.cat([t.flatten().float() for t in out])


def test_L1_none_is_bit_exact_and_leaves_flags_alone():
    s = _session()
    _feed(s)
    lg = s._next_logits()
    x = torch.tensor([s.pending], dtype=torch.long)
    with torch.no_grad():
        ref, _, _ = s.m(x, state_copy(s.st), None)
    s.m.pop_write_cost(); s.m.pop_recon()
    assert torch.equal(lg, ref[0, -1].float())
    assert s.m.lesioned == set() and s.m.store_read_off is False
    assert s.lesion_mode == "none"


@pytest.mark.parametrize("mode", ["bands", "store"])
def test_L2_commits_are_certified_under_any_switch(mode):
    base, cut = _session(), _session(mode=mode)
    _feed(base); _feed(cut)
    assert base.n_committed == cut.n_committed > 0
    assert torch.equal(_flat(base.st), _flat(cut.st))
    # the removed session read its logits through the switch; its
    # committed state is still the base's to the bit
    cut._next_logits()
    assert torch.equal(_flat(base.st), _flat(cut.st))


def test_L3_switches_bite_on_the_read_path():
    base = _session(); _feed(base)
    lg_base = base._next_logits()
    for mode in ("bands", "store"):
        s = _session(mode=mode); _feed(s)
        with s.lesion_scope() as m:
            if mode == "bands":
                assert m.lesioned == set(m.bands)
                assert m.store_read_off is False
                mem = m._mem_tokens(s.st, 1).detach()
                assert float(mem.abs().sum()) == 0.0
            else:
                assert m.lesioned == set()
                assert m.store_read_off is True
                mem = m._mem_tokens(s.st, 1).detach()
                assert float(mem.abs().sum()) > 0.0
        assert m.lesioned == set() and m.store_read_off is False
        lg = s._next_logits()
        assert not torch.equal(lg, lg_base), mode
    # the base's own memory tokens are live (the bands wrote state)
    assert float(base.m._mem_tokens(base.st, 1).detach().abs().sum()) > 0.0


def test_L4_switch_never_outlives_a_forward_even_on_error():
    s = _session(mode="bands")
    with pytest.raises(RuntimeError):
        with s.lesion_scope() as m:
            assert m.lesioned == set(m.bands)
            raise RuntimeError("mid-read failure")
    assert s.m.lesioned == set() and s.m.store_read_off is False


def test_L5_modes_are_closed_and_logged():
    s = _session()
    with pytest.raises(ValueError):
        s.lesion("cortex")
    assert s.lesion("STORE") == "store"
    assert s.lesion("") == "none"
    kinds = [e for e in s.events if e["kind"] == "lesion"]
    assert [e["mode"] for e in kinds] == ["store", "none"]
    assert set(LESION_MODES) == {"none", "bands", "store"}
    assert s.panel()["lesion"] == "none"
