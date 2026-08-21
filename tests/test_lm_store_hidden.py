"""Content-keyed store (keyed="hidden"), docs/MEMORY_MATH.md section 4.

Laws:
  H1  The write key never sees the value it stores: changing x_t leaves
      every write key at positions <= t unchanged (key_t = proj of the
      hidden at t-1; causal attention).
  H2  Keys are content, not position: the same token under two
      different contexts gets two different keys.
  H3  The A38 credit survives: the stored M carries exactly one
      write-op of graph across the chunk boundary — the NEXT chunk's
      CE backward reaches key_proj and tok_u; depth does not grow.
  H4  The recon pass teaches the trunk: the write-fidelity loss alone
      puts gradient on trunk blocks (the live hidden is in the key).
  H5  The read path is the logit path: with alpha = 0 or
      store_read_off the logits equal the plain head; with alpha > 0
      they differ; query_proj gets gradient through the read.
  H6  keyed="logit" is untouched: it still has qmix and no
      projections; keyed="hidden" has projections and no qmix.
  H7  train() runs end to end in hidden mode (bf16 path included).
"""

import torch
import pytest

from iga.lm_hybrid import HybridLM
from iga.lm_sleep import state_copy

KW = dict(store="matrix", norm_mix=True, aux_trunk=0.2, use_xl=False,
          gate_init=-2.0, clocks={3: 1, 4: 8, 5: 64, 6: 512})
V, T = 512, 32


def _model(keyed="hidden", seed=0):
    torch.manual_seed(seed)
    m = HybridLM(V, d=64, n_layers=2, n_heads=4, max_T=T, keyed=keyed, **KW)
    m.read_drop = 0.0              # reads always on in training (tests)
    return m


def _toks(seed, B=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (B, T), generator=g)


def _write_keys(m, x):
    """the write keys (pre-lift) exactly as the write block builds them"""
    m.eval()
    with torch.no_grad():
        st = m.init_state(x.shape[0], "cpu")
        # replicate: hidden at t-1 through key_proj, unit-normalized
        # (run the forward to get hidden via a hook on lnf input)
        hid = {}
        h = m.lnf.register_forward_hook(lambda mod, i, o: hid.setdefault("h", i[0]))
        m(x, st, None); m.pop_write_cost(); m.pop_recon()
        h.remove()
        hidden = hid["h"]
        h_prev = torch.cat([torch.zeros_like(hidden[:, :1]), hidden[:, :-1]], 1)
        return torch.nn.functional.normalize(h_prev @ m.key_proj.weight.t(), dim=-1)


def test_H1_write_key_never_sees_its_value():
    m = _model()
    x = _toks(1)
    k1 = _write_keys(m, x)
    x2 = x.clone(); t = 10; x2[0, t] = (x2[0, t] + 7) % V or 1
    k2 = _write_keys(m, x2)
    assert torch.allclose(k1[0, :t + 1], k2[0, :t + 1], atol=1e-6)
    assert not torch.allclose(k1[0, t + 1:], k2[0, t + 1:], atol=1e-4)


def test_H2_keys_are_content_not_position():
    m = _model()
    x = _toks(2); y = _toks(3)
    y[0, 20] = x[0, 20]                      # same token, different context
    kx, ky = _write_keys(m, x), _write_keys(m, y)
    # key at 21 (context strictly before 21 includes the shared token 20
    # but different earlier context) differs; identical context -> identical key
    assert not torch.allclose(kx[0, 21], ky[0, 21], atol=1e-4)
    assert torch.equal(_write_keys(m, x)[0, 21], kx[0, 21])
    # and the separation is the trunk's job, not the key's: at init the
    # untrained hiddens nearly coincide (the A55 geometry) — recorded,
    # not asserted beyond "different"


def test_H3_a38_credit_one_write_op_across_boundary():
    m = _model(); m.train()
    with torch.no_grad():
        for k in m.alpha: m.alpha[k].fill_(0.5)
    st = m.init_state(1, "cpu")
    lg, st, _ = m(_toks(4), st, None); m.pop_write_cost(); m.pop_recon()
    st = m.detach_state(st)
    for k in m.bands:
        assert st["M"][k].requires_grad            # one write-op of graph
    m.zero_grad()
    x2 = _toks(5)
    lg2, st2, _ = m(x2, st, None); m.pop_write_cost(); m.pop_recon()
    ce = torch.nn.functional.cross_entropy(lg2[0], x2[0])
    ce.backward()
    assert m.key_proj.weight.grad is not None and float(m.key_proj.weight.grad.abs().sum()) > 0
    assert m.tok_u.grad is not None and float(m.tok_u.grad.abs().sum()) > 0
    # depth does not grow: the next write starts from M.detach()
    st2 = m.detach_state(st2)
    m.zero_grad()
    x3 = _toks(6)
    lg3, _, _ = m(x3, st2, None); m.pop_write_cost(); m.pop_recon()
    torch.nn.functional.cross_entropy(lg3[0], x3[0]).backward()   # no double-free


def test_H4_recon_pass_trains_the_trunk():
    m = _model(); m.train()
    st = m.init_state(1, "cpu")
    m.zero_grad()
    _, st, _ = m(_toks(7), st, None); m.pop_write_cost()
    rc = m.pop_recon()
    assert rc is not None and rc.requires_grad
    rc.backward()
    blk = [p for n, p in m.named_parameters() if n.startswith("blocks.")]
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in blk)
    assert m.key_proj.weight.grad is not None


def test_H5_read_path_is_the_logit_path():
    m = _model(); m.eval()
    x = _toks(8)
    with torch.no_grad():
        st = m.init_state(1, "cpu")
        _, st, _ = m(x, st, None); m.pop_write_cost(); m.pop_recon()
        st = m.detach_state(st)
        x2 = _toks(9)
        lg_a0, _, _ = m(x2, state_copy(st), None); m.pop_write_cost(); m.pop_recon()
        for k in m.alpha: m.alpha[k].fill_(0.5)
        lg_on, _, _ = m(x2, state_copy(st), None); m.pop_write_cost(); m.pop_recon()
        m.store_read_off = True
        lg_off, _, _ = m(x2, state_copy(st), None); m.pop_write_cost(); m.pop_recon()
        m.store_read_off = False
    assert torch.allclose(lg_a0, lg_off, atol=1e-6)      # alpha 0 == reads off
    assert not torch.allclose(lg_on, lg_off, atol=1e-4)  # reads bite
    # query_proj learns through the read
    m.train(); m.zero_grad()
    lg_t, _, _ = m(x2, state_copy(st), None); m.pop_write_cost(); m.pop_recon()
    torch.nn.functional.cross_entropy(lg_t[0], x2[0]).backward()
    assert m.query_proj.weight.grad is not None and float(m.query_proj.weight.grad.abs().sum()) > 0


def test_H6_modes_have_their_own_parameters():
    ml, mh = _model("logit"), _model("hidden")
    assert hasattr(ml, "qmix") and not hasattr(ml, "key_proj")
    assert hasattr(mh, "key_proj") and hasattr(mh, "query_proj") and not hasattr(mh, "qmix")
    assert torch.equal(mh.key_proj.weight, torch.eye(64))
    assert set(mh.stores.keys()) == set(ml.stores.keys())


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_H7_train_runs_in_hidden_mode(tmp_path, precision):
    from iga.lm_train import train
    model, drive, vocab, ce0, ce1 = train(
        d=32, n_layers=2, lanes=2, T=32, steps=500, seed=0, device="cpu",
        arch="hybrid", store="matrix", keyed="hidden", norm_mix=True,
        aux_trunk=0.2, use_xl=False, gate_init=-2.0, lam=0.02,
        clocks={3: 1, 4: 8, 5: 64, 6: 512}, precision=precision,
        ckpt=str(tmp_path / "h.pt"), log_every=2)
    assert ce1 == ce1 and model.keyed == "hidden"
    blob = torch.load(str(tmp_path / "h.pt"), map_location="cpu", weights_only=False)
    assert blob["cfg"]["keyed"] == "hidden"
