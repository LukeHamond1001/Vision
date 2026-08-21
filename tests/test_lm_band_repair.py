"""Band repair (docs/MEMORY_MATH.md section 5): band_credit, band_center,
tail_tokens.

Measured before the repair (2026-08-21): cells.k.cand and pred.k never
received gradient; cells.k.z only the write-cost penalty. Laws:

  B1  Defaults are bit-exact: flags off reproduce the certified forward
      and state values exactly, and the parameter set is unchanged.
  B2  With band_credit the VALUES are unchanged (same states, same
      logits) — only gradient routing differs.
  B3  With band_credit, over a multi-chunk run with the optimizer
      stepping between chunks and clocks > 1: cand, pred and z receive
      gradient beyond the write-cost penalty; no graph is traversed
      twice; no version error.
  B4  Hindsight: the next chunk's CE alone (no fidelity, no write cost)
      reaches the cell that wrote the state it read.
  B5  band_center: the fidelity target is the centred read; band_mu
      moves only in training.
  B6  tail_tokens: the extra token is tail_proj of the previous chunk's
      last TAIL_W hiddens' mean; zero in the first chunk, zero under
      mem_off and under the full amputation, live under store_read_off
      and under a partial lesion; the serve switches still hold.
  B7  train() runs with all three on (fp32 and bf16) and the flags ride
      the checkpoint cfg.
"""

import torch
import pytest

from iga.lm_hybrid import HybridLM
from iga.lm_sleep import state_copy

KW = dict(store="matrix", keyed="logit", norm_mix=True, aux_trunk=0.2,
          use_xl=False, gate_init=-2.0)
V, T = 512, 32
CLK = {3: 1, 4: 2, 5: 4, 6: 8}     # short clocks so every band ticks in a test


def _model(seed=0, **flags):
    torch.manual_seed(seed)
    m = HybridLM(V, d=64, n_layers=2, n_heads=4, max_T=T, clocks=CLK, **KW, **flags)
    m.read_drop = 0.0
    return m


def _toks(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (1, T), generator=g)


def _loss(m, lg, x, ticks, wc, rc, fid=True, cost=True):
    loss = torch.nn.functional.cross_entropy(lg[0], x[0])
    if fid:
        ft = [(1 - f).mean() for k in range(1, len(ticks)) for _, f in ticks[k] if f.requires_grad]
        if ft:
            loss = loss + 0.1 * torch.stack(ft).mean()
    if cost and wc is not None:
        loss = loss + 0.01 * wc
    if rc is not None:
        loss = loss + 0.05 * rc
    return loss


def _run(m, n, opt=None, fid=True, cost=True, seed0=100):
    st = m.init_state(1, "cpu")
    grads = {n_: 0.0 for n_, _ in m.named_parameters()}
    for c in range(n):
        x = _toks(seed0 + c)
        if opt: opt.zero_grad()
        else: m.zero_grad()
        lg, st, ticks = m(x, st, None)
        wc, rc = m.pop_write_cost(), m.pop_recon()
        _loss(m, lg, x, ticks, wc, rc, fid, cost).backward()
        for n_, p in m.named_parameters():
            if p.grad is not None:
                grads[n_] += float(p.grad.abs().sum())
        if opt: opt.step()
        st = m.detach_state(st)
    return grads, st


def test_B1_defaults_bit_exact():
    a, b = _model(), _model()
    names_a = {n for n, _ in a.named_parameters()}
    assert names_a == {n for n, _ in b.named_parameters()}
    assert "tail_proj.weight" not in names_a
    x = _toks(1)
    with torch.no_grad():
        sa, sb = a.init_state(1, "cpu"), b.init_state(1, "cpu")
        la, sa, _ = a(x, sa, None); a.pop_write_cost(); a.pop_recon()
        lb, sb, _ = b(x, sb, None); b.pop_write_cost(); b.pop_recon()
    assert torch.equal(la, lb)
    for k in a.bands:
        assert torch.equal(sa["h"][k], sb["h"][k])


def test_B2_credit_changes_gradients_not_values():
    a, b = _model(), _model(band_credit=True)
    with torch.no_grad():
        sa, sb = a.init_state(1, "cpu"), b.init_state(1, "cpu")
        for c in range(4):
            x = _toks(10 + c)
            la, sa, _ = a(x, sa, None); a.pop_write_cost(); a.pop_recon()
            lb, sb, _ = b(x, sb, None); b.pop_write_cost(); b.pop_recon()
            assert torch.allclose(la, lb, atol=1e-6)
            for k in a.bands:
                assert torch.allclose(sa["h"][k], sb["h"][k], atol=1e-6)
                if sa["pend"][k] is not None:
                    assert torch.allclose(sa["pend"][k], sb["pend"][k], atol=1e-6)
            sa, sb = a.detach_state(sa), b.detach_state(sb)


def test_B3_credit_trains_cand_pred_gate_across_optimizer_steps():
    m = _model(band_credit=True); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g, _ = _run(m, 12, opt=opt)          # 12 chunks: band 6 ticks once, band 5 thrice
    for k in (3, 4, 5):
        assert g[f"cells.{k}.cand.weight"] > 0, k
        assert g[f"pred.{k}.weight"] > 0, k
    # the gate gets more than the penalty: with the write cost off it still learns
    m2 = _model(band_credit=True); m2.train()
    g2, _ = _run(m2, 8, opt=torch.optim.AdamW(m2.parameters(), lr=1e-3), cost=False)
    assert g2["cells.3.z.weight"] > 0 and g2["cells.4.z.weight"] > 0
    # and the old behaviour, for the record: nothing reached cand/pred
    m0 = _model(); m0.train()
    g0, _ = _run(m0, 8, opt=torch.optim.AdamW(m0.parameters(), lr=1e-3))
    assert g0["cells.3.cand.weight"] == 0 and g0["pred.3.weight"] == 0


def test_B4_hindsight_next_chunk_ce_reaches_the_writer():
    m = _model(band_credit=True); m.train()
    st = m.init_state(1, "cpu")
    lg, st, _ = m(_toks(20), st, None); m.pop_write_cost(); m.pop_recon()
    st = m.detach_state(st)
    m.zero_grad()
    x = _toks(21)
    lg, st2, ticks = m(x, st, None); m.pop_write_cost(); m.pop_recon()
    torch.nn.functional.cross_entropy(lg[0], x[0]).backward()   # CE only
    assert float(m.cells["3"].cand.weight.grad.abs().sum()) > 0   # band 3 ticked last chunk
    assert float(m.cells["3"].z.weight.grad.abs().sum()) > 0


def test_B5_center_target_and_running_mean():
    m = _model(band_credit=True, band_center=True); m.train()
    assert float(m.band_mu.abs().sum()) == 0.0
    _run(m, 3)
    assert float(m.band_mu.abs().sum()) > 0.0
    mu = m.band_mu.clone()
    m.eval()
    with torch.no_grad():
        st = m.init_state(1, "cpu"); m(_toks(30), st, None); m.pop_write_cost(); m.pop_recon()
    assert torch.equal(mu, m.band_mu)          # eval never moves the centre


def test_B6_tail_token_content_and_switches():
    m = _model(tail_tokens=1); m.eval()
    assert m.pos.num_embeddings == T + len(m.bands) + 1
    x1, x2 = _toks(40), _toks(41)
    with torch.no_grad():
        st = m.init_state(1, "cpu")
        assert float(m._mem_tokens(st, 1)[:, -1].abs().sum()) == 0.0   # first chunk: nothing before
        hid = {}
        hk = m.lnf.register_forward_hook(lambda mod, i, o: hid.setdefault("h", i[0]))
        _, st, _ = m(x1, st, None); m.pop_write_cost(); m.pop_recon(); hk.remove()
        want = m.tail_proj(hid["h"][:, -m.TAIL_W:].mean(1))
        st = m.detach_state(st)
        got = m._mem_tokens(st, 1)[:, -1]
        assert torch.allclose(got, want, atol=1e-6)
        m.mem_off = True
        assert float(m._mem_tokens(st, 1)[:, -1].abs().sum()) == 0.0
        m.mem_off = False
        m.lesioned = set(m.bands)
        assert float(m._mem_tokens(st, 1)[:, -1].abs().sum()) == 0.0
        m.lesioned = {3}
        assert float(m._mem_tokens(st, 1)[:, -1].abs().sum()) > 0.0
        m.lesioned = set()
        m.store_read_off = True
        assert float(m._mem_tokens(st, 1)[:, -1].abs().sum()) > 0.0
        m.store_read_off = False
        lg, _, _ = m(x2, state_copy(st), None); m.pop_write_cost(); m.pop_recon()
        m.mem_off = True
        lg_off, _, _ = m(x2, state_copy(st), None); m.pop_write_cost(); m.pop_recon()
        m.mem_off = False
    assert not torch.allclose(lg, lg_off, atol=1e-5)          # the tail is read


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_B7_train_runs_with_all_three(tmp_path, precision):
    from iga.lm_train import train
    model, drive, vocab, ce0, ce1 = train(
        d=32, n_layers=2, lanes=2, T=32, steps=500, seed=0, device="cpu",
        arch="hybrid", store="matrix", keyed="hidden", norm_mix=True,
        aux_trunk=0.2, use_xl=False, gate_init=-2.0, lam=0.02,
        clocks=CLK, precision=precision, band_credit=True, band_center=True,
        tail_tokens=1, ckpt=str(tmp_path / "b.pt"), log_every=100)
    assert ce1 == ce1
    blob = torch.load(str(tmp_path / "b.pt"), map_location="cpu", weights_only=False)
    assert blob["cfg"]["band_credit"] and blob["cfg"]["tail_tokens"] == 1
    assert "tail_proj.weight" in blob["model"]
