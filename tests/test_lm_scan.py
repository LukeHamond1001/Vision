"""The one-token organism (iga/lm_scan.py) — laws.

  S1  One token: with every organ off (all bands lesioned, reads off)
      the logits at t are a function of x_t alone — changing x_{t-1}
      leaves them bit-identical; with the organs on, x_{t-1} reaches
      them (through the bands).
  S2  Credit: CE alone trains the cells within the chunk (BPTT), and
      the NEXT chunk's CE alone reaches the cell that wrote the state
      it read (the carried cloned graph); many chunks with optimizer
      steps between them and bands ticking every 2 chunks never
      traverse a graph twice.
  S3  Veto: the other bands' vetoes multiply the gate — fully vetoed,
      a band's state does not move; un-vetoed it moves exactly as
      with veto=False.
  S4  Hippocampus: the read is the logit path (alpha 0 == reads off;
      alpha > 0 changes logits; store_read_off recovers the plain
      head) AND a council slot (store_in receives gradient through
      the slot).
  S5  Removal switches: mem_off zeroes the band slots but keeps the
      reads; lesioned = all bands removes both; the switches restore
      through the serve room's lesion_scope.
  S6  Write cadence: M changes only every write_every chunks; after a
      write it carries one write-op of graph for exactly the first
      reading chunk, then reads a detached M.
  S7  pfc_first (the user's design): same API; the neocortex's query
      is the PFC's token slot and its key/value slots are the rest of
      the PFC bundle — it never receives the raw embedding or the raw
      band states; band removal changes logits only through the PFC.
  S8  train() runs end to end in fp32 and bf16 (states, stores and
      logits stay fp32 under bf16); arch, clocks and the scan options
      ride the checkpoint cfg; horizons = clocks in tokens.
  S11 The batched decoder is the per-token decoder: pfc_first logits
      from the one-call neocortex equal a token-by-token reference run
      of the same blocks on the same bundles (fp32, 1e-5).
  S12 The hippocampus is a PFC organ: the slot-refresh query, the logit
      read query and the write keys are the council's token slot — in
      both orders — and prev_c carries the last token's slot.
  S10 band_center: in training the target is centred by the batch mean
      at the tick (the cortex's training drift cancels — a lagging mean
      left it in, and a predictor bias matched it: scan1 fid:5 = +1.000);
      each band keeps a running mean of its own pooled target (seeded by
      its first scored tick) for eval/serve; bands that never ticked keep
      a zero row; eval never moves it.
  S9  Precision law: under bf16 every Linear in the trunk (neocortex)
      computes in bf16 and every Linear in the council (PFC), the
      cells, the predictors and the head computes in fp32 — in both
      orders; in fp32 everything is fp32.
"""

import torch
import pytest

from iga.lm_scan import ScanLM
from iga.lm_sleep import state_copy

V, T = 512, 32
CLK = {3: 1, 4: 8, 5: 64, 6: 512}


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    opts = dict(d=32, n_layers=2, n_heads=4, max_T=T, clocks=CLK,
                n_council=1, write_every=2)
    opts.update(kw)
    m = ScanLM(V, **opts)
    m.read_drop = 0.0
    return m


def _toks(seed, B=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, (B, T), generator=g)


def _fwd(m, x, st):
    lg, st, ticks = m(x, st, None)
    wc, rc = m.pop_write_cost(), m.pop_recon()
    return lg, st, ticks, wc, rc


def _loss(m, lg, x, ticks, wc, rc, ce_only=False):
    loss = torch.nn.functional.cross_entropy(lg[0], x[0])
    if ce_only:
        return loss
    ft = [(1 - f).mean() for k in range(len(ticks)) for _, f in ticks[k]
          if f.requires_grad]
    if ft:
        loss = loss + 0.1 * torch.stack(ft).mean()
    if wc is not None:
        loss = loss + 0.01 * wc
    if rc is not None:
        loss = loss + 0.05 * rc
    return loss


def test_S1_one_token():
    m = _model(); m.eval()
    x = _toks(1); x2 = x.clone(); x2[0, 9] = (x2[0, 9] + 5) % V or 1
    with torch.no_grad():
        m.lesioned = set(m.bands); m.store_read_off = True
        a, _, _, _, _ = _fwd(m, x, m.init_state(1, "cpu"))
        b, _, _, _, _ = _fwd(m, x2, m.init_state(1, "cpu"))
        m.lesioned = set(); m.store_read_off = False
        assert torch.equal(a[0, 10:], b[0, 10:])        # nothing but x_t
        assert not torch.equal(a[0, 9], b[0, 9])
        a, _, _, _, _ = _fwd(m, x, m.init_state(1, "cpu"))
        b, _, _, _, _ = _fwd(m, x2, m.init_state(1, "cpu"))
        assert not torch.allclose(a[0, 10], b[0, 10])    # the bands carry it
        assert torch.equal(a[0, :9], b[0, :9])           # causal


def test_S2_credit_within_and_across_chunks():
    m = _model(); m.train()
    st = m.init_state(1, "cpu")
    x = _toks(2)
    m.zero_grad()
    lg, st, ticks, wc, rc = _fwd(m, x, st)
    _loss(m, lg, x, ticks, wc, rc, ce_only=True).backward()
    for k in (3, 4):
        assert float(m.cells[str(k)].cand.weight.grad.abs().sum()) > 0, k
        assert float(m.cells[str(k)].z.weight.grad.abs().sum()) > 0, k
    st = m.detach_state(st)
    for k in m.bands:
        if k in (3, 4):
            assert st["h"][k].requires_grad          # the carried one-op graph
    m.zero_grad()
    x2 = _toks(3)
    lg, st, ticks, wc, rc = _fwd(m, x2, st)
    torch.nn.functional.cross_entropy(lg[0, :1], x2[0, :1]).backward()  # first token's CE only
    assert float(m.cells["3"].cand.weight.grad.abs().sum()) > 0      # hindsight reaches the writer
    # many chunks, optimizer stepping, band 5 ticking every 2 chunks: no double traversal
    m2 = _model(seed=1); m2.train()
    opt = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    st = m2.init_state(1, "cpu")
    for c in range(9):
        x = _toks(100 + c)
        opt.zero_grad()
        lg, st, ticks, wc, rc = _fwd(m2, x, st)
        _loss(m2, lg, x, ticks, wc, rc).backward()
        opt.step()
        st = m2.detach_state(st)
    assert st["tok"] == 9 * T and st["chunk"] == 9


def test_S3_veto_multiplies_the_gate():
    a = _model(seed=3, veto=True)
    b = _model(seed=3, veto=False)
    x = _toks(4)
    with torch.no_grad():
        for k in a.veto_b:
            a.veto_b[k].fill_(-30.0)                  # no veto at all
        _, sa, _, _, _ = _fwd(a, x, a.init_state(1, "cpu"))
        _, sb, _, _, _ = _fwd(b, x, b.init_state(1, "cpu"))
        for k in a.bands:
            assert torch.allclose(sa["h"][k], sb["h"][k], atol=1e-5), k
        for k in a.veto_b:
            a.veto_b[k].fill_(30.0)                   # every band vetoed by every other
        _, sv, _, _, _ = _fwd(a, x, a.init_state(1, "cpu"))
        for k in a.bands:
            assert float(sv["h"][k].abs().max()) < 1e-5, k   # the state never moved
        assert all(v > 0.99 for v in a._veto_mean.values())


def test_S4_hippocampus_logit_path_and_council_slot():
    m = _model(seed=5); m.eval()
    x, x2 = _toks(6), _toks(7)
    with torch.no_grad():
        st = m.init_state(1, "cpu")
        _, st, _, _, _ = _fwd(m, x, st); st = m.detach_state(st)
        _, st, _, _, _ = _fwd(m, x2, st); st = m.detach_state(st)   # chunk 2: a write
        assert st["chunk"] == 2 and float(st["M"][3].abs().sum()) > 0   # written
        x3 = _toks(8)
        lg_a0, _, _, _, _ = _fwd(m, x3, state_copy(st))
        for k in m.alpha:
            m.alpha[k].fill_(0.5)
        lg_on, _, _, _, _ = _fwd(m, x3, state_copy(st))
        m.store_read_off = True
        lg_off, _, _, _, _ = _fwd(m, x3, state_copy(st))
        m.store_read_off = False
    assert torch.allclose(lg_a0, lg_off, atol=1e-6)
    assert not torch.allclose(lg_on, lg_off, atol=1e-4)
    # the council slot: store_in (zero at init) gets gradient only when a read exists
    m.train(); m.zero_grad()
    lg_t, _, _, _, _ = _fwd(m, x3, state_copy(st))
    torch.nn.functional.cross_entropy(lg_t[0], x3[0]).backward()
    assert float(m.store_in.weight.grad.abs().sum()) > 0
    assert float(m.query_proj.weight.grad.abs().sum()) > 0


def test_S5_removal_switches_and_serve_scope():
    m = _model(seed=9); m.eval()
    with torch.no_grad():
        for k in m.alpha:
            m.alpha[k].fill_(0.5)
        st = m.init_state(1, "cpu")
        for s in (10, 11):
            _, st, _, _, _ = _fwd(m, _toks(s), st); st = m.detach_state(st)
        x = _toks(12)
        base, _, _, _, _ = _fwd(m, x, state_copy(st))
        m.mem_off = True
        thread_off, _, _, _, _ = _fwd(m, x, state_copy(st))
        m.mem_off = False
        m.store_read_off = True
        store_off, _, _, _, _ = _fwd(m, x, state_copy(st))
        m.store_read_off = False
        m.lesioned = set(m.bands)
        both, _, _, _, _ = _fwd(m, x, state_copy(st))
        m.lesioned = set()
        m.mem_off = True; m.store_read_off = True
        both2, _, _, _, _ = _fwd(m, x, state_copy(st))
        m.mem_off = False; m.store_read_off = False
    assert not torch.allclose(base, thread_off, atol=1e-5)
    assert not torch.allclose(base, store_off, atol=1e-5)
    assert not torch.allclose(thread_off, both, atol=1e-5)
    assert torch.allclose(both, both2, atol=1e-5)       # lesioned-all == thread off + reads off
    # the serve room's scope restores the switches
    from iga.lm_serve import ServeSession

    class _Tok:
        def token_to_id(self, t):
            return {"<eot_human>": 1, "<eot_model>": 2, "<pad>": 0,
                    "<+1>": 3, "<+2>": 4, "<-1>": 5, "<-2>": 6}.get(t)

        def get_vocab_size(self):
            return V

    s = ServeSession(m, _Tok(), T=T, device="cpu")
    s.lesion("bands")
    with s.lesion_scope():
        assert m.mem_off is True
    assert m.mem_off is False and m.lesioned == set() and m.store_read_off is False


def test_S6_write_cadence_and_one_op_credit():
    m = _model(seed=13); m.train()
    st = m.init_state(1, "cpu")
    M0 = {k: v.clone() for k, v in st["M"].items()}
    lg, st, ticks, wc, rc = _fwd(m, _toks(20), st)
    assert rc is None and all(torch.equal(st["M"][k], M0[k]) for k in m.bands)
    _loss(m, lg, _toks(20), ticks, wc, rc).backward(); st = m.detach_state(st)
    lg, st, ticks, wc, rc = _fwd(m, _toks(21), st)          # chunk 2: the write
    assert rc is not None and rc.requires_grad
    assert all(st["M"][k].requires_grad for k in m.bands)
    _loss(m, lg, _toks(21), ticks, wc, rc).backward(); st = m.detach_state(st)
    assert all(st["M"][k].requires_grad for k in m.bands)    # kept for the first reader
    m.zero_grad()
    with torch.no_grad():
        for k in m.alpha:
            m.alpha[k].fill_(0.5)
    x = _toks(22)
    lg, st, ticks, wc, rc = _fwd(m, x, st)
    torch.nn.functional.cross_entropy(lg[0], x[0]).backward()   # CE only
    assert float(m.key_proj.weight.grad.abs().sum()) > 0      # the read credits the write
    assert float(m.tok_u.grad.abs().sum()) > 0
    st = m.detach_state(st)
    assert not any(st["M"][k].requires_grad for k in m.bands)  # detached after its reader
    x = _toks(23)
    lg, st, ticks, wc, rc = _fwd(m, x, st)                       # chunk 4: a write again
    _loss(m, lg, x, ticks, wc, rc).backward()                    # no double traversal


def test_S7_pfc_first_order():
    m = _model(seed=17, order="pfc_first"); m.eval()
    x = _toks(30)
    # the neocortex sees only what the PFC outputs: capture what the
    # council returns token by token and what the decoder receives
    seen = {"council": [], "trunk": []}
    council0, trunk0 = m._council, m._trunk

    def council(c, r, mm, dev):
        S = council0(c, r, mm, dev)
        seen["council"].append(S.detach().clone())
        return S

    def trunk(e, slots, dev):
        seen["trunk"].append((e.detach().clone(),
                              None if slots is None else slots.detach().clone()))
        return trunk0(e, slots, dev)
    m._council, m._trunk = council, trunk
    with torch.no_grad():
        st = m.init_state(1, "cpu")
        lg, st, ticks, _, _ = _fwd(m, x, st)
        assert lg.shape == (1, T, V) and len(ticks[3]) == T - 1
        emb = m.embed(x)[0]
        K = len(m.bands)
        assert len(seen["council"]) == T
        # ONE decoder call per chunk (the neocortex is off the recurrent
        # path), fed the PFC's token slots as queries and the rest of
        # each token's bundle as its key/value slots
        assert len(seen["trunk"]) == 1
        e_in, slots = seen["trunk"][0]
        assert e_in.shape == (T, m.d) and slots.shape == (T, 1 + K, m.d)
        S_all = torch.cat(seen["council"], 0)           # [T, 2+K, d]
        assert torch.equal(e_in, S_all[:, 0])            # query = PFC token slot
        assert torch.equal(slots, S_all[:, 1:])          # kv = the PFC bundle
        assert not torch.allclose(e_in, emb)             # never the raw tokens
    m._council, m._trunk = council0, trunk0
    with torch.no_grad():
        st = m.detach_state(st)
        x2 = _toks(31)
        base, _, _, _, _ = _fwd(m, x2, state_copy(st))
        m.mem_off = True
        off, _, _, _, _ = _fwd(m, x2, state_copy(st))
        m.mem_off = False
    assert not torch.allclose(base, off, atol=1e-5)


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_S8_train_runs_both_precisions(tmp_path, precision):
    from iga.lm_train import train
    model, drive, vocab, ce0, ce1 = train(
        d=32, n_layers=2, lanes=2, T=32, steps=500, seed=0, device="cpu",
        arch="scan", store="matrix", keyed="hidden", norm_mix=True,
        aux_trunk=0.2, use_xl=False, gate_init=-2.0, lam=0.02,
        clocks=CLK, precision=precision,
        scan={"n_council": 1, "write_every": 2},
        ckpt=str(tmp_path / "s.pt"), log_every=250)
    assert ce1 == ce1 and ce1 < ce0
    assert model.autocast_bf16 == (precision == "bf16")
    st = model._st
    assert st["h"][3].dtype == torch.float32 and st["M"][3].dtype == torch.float32
    with torch.no_grad():
        x = torch.randint(1, model.vocab_size, (2, T))
        lg, _, _, _, _ = _fwd(model, x, state_copy(st))
    assert lg.dtype == torch.float32
    blob = torch.load(str(tmp_path / "s.pt"), map_location="cpu", weights_only=False)
    cfg = blob["cfg"]
    assert cfg["arch"] == "scan" and cfg["scan"] == {"n_council": 1, "write_every": 2}
    assert {int(k): v for k, v in cfg["clocks"].items()} == CLK
    assert dict(drive._horizons) == CLK


@pytest.mark.parametrize("order", ["cortex_first", "pfc_first"])
@pytest.mark.parametrize("bf16", [False, True])
def test_S9_precision_law_trunk_bf16_pfc_fp32(order, bf16):
    m = _model(seed=5, order=order); m.eval()
    m.autocast_bf16 = bf16
    dtypes = {"trunk": set(), "council": set(), "cells": set(), "pred": set(),
              "head": set()}

    def hook(name):
        def f(mod, inp, out):
            dtypes[name].add(out.dtype)
        return f
    hs = []
    for lin in m.blocks.modules():
        if isinstance(lin, torch.nn.Linear):
            hs.append(lin.register_forward_hook(hook("trunk")))
    for lin in m.council.modules():
        if isinstance(lin, torch.nn.Linear):
            hs.append(lin.register_forward_hook(hook("council")))
    for lin in m.cells.modules():
        if isinstance(lin, torch.nn.Linear):
            hs.append(lin.register_forward_hook(hook("cells")))
    for lin in m.pred.modules():
        if isinstance(lin, torch.nn.Linear):
            hs.append(lin.register_forward_hook(hook("pred")))
    hs.append(m.head.register_forward_hook(hook("head")))
    x = _toks(40)
    with torch.no_grad():
        st = m.init_state(1, "cpu")
        lg, st, _, _, _ = _fwd(m, x, st)
        st = m.detach_state(st)
        lg, st, _, _, _ = _fwd(m, _toks(41), st)     # a chunk with reads + writes
    for h in hs:
        h.remove()
    assert lg.dtype == torch.float32
    assert dtypes["trunk"] == {torch.bfloat16 if bf16 else torch.float32}
    for name in ("council", "cells", "pred", "head"):
        assert dtypes[name] == {torch.float32}, (name, dtypes[name])
    for k in m.bands:
        assert st["h"][k].dtype == torch.float32
        assert st["M"][k].dtype == torch.float32


def test_S10_band_center_is_per_band():
    m = _model(seed=9, order="pfc_first"); m.train()
    st = m.init_state(2, "cpu")
    x = _toks(50, B=2)
    seen = {}
    pred0 = {k: m.pred[str(k)] for k in m.bands}
    lg, st, ticks, wc, rc = _fwd(m, x, st)
    # band 3 ticks every token, band 4 every 8: both rows seeded and
    # moving (the mean is updated at SCORED ticks — a tick with a pend —
    # so the first tick of each band does not count)
    assert m.band_mu.shape == (max(m.bands) + 1, m.d)
    assert int(m.band_mu_n[3]) == T - 1 and int(m.band_mu_n[4]) == T // 8 - 1
    assert m.band_mu[3].abs().sum() > 0 and m.band_mu[4].abs().sum() > 0
    assert not torch.allclose(m.band_mu[3], m.band_mu[4])
    # bands 5 (clock 64) and 6 (512) never ticked in 32 tokens: zero rows
    assert int(m.band_mu_n[5]) == 0 and m.band_mu[5].abs().sum() == 0
    assert int(m.band_mu_n[6]) == 0 and m.band_mu[6].abs().sum() == 0
    # the fidelity target is the deviation: at band 3's tick the fid is
    # cos(pend, pooled_c - batch mean) — not saturated at 1 for a random net
    fids = torch.stack([f for _, f in ticks[3]])
    assert fids.abs().max() < 0.999
    # training centres by the batch mean at the tick: with 2 lanes the two
    # targets are exact negatives, so the two lanes' fids against the same
    # pend would be negatives — check the target itself through a tick of
    # a fresh model whose pend is the all-ones direction
    m2 = _model(seed=9, order="pfc_first"); m2.train()
    for k in m2.bands:
        torch.nn.init.zeros_(m2.pred[str(k)].weight); torch.nn.init.ones_(m2.pred[str(k)].bias)
    st2 = m2.init_state(2, "cpu")
    _, _, ticks2, _, _ = _fwd(m2, _toks(52, B=2), st2)
    f = torch.stack([f for _, f in ticks2[3]])            # [n_ticks, 2]
    assert torch.allclose(f[:, 0], -f[:, 1], atol=1e-5)
    # eval never moves the means
    m.eval()
    mu = m.band_mu.clone(); n = m.band_mu_n.clone()
    with torch.no_grad():
        _fwd(m, _toks(51, B=2), m.detach_state(st))
    assert torch.equal(mu, m.band_mu) and torch.equal(n, m.band_mu_n)


def test_S11_batched_decoder_equals_per_token():
    m = _model(seed=23, order="pfc_first"); m.eval()
    m.store_read_off = True                     # logits = head(lnf(C)) only
    seen = []
    council0 = m._council

    def council(c, r, mm, dev):
        S = council0(c, r, mm, dev)
        seen.append(S.detach().clone())
        return S
    m._council = council
    x = _toks(60)
    with torch.no_grad():
        st = m.init_state(1, "cpu")
        lg, st, _, _, _ = _fwd(m, x, st)
        # the reference: the same decoder blocks, one token at a time
        ref = torch.stack([m._trunk(S[:, 0], S[:, 1:], torch.device("cpu"))
                           for S in seen], dim=1)            # [1, T, d]
        lg_ref = m.head(m.lnf(ref))
    m._council = council0
    assert torch.allclose(lg, lg_ref, atol=1e-5), (lg - lg_ref).abs().max()


@pytest.mark.parametrize("order", ["cortex_first", "pfc_first"])
def test_S12_hippocampus_is_a_pfc_organ(order):
    m = _model(seed=29, order=order, slot_every=8); m.eval()
    m.read_drop = 0.0
    council_out, read_q = [], []
    council0, read0 = m._council, m._read

    def council(c, r, mm, dev):
        S = council0(c, r, mm, dev)
        council_out.append(S.detach().clone())
        return S

    def read(st, q, ok):
        read_q.append(q.detach().clone())
        return read0(st, q, ok)
    m._council, m._read = council, read
    with torch.no_grad():
        st = m.init_state(1, "cpu")
        _, st, _, _, _ = _fwd(m, _toks(70), st)      # chunk 0: no reads (no M yet) but queries are recorded
        council_out.clear(); read_q.clear()
        st = m.detach_state(st)
        _, st, _, _, _ = _fwd(m, _toks(71), st)      # chunk 1
    S_all = torch.cat(council_out, 0)                 # [T, 2+K, d]
    s0 = S_all[:, 0]
    # chunk-start read (prev_c), slot refreshes at t = 7, 15, 23, then the
    # batched logit read over the chunk — every query is a PFC token slot
    refresh = [q for q in read_q if q.dim() == 2]
    batched = [q for q in read_q if q.dim() == 3]
    assert len(batched) == 1 and torch.equal(batched[0][0], s0)
    assert len(refresh) == 1 + (T // 8 - 1)
    for j, q in enumerate(refresh[1:]):
        assert torch.equal(q[0], s0[8 * (j + 1) - 1])
    assert torch.equal(st["prev_c"][0], s0[-1])       # the write key for the next token
    m._council, m._read = council0, read0
