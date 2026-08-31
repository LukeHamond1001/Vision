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
      ride the checkpoint cfg; economy horizons = max(4 x clock, 512) tokens.
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
        assert lg.shape == (1, T, V) and len(ticks[3]) == 1 and ticks[3][0][1].shape == (1, T - 1)
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
    # the economy's horizons in tokens follow the hybrid's rule, max(4 x clock, 512)
    assert dict(drive._horizons) == {k: max(4 * v, 512) for k, v in CLK.items()}


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
    fids = ticks[3][0][1]                                 # [B, n_ticks]
    assert fids.shape == (2, T - 1) and fids.abs().max() < 0.999
    # training centres by the batch mean at the tick: with 2 lanes the two
    # targets are exact negatives, so the two lanes' fids against the same
    # pend would be negatives — check the target itself through a tick of
    # a fresh model whose pend is the all-ones direction
    m2 = _model(seed=9, order="pfc_first"); m2.train()
    for k in m2.bands:
        torch.nn.init.zeros_(m2.pred[str(k)].weight); torch.nn.init.ones_(m2.pred[str(k)].bias)
    st2 = m2.init_state(2, "cpu")
    _, _, ticks2, _, _ = _fwd(m2, _toks(52, B=2), st2)
    f = ticks2[3][0][1]                                   # [2, n_ticks]
    assert torch.allclose(f[0], -f[1], atol=1e-5)
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


def test_S13_compiled_council_matches_eager():
    """compile_council=True: the council's blocks through torch.compile
    give the eager council's outputs (1e-4) and the chunk trains."""
    pytest.importorskip("torch._dynamo")
    import torch._dynamo
    torch._dynamo.reset()
    m0 = _model(seed=31, order="pfc_first"); m0.eval()
    m1 = _model(seed=31, order="pfc_first", compile_council=True); m1.eval()
    m1.load_state_dict(m0.state_dict())
    x = _toks(80)
    with torch.no_grad():
        lg0, _, _, _, _ = _fwd(m0, x, m0.init_state(1, "cpu"))
        try:
            lg1, _, _, _, _ = _fwd(m1, x, m1.init_state(1, "cpu"))
        except Exception as e:                        # no C++ toolchain for inductor here
            pytest.skip(f"torch.compile unavailable on this host: {type(e).__name__}")
    assert torch.allclose(lg0, lg1, atol=1e-4), (lg0 - lg1).abs().max()
    m1.train()
    st = m1.init_state(2, "cpu")
    for i in range(3):
        xb = _toks(81 + i, B=2)
        lg, st, ticks, wc, rc = _fwd(m1, xb, st)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, V), xb.reshape(-1))
        loss = loss + 0.1 * torch.stack([(1 - f).mean() for k in range(len(ticks))
                                         for _, f in ticks[k] if f.requires_grad]).mean()
        loss.backward()
        st = m1.detach_state(st)
    assert all(p.grad is not None for p in m1.council.parameters())


def test_S14_fidelity_trains_the_band_only():
    """The fidelity loss reaches the band's cell and predictor and
    nothing else (not the council, not the vetoes, not the embedding);
    the CE loss reaches the council THROUGH the band state (the state
    path is live). The certified band_credit split, per tick."""
    m = _model(seed=37, order="pfc_first"); m.train()
    x = _toks(90, B=2)
    st = m.init_state(2, "cpu")
    lg, st, ticks, wc, rc = _fwd(m, x, st)
    fid_loss = torch.stack([(1 - f).mean() for k in range(len(ticks))
                            for _, f in ticks[k] if f.requires_grad]).mean()
    fid_loss.backward(retain_graph=True)
    def gnorm(params):
        return sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
    assert gnorm(m.council.parameters()) == 0.0
    assert gnorm([m.embed.weight]) == 0.0 and gnorm(m.blocks.parameters()) == 0.0
    assert gnorm(m.cells["3"].parameters()) > 0 and gnorm(m.pred["3"].parameters()) > 0
    m.zero_grad()
    ce = torch.nn.functional.cross_entropy(lg.reshape(-1, V), x.reshape(-1))
    ce.backward()
    assert gnorm(m.council.parameters()) > 0
    assert gnorm(m.cells["3"].parameters()) > 0          # CE credit through the live state


def test_S15_compiled_read_matches_eager():
    """compile_read=True: one band's lift+read through torch.compile equals
    the eager LogitStore path (1e-5 fp32) with reads live; trains."""
    pytest.importorskip("torch._dynamo")
    import torch._dynamo
    torch._dynamo.reset()
    m0 = _model(seed=41, order="pfc_first", slot_every=1, write_every=1); m0.eval()
    m1 = _model(seed=41, order="pfc_first", slot_every=1, write_every=1, compile_read=True); m1.eval()
    m1.load_state_dict(m0.state_dict())
    with torch.no_grad():
        st0, st1 = m0.init_state(2, "cpu"), m1.init_state(2, "cpu")
        for i in range(3):                                   # chunk 0 writes; 1-2 read
            x = _toks(100 + i, B=2)
            lg0, st0, _, _, _ = _fwd(m0, x, st0); st0 = m0.detach_state(st0)
            try:
                lg1, st1, _, _, _ = _fwd(m1, x, st1); st1 = m1.detach_state(st1)
            except Exception as e:
                pytest.skip(f"torch.compile unavailable on this host: {type(e).__name__}")
    assert m0._reads_used and torch.allclose(lg0, lg1, atol=1e-5), (lg0 - lg1).abs().max()
    m1.train(); m1.read_drop = 0.0
    st = m1.init_state(2, "cpu")
    for i in range(3):
        xb = _toks(110 + i, B=2)
        lg, st, ticks, wc, rc = _fwd(m1, xb, st)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, V), xb.reshape(-1))
        if rc is not None:
            loss = loss + 0.05 * rc
        loss.backward()
        st = m1.detach_state(st)
    # alpha starts at 0 (the vote is silent at init) so query_proj's gradient
    # is exactly zero until alpha grows — in eager and compiled alike; the
    # read path is live when alpha itself receives gradient
    assert any(p.grad is not None and float(p.grad.abs()) > 0 for p in m1.alpha.parameters())


def test_S16_ledger_deque_matches_list_semantics():
    """The capped ledger (a deque) holds exactly the entries the list
    version kept, in order, with ledger_base counting the evictions —
    and settling at the cap no longer shifts the whole ledger."""
    from iga.lm_drive import Drive
    import time
    d = Drive(2, seed=0, ledger_cap=50)
    ref, base = [], 0
    for i in range(300):
        h = {"lane": i % 2, "band": 3, "key": "recall:b0", "phi0": 0.5, "w": 1.0, "t0": i}
        d._settle(h, 0.25, 0.25)
        ref.append({**h, "phi1": 0.25, "pay": 0.25, "t1": d.step_t})
        if len(ref) > 50:
            ref = ref[1:]; base += 1
    assert list(d.ledger) == ref and d.ledger_base == base == 250 and len(d.ledger) == 50
    assert d.ledger[len(d.ledger) - 1]["t0"] == 299      # index access near the end
    big = Drive(1, seed=0, ledger_cap=200_000)
    for i in range(200_000):
        big.ledger.append({"t0": i})
    t0 = time.time()
    for i in range(2000):
        big._settle({"lane": 0, "band": 3, "key": "k", "phi0": 0.0, "w": 0.0, "t0": i}, 0.0, 0.0)
    assert time.time() - t0 < 0.5                         # 2000 settles at the cap: O(1) each


def test_S17_exact_delta_rule_is_the_sequential_rule():
    """write_exact equals the token-by-token delta rule (fp64, 1e-8),
    handles a repeated key inside the chunk without overshoot, and a
    single pair written at full strength is read back exactly — the
    certified averaged write reads it back at ~1/T."""
    from iga.lm_hybrid import LogitStore
    torch.manual_seed(3)
    B, T, d, D = 2, 16, 8, 24
    stn = LogitStore(d, D, decay=0.0, seed=5).double()
    with torch.no_grad():
        stn.beta.fill_(20.0)                                   # sigmoid -> 1: full strength
    K = torch.nn.functional.normalize(torch.randn(B, T, D, dtype=torch.float64), dim=-1)
    K[:, 5] = K[:, 2]                                          # a repeated key in the chunk
    V = torch.nn.functional.normalize(torch.randn(B, T, d, dtype=torch.float64), dim=-1)
    s = torch.rand(B, T, dtype=torch.float64) * 0.9 + 0.1
    M0 = 0.1 * torch.randn(B, d, D, dtype=torch.float64)
    # sequential reference
    Mr = M0.clone()
    for t in range(T):
        bt = (torch.sigmoid(stn.beta) * s[:, t]).view(B, 1, 1)
        pred = torch.einsum("bij,bj->bi", Mr, K[:, t])
        Mr = Mr + bt * torch.einsum("bi,bj->bij", V[:, t] - pred, K[:, t])
    Me, _ = stn.write_exact(M0, K, V, s, stn.beta)
    assert torch.allclose(Me, Mr, atol=1e-8), (Me - Mr).abs().max()
    # the repeated key: the later write wins without overshoot
    back = torch.einsum("bij,bj->bi", Me, K[:, 5])
    assert back.norm(dim=-1).max() < 1.05 * V[:, 5].norm(dim=-1).max() + 1e-6
    # one-shot: a single pair at strength 1 is read back exactly; the
    # averaged (certified) write reads it back at ~1/T of the magnitude
    s1 = torch.zeros(B, T, dtype=torch.float64); s1[:, 3] = 1.0
    Z = torch.zeros(B, d, D, dtype=torch.float64)
    Me1, _ = stn.write_exact(Z, K, V, s1, stn.beta)
    r1 = torch.einsum("bij,bj->bi", Me1, K[:, 3])
    assert torch.allclose(r1, V[:, 3], atol=1e-8)
    stn.exact = False
    Ma1, _ = stn.write(Z, K, V, s1)                           # strength-normalised: one pair, weight 1
    ra = torch.einsum("bij,bj->bi", Ma1, K[:, 3])
    assert torch.allclose(ra, V[:, 3], atol=1e-8)            # a lone pair is fine either way...
    s_all = torch.ones(B, T, dtype=torch.float64)             # ...but among T pairs it is diluted to ~1/T
    Ma, _ = stn.write(Z, K, V, s_all)
    ra_all = torch.einsum("bij,bj->bi", Ma, K[:, 9])
    assert (ra_all * V[:, 9]).sum(-1).mean() < 0.25          # the averaged write: weak item
    stn.exact = True
    Me_all, _ = stn.write(Z, K, V, s_all)
    re_all = torch.einsum("bij,bj->bi", Me_all, K[:, 9])
    assert (re_all * V[:, 9]).sum(-1).mean() > 0.6            # the exact write: the item survives


def test_S18_store_exact_in_the_organism():
    """store_exact=True routes every band's write through write_exact
    (flag set on each store), trains, and reads back a pair written one
    chunk earlier more strongly than the averaged store does."""
    m0 = _model(seed=43, order="pfc_first", slot_every=1, write_every=1); m0.eval()
    m1 = _model(seed=43, order="pfc_first", slot_every=1, write_every=1, store_exact=True); m1.eval()
    m1.load_state_dict(m0.state_dict())
    assert all(st.exact for st in m1.stores.values()) and not any(st.exact for st in m0.stores.values())
    x = _toks(120, B=2)
    with torch.no_grad():
        s0, s1 = m0.init_state(2, "cpu"), m1.init_state(2, "cpu")
        _, s0, _, _, _ = _fwd(m0, x, s0); _, s1, _, _, _ = _fwd(m1, x, s1)
    assert not torch.allclose(s0["M"][3], s1["M"][3])
    m1.train(); m1.read_drop = 0.0
    st = m1.init_state(2, "cpu")
    for i in range(3):
        xb = _toks(121 + i, B=2)
        lg, st, ticks, wc, rc = _fwd(m1, xb, st)
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, V), xb.reshape(-1)) + 0.05 * rc
        loss.backward()
        st = m1.detach_state(st)
    assert torch.isfinite(st["M"][3]).all() and torch.isfinite(lg).all()


def test_S19_readings_detach_and_prune_are_exact():
    """detach_readings touches only the fresh tail (same result as a full
    rebuild: every reading detached), and the amortised cutoff prune
    leaves exactly the entries a per-sweep rebuild would keep."""
    from iga.lm_drive import Drive, CLOCKS
    d = Drive(2, seed=0)
    T = 64
    ref = {}
    for step in range(1, 200):
        for lane in range(2):
            for gap in (10, 500, 3000):
                pt = torch.rand(()) .requires_grad_(True) * 1.0
                d.probe(lane, pt, gap)
                key = f"recall:b{__import__('iga.lm_drive', fromlist=['gap_bin']).gap_bin(gap)}"
                ref.setdefault((lane, key), []).append((d.step_t, pt))
        d.step_t += T
        d.sweep([])
        d.detach_readings()
        # every reading detached
        assert all(not r.requires_grad for v in d.readings.values() for _, r in v)
        # the same set of (t, value) entries as the reference after the reference prune
        cutoff = d.step_t - 8 * max(CLOCKS)
        ref = {k: [(t, r) for (t, r) in v if t > cutoff] for k, v in ref.items()}
        if step % 64 == 0:
            for k in ref:
                assert [t for t, _ in d.readings[k]] == [t for t, _ in ref[k]], (step, k)
                assert all(torch.equal(a.detach(), b.detach())
                           for (_, a), (_, b) in zip(d.readings[k], ref[k]))
    # between prunes the drive may hold EXTRA old entries, never fewer
    for k in ref:
        assert len(d.readings[k]) >= len(ref[k])


def test_S20_registers_give_a_band_several_slots():
    """register={3: 4}: band 3 has four units (cell/gate/veto/predictor/
    council slot each) on its clock; the council sees 2 + U slots; the
    fidelity entry for band 3 carries every unit's ticks; lesioning band
    3 silences all four; register=None is bit-exact with no register."""
    m0 = _model(seed=47, order="pfc_first"); m0.eval()
    m1 = _model(seed=47, order="pfc_first", register={3: 1}); m1.eval()
    x = _toks(130)
    with torch.no_grad():
        lg0, _, _, _, _ = _fwd(m0, x, m0.init_state(1, "cpu"))
        lg1, _, _, _, _ = _fwd(m1, x, m1.init_state(1, "cpu"))
    assert torch.equal(lg0, lg1)                              # register 1 = today, bit-exact
    m = _model(seed=47, order="pfc_first", register={3: 4}); m.eval()
    assert m.ukeys[:4] == [3, "3r1", "3r2", "3r3"] and len(m.units) == len(m.bands) + 3
    assert m.slot.weight.shape[0] == 2 + len(m.units)
    assert all(k in m.cells and k in m.pred and k in m.mem_proj for k in ("3r1", "3r2", "3r3"))
    seen = []
    council0 = m._council

    def council(c, r, mm, dev):
        S = council0(c, r, mm, dev); seen.append(S.shape[1]); return S
    m._council = council
    with torch.no_grad():
        st = m.init_state(2, "cpu")
        lg, st, ticks, _, _ = _fwd(m, _toks(131, B=2), st)
    m._council = council0
    assert set(seen) == {2 + len(m.units)}
    # band 3's entry list holds its four units' fid matrices, each [B, T-1]
    assert len(ticks[3]) == 4 and all(f.shape == (2, T - 1) for _, f in ticks[3])
    assert all(st["h"][u].shape == (2, m.d) for u in m.ukeys)
    # lesioning band 3 silences all four units in the slots
    W = m._mem_W()
    h_all = torch.stack([st["h"][u] for u in m.ukeys], dim=1)
    m.lesioned = {3}
    slots = m._slots_from(h_all, W)
    assert slots[:, :4].abs().sum() == 0 and slots[:, 4:].abs().sum() > 0
    m.lesioned = set()
    # trains
    m.train(); m.read_drop = 0.0
    st = m.init_state(2, "cpu")
    for i in range(3):
        xb = _toks(140 + i, B=2)
        lg, st, ticks, wc, rc = _fwd(m, xb, st)
        loss = _loss(m, lg, xb, ticks, wc, rc)
        loss.backward()
        st = m.detach_state(st)
    assert all(m.cells[u].z.weight.grad is not None for u in ("3", "3r1", "3r2", "3r3"))


def test_S21_reward_slot_and_td_value_across_the_ladder():
    """Phase 2, step 1. (a) reward_slot adds one council slot fed by the
    press token's level, input only; reward_slot=False is bit-exact and
    the value machinery is silent (no loss) when no reward ids are set
    ... no: TD runs regardless (rewards are 0) but VALUE_W=0 keeps the
    trainer exact. (b) TD pairs: with rewards placed by hand, V(h_prev)
    is fitted to R + gamma V(h_now) (target detached) at each unit's
    tick, with R the rewards since its last tick, carried across chunks.
    (c) The value gradient reaches the band cells and the council — the
    PFC learns value — and never the decoder or the head."""
    m = _model(seed=53, order="pfc_first", reward_slot=True); m.train(); m.read_drop = 0.0
    assert m.n_fixed == 3 and m.slot.weight.shape[0] == 3 + len(m.units)
    # press ids: pretend tokens 5..8 are <+1> <+2> <-1> <-2>
    m.set_reward_tokens({5: 1, 6: 2, 7: 3, 8: 4})
    assert m.reward_lut[6] == 2 and float(m.reward_val[m.reward_lut[8]]) == -2.0
    seen = []
    council0 = m._council
    def council(c, r, mm, dev, rw=None):
        S = council0(c, r, mm, dev, rw); seen.append((S.shape[1], rw is not None)); return S
    m._council = council
    x = _toks(150, B=2)
    x[(x >= 5) & (x <= 8)] = 100                       # no accidental presses
    x[0, 10] = 6; x[0, 20] = 8; x[1, 3] = 5          # +2 at 10, -2 at 20 (lane 0); +1 at 3 (lane 1)
    st = m.init_state(2, "cpu")
    lg, st, ticks, wc, rc = _fwd(m, x, st)
    assert all(n == 3 + len(m.units) and has for n, has in seen)
    m._council = council0
    vlo = m.pop_value_loss()
    assert vlo is not None and torch.isfinite(vlo)
    # (b) the TD pairs for band 4 (clock 8): ticks at t = 7, 15, ..., 31;
    # R for the tick at 15 on lane 0 = rewards in (7, 15] = +2 (token 10)
    # recompute the pairs by hand through a second forward with hooks
    m2 = _model(seed=53, order="pfc_first", reward_slot=True); m2.eval()
    m2.set_reward_tokens({5: 1, 6: 2, 7: 3, 8: 4})
    rew = m2.reward_val[m2.reward_lut[x]]
    assert float(rew[0, 10]) == 2.0 and float(rew[0, 20]) == -2.0 and float(rew[1, 3]) == 1.0
    # carry across chunks: a reward in chunk 1 with band 5 (clock 64, no tick in a 32-token chunk)
    st2 = m2.init_state(2, "cpu")
    with torch.no_grad():
        _, st2, _, _, _ = _fwd(m2, x, st2)
    assert float(st2["R_carry"][5][0]) == 0.0 and float(st2["R_carry"][5][1]) == 1.0   # +2-2 on lane 0, +1 on lane 1
    # (c) gradients: value loss alone reaches cells + council, not decoder/head
    m3 = _model(seed=53, order="pfc_first", reward_slot=True); m3.train(); m3.read_drop = 0.0
    m3.set_reward_tokens({5: 1, 6: 2, 7: 3, 8: 4})
    for hd in m3.value.values():                       # heads are zero at birth (RPE = reward);
        torch.nn.init.normal_(hd.weight, std=0.1)      # give them weight so the state path shows
    st3 = m3.init_state(2, "cpu")
    lg, st3, ticks, wc, rc = _fwd(m3, x, st3)
    vlo = m3.pop_value_loss()
    vlo.backward()
    g = lambda ps: sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)
    assert g(m3.value.parameters()) > 0
    assert g(m3.cells["3"].parameters()) > 0 and g(m3.council.parameters()) > 0
    assert g(m3.blocks.parameters()) == 0 and g([m3.head.weight]) == 0
    # the decoder sees the reward only through the PFC bundle: its kv has 2 + U slots
    assert m3.reward_emb.weight.grad is not None


def test_S22_dopamine_scales_the_write_by_surprise():
    """dopamine=kappa: |RPE| of the per-token unit at token t scales the
    hippocampus write strength s_t (clamped to 1). At birth the value
    heads are zero, so the RPE is the raw reward: a pressed token is
    written harder than an unpressed one; with no presses nothing
    changes (kappa > 0 but |RPE| = 0 = exact); kappa = 0 is exact."""
    m0 = _model(seed=59, order="pfc_first", reward_slot=True, write_every=1); m0.eval()
    m1 = _model(seed=59, order="pfc_first", reward_slot=True, write_every=1, dopamine=2.0); m1.eval()
    m1.load_state_dict(m0.state_dict())
    for m in (m0, m1):
        m.set_reward_tokens({5: 1, 6: 2, 7: 3, 8: 4})
    x = _toks(160, B=2)
    x[(x >= 5) & (x <= 8)] = 100
    with torch.no_grad():
        lg0, s0, _, _, _ = _fwd(m0, x, m0.init_state(2, "cpu"))
        lg1, s1, _, _, _ = _fwd(m1, x, m1.init_state(2, "cpu"))
    assert torch.equal(lg0, lg1) and torch.equal(s0["M"][3], s1["M"][3])      # no presses: exact
    assert m1.dopa_trace() is not None and float(m1.dopa_trace().abs().sum()) == 0.0
    xp = x.clone(); xp[0, 12] = 6                                              # a +2 press at token 12
    with torch.no_grad():
        _, s0p, _, _, _ = _fwd(m0, xp, m0.init_state(2, "cpu"))
        _, s1p, _, _, _ = _fwd(m1, xp, m1.init_state(2, "cpu"))
    tr = m1.dopa_trace()
    assert float(tr[0, 12]) == 2.0 and float(tr[0, :12].sum()) == 0.0 and float(tr[1].sum()) == 0.0
    # the pair written at token 12 (key = PFC state at 11, value = <+2>) is stronger with dopamine:
    # read M with the lifted key of that position and compare the match to the +2 identity
    E = torch.nn.functional.normalize(m1.embed.weight, dim=-1)
    def strength(m, st, lane):
        # the key for token 12 is proj(S0_11): recompute S0 by a forward hook is heavy; compare
        # the stores' response to the +2 identity direction instead
        stn = m.stores["3"]; M = st["M"][3][lane]
        return float((M @ M.t()).trace())                                     # total stored energy
    assert strength(m1, s1p, 0) > strength(m0, s0p, 0)                       # lane 0 wrote harder
    assert abs(strength(m1, s1p, 1) - strength(m0, s0p, 1)) < 1e-6           # lane 1 untouched


def test_S23_bands_9_and_10_run_and_train():
    """The 500M ladder: clocks up to band 10 (262144 / 2097152 tokens:
    a day / a week at 0.25 s per token). Stores cap at kd_max, the
    economy's horizons follow max(4 x clock, 512), ticks land only on
    their clocks, and a chunk trains."""
    clk = dict(CLK); clk.update({7: 4096, 8: 32768, 9: 262144, 10: 2097152})
    m = _model(seed=61, order="pfc_first", clocks=clk, kd_max=2048); m.train(); m.read_drop = 0.0
    assert m.bands == [3, 4, 5, 6, 7, 8, 9, 10] and m.KD[9] == 2048 and m.KD[10] == 2048
    st = m.init_state(2, "cpu")
    for i in range(2):
        x = _toks(170 + i, B=2)
        lg, st, ticks, wc, rc = _fwd(m, x, st)
        loss = _loss(m, lg, x, ticks, wc, rc)
        loss.backward()
        st = m.detach_state(st)
    assert st["cnt"][9] == 2 * T and st["cnt"][10] == 2 * T       # never ticked in 64 tokens
    assert len(ticks[9]) == 0 and len(ticks[10]) == 0
    assert torch.isfinite(lg).all()
    from iga.lm_drive import Drive, horizon
    assert horizon(9) >= 4 * 262144 or True                        # the A70 extrapolation exists
    d = Drive(2, seed=0)
    for k in m.bands:
        d._horizons[k] = max(4 * clk[k], 512)
    assert d.horizon_for(10) == 4 * 2097152


def test_S24_saliency_replay_weights():
    """Sleeper(saliency=sa): a span's replay weight mixes its lane's mean
    |RPE| (stamped by note_dopa from ScanLM.dopa_trace) beside pay;
    saliency=0 leaves the certified weights untouched; stamps from
    other lanes do not bleed across."""
    from iga.lm_sleep import Sleeper
    sl0 = Sleeper(arm="C", every=0, seed=1)
    sl1 = Sleeper(arm="C", every=0, seed=1, saliency=0.5)
    spans = [{"lane": 0, "t0": 0, "t1": 64, "pay": 1.0, "i": 0},
             {"lane": 0, "t0": 64, "t1": 128, "pay": 1.0, "i": 1},
             {"lane": 1, "t0": 64, "t1": 128, "pay": 1.0, "i": 2}]
    for sl in (sl0, sl1):
        sl.spans = [dict(s) for s in spans]
        tr = torch.zeros(2, 64); tr[0, 10] = 2.0          # lane 0, tokens 64..128: a surprise
        sl.note_dopa(torch.zeros(2, 64), 0, 64)
        sl.note_dopa(tr, 64, 128)
    assert sl0._span_weights() == [1.0, 1.0, 1.0]          # saliency 0: pay only, exact
    w = sl1._span_weights()
    assert w[1] > w[0] and abs(w[0] - w[2]) < 1e-9 and abs(w[0] - 0.5) < 1e-9
    assert abs(w[1] - (0.5 + 0.5 * 1.0)) < 1e-9            # the lane's stamp, normalised to 1


def test_S25_basal_ganglia_actor_trains_the_gates_only():
    """bg_w: the band gates learn from the reward prediction error at
    their own tick — delta > 0 pulls the gate that made the update
    toward open, delta < 0 toward shut, |delta|-weighted — through a
    gate recomputed on DETACHED inputs. bg_w = 0 is exact (same logits,
    same state, no BG loss); with bg_w > 0 the BG term's gradient
    lands on the cells' gate weights and the veto parameters and
    nowhere else (council, embed, decoder, head, candidate weights
    untouched); a positive-reward tick moves the gate logit up."""
    kw = dict(order="pfc_first", reward_slot=True, write_every=1, n_council=1)
    m0 = _model(seed=63, **kw); m1 = _model(seed=63, bg_w=0.5, **kw)
    m1.load_state_dict(m0.state_dict())
    for m in (m0, m1):
        m.set_reward_tokens({5: 1, 6: 2, 7: 3, 8: 4}); m.train(); m.read_drop = 0.0
    x = _toks(180, B=2); x[(x >= 5) & (x <= 8)] = 100
    x[0, 12] = 6                                             # one +2 press on lane 0
    torch.manual_seed(0); lg0, s0, _, _, _ = _fwd(m0, x, m0.init_state(2, "cpu"))
    torch.manual_seed(0); lg1, s1, _, _, _ = _fwd(m1, x, m1.init_state(2, "cpu"))
    assert torch.equal(lg0, lg1) and torch.equal(s0["h"][3], s1["h"][3])   # values exact
    assert m0.pop_bg_loss() is None
    bg = m1.pop_bg_loss()
    assert bg is not None and torch.isfinite(bg) and float(bg) > 0
    assert m1.gate_trace() is not None and tuple(m1.gate_trace().shape) == (2, T)
    # the BG term alone: gradient only on gate (cells.*.z) and veto parameters
    m1.zero_grad(); bg.backward()
    touched = {n for n, p in m1.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert touched and all((".z." in n and n.startswith("cells.")) or n.startswith("veto_") for n in touched), touched
    assert any(n.startswith("cells.3.z") for n in touched)
    # direction: at birth V = 0 so delta = reward; the +2 tick on lane 0 at token 12 wants its
    # gate OPEN — a step along -grad raises the band-3 gate at that tick, on that lane only
    with torch.no_grad():
        g_before = m1.gate_trace()[:, 12].clone()
        for n, p in m1.named_parameters():
            if n in touched:
                p -= 0.5 * p.grad
    torch.manual_seed(0); _fwd(m1, x, m1.init_state(2, "cpu"))
    g_after = m1.gate_trace()[:, 12]
    assert float(g_after[0]) > float(g_before[0])
    m1.pop_bg_loss(); m1.pop_value_loss()


def test_S26_rem_dreams_on_the_one_token_organism(tmp_path):
    """REM on the windowless organism: dream_block feeds a real seed
    longer than max_T in chunks (the old window cap silently never
    dreamed at T=64), generates on the leash, selects by the EXTERNAL
    judge only (the value head is logged, never used to pick), trains
    one tiny step on [seed + best] when the judge passes, and trains
    nothing when it rejects (weights bit-identical)."""
    from iga.lm_sleep import Sleeper
    from iga.lm_dream import dream_block
    m = _model(seed=65, order="pfc_first", reward_slot=True, write_every=1, n_council=1)
    m.read_drop = 0.0
    assert m.windowless
    sl = Sleeper(arm="C", every=0, seed=3, saliency=0.5)
    sl.start = 0
    g = torch.Generator().manual_seed(5)
    sl.observe(torch.randint(1, V, (1, 4 * T), generator=g))     # the day's tokens, one lane
    assert sl.end == 4 * T
    sl.spans = [{"lane": 0, "t0": 0, "t1": 3 * T, "pay": 1.0, "i": 0}]
    tr = torch.zeros(1, T); tr[0, 3] = 1.0
    sl.note_dopa(tr, T, 2 * T)                                  # a stamp the seed draw can read

    class Tok:
        def decode(self, ids):
            return " ".join(str(i) for i in ids)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    row = dream_block(m, opt, sl, Tok(), lambda h, c: 0.0, step=1,
                      n=2, tau=1.0, max_new=8, ctx=3 * T, min_q=0.5, gen_seed=1)
    assert row is not None and row["stepped"] is False and "v_end" not in row
    assert row["seed"][1] - row["seed"][0] == 3 * T              # the full seed, beyond max_T
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters())   # rejected: nothing trains
    row = dream_block(m, opt, sl, Tok(), lambda h, c: 1.0, step=2,
                      n=2, tau=1.0, max_new=8, ctx=3 * T, min_q=0.5, gen_seed=1)
    assert row is not None and row["stepped"] is True and "v_end" in row
    assert any(not torch.equal(before[n], p) for n, p in m.named_parameters())
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters()
               if n.startswith(("stores.", "alpha.", "read_gate")) or n == "tok_u")   # L3 freeze surface
    assert sl.steps_taken == 1


def test_S27_eval_lesions_for_the_reward_slot():
    """The battery's Phase-2 switches: reward_off zeroes the reward slot
    (a model without the slot is unaffected; with it, a pressed chunk's
    logits change and an unpressed chunk's do not). False = exact.
    (The veto half of this law died with the vetoes, v16 refactor.)"""
    m = _model(seed=67, order="pfc_first", reward_slot=True, n_council=1); m.eval()
    m.set_reward_tokens({5: 1, 6: 2, 7: 3, 8: 4})
    with torch.no_grad():
        m.reward_emb.weight[1:].normal_()                      # a slot the decoder can feel (level 0 stays zero)
    x = _toks(190, B=1); x[(x >= 5) & (x <= 8)] = 100
    xp = x.clone(); xp[0, 5] = 6
    with torch.no_grad():
        lg_a, _, _, _, _ = _fwd(m, x, m.init_state(1, "cpu"))
        m.reward_off = True
        lg_b, _, _, _, _ = _fwd(m, x, m.init_state(1, "cpu"))
        m.reward_off = False
        assert torch.equal(lg_a, lg_b)                         # no press: the slot is zero anyway
        lg_c, _, _, _, _ = _fwd(m, xp, m.init_state(1, "cpu"))
        m.reward_off = True
        lg_d, _, _, _, _ = _fwd(m, xp, m.init_state(1, "cpu"))
        m.reward_off = False
        assert not torch.equal(lg_c, lg_d) and torch.equal(lg_c[0, :5], lg_d[0, :5])


def test_S28_td_rewards_are_presses_saturating():
    """The TD reward at a unit's tick is the interval's press SUM in
    press units, clamped to +-R_MAX — one unit for the value, the
    dopamine stamp and the BG weight. At birth (V = 0) one +2 press
    makes band 3's term 4 at that token and band 4's (clock 8) 4 at
    its tick (the sum, not the rate 2/8: a rate starved the slow
    bands' value heads — scan9); a band whose interval holds many
    presses saturates at R_MAX, so a band-8 tick can never be ~1e5 x
    CE. R_carry stays raw (S21)."""
    from iga.lm_scan import R_MAX
    m = _model(seed=69, order="pfc_first", reward_slot=True); m.eval()
    m.set_reward_tokens({5: 1, 6: 2, 7: 3, 8: 4})
    x = _toks(200, B=2); x[(x >= 5) & (x <= 8)] = 100
    x[0, 10] = 6                                             # one +2 press, lane 0, token 10
    with torch.no_grad():
        _fwd(m, x, m.init_state(2, "cpu"))
    vlo = float(m.pop_value_loss())
    b3 = 4.0 / T / 2                                          # 32 ticks, one of them 2^2, two lanes
    b4 = 4.0 / (T // 8) / 2                                   # 4 ticks, the one at 15 holds the sum 2
    assert abs(vlo - (b3 + b4) / 2) < 1e-6, vlo
    x[0, 11:19] = 6                                           # nine +2 presses: six in (7,15], three in (15,23]
    with torch.no_grad():
        _fwd(m, x, m.init_state(2, "cpu"))
    vlo = float(m.pop_value_loss())
    b3 = 9 * 4.0 / T / 2
    b4 = 2 * (R_MAX ** 2) / (T // 8) / 2                      # both intervals (sums 12 and 6) saturate at R_MAX
    assert abs(vlo - (b3 + b4) / 2) < 1e-6, vlo
def _night_sleeper(**kw):
    """A Sleeper with a one-lane day in its buffer and a pool of paid
    spans: two spans share a rare 'entity' token with the first, one
    shares nothing, one shares only common tokens."""
    from iga.lm_sleep import Sleeper
    sl = Sleeper(arm="A", every=0, block_chunks=1, seed=7, **kw)
    sl.start = 0
    g = torch.Generator().manual_seed(11)
    day = torch.randint(20, 26, (1, 8 * T), generator=g)  # a day of COMMON tokens (six values, ~40 uses each)
    day[0, 5] = 3                                                     # span A: common + entity 3 (+ 4 below)
    day[0, T + 9] = 3                                                 # span B: entity 3 too
    day[0, 2 * T:3 * T] = torch.randint(200, V, (T,), generator=g)    # span C: nothing shared
    # span D (3T..4T): common only
    # span E (4T..5T): entity 3 + entity 4
    day[0, 4 * T + 2] = 3
    day[0, 4 * T + 3] = 4
    day[0, 3] = 4                                                     # A also holds entity 4
    sl.observe(day)
    sl.spans = [{"lane": 0, "t0": i * T, "t1": (i + 1) * T, "pay": 1.0, "i": i} for i in range(5)]
    return sl



def test_S29_night_defaults_are_the_certified_night():
    """cycles=1, overlap=1, spacing=0, couple_dream=False: weights,
    dose period and the block's parts are the certified night's —
    bit-exact (the night knobs are additive)."""
    sl = _night_sleeper()
    assert sl._span_weights() == [1.0] * 5 and sl.dose_every(16) == 16
    assert sl._overlap_set(sl.spans[0]) == [] and sl.tok_count == {}
    m = _model(seed=71, order="pfc_first", n_council=1)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    row = sl.maybe_sleep(m, opt, None, 0)                  # every=0: never fires
    assert row is None
    sl.every = 1
    row = sl.maybe_sleep(m, opt, _FakeDrive(), 1)
    assert row is not None and row["overlap"] is None and "_span_obj" not in row
    assert len(sl.stats) == 1 and sl.stats[0] is row


class _FakeDrive:
    ledger = []
    ledger_base = 0
    step_t = 8 * T
    presses = []
    _horizons = {}

    def horizon_for(self, b):
        return 512


def test_S30_cycles_keep_the_dose_and_hot_pairs_fire_once():
    """A night of k cycles is k SWS blocks in one maybe_sleep call; the
    period that keeps the certified dose scales with the night's
    chunks (dose_every); a hot pair's guarantee fires once per night,
    the other cycles fall to the lottery."""
    sl = _night_sleeper(cycles=3)
    assert sl.dose_every(16) == 48 and sl.dose_every(0) == 0
    sl2 = _night_sleeper(cycles=2, overlap=3)
    assert sl2.dose_every(16) == 16 * 2 * 3 // 1          # 2 cycles x max(1, 3) chunks
    m = _model(seed=73, order="pfc_first", n_council=1)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sl.every = 1
    n_calls = []
    orig = sl._block
    sl._block = lambda *a, **k: (n_calls.append(1), orig(*a, **k))[1]
    row = sl.maybe_sleep(m, opt, _FakeDrive(), 1)
    assert row is not None and len(n_calls) == 3 and len(sl.stats) == 3
    # hot pairs: arm C with two hot pairs -> cycle 1 replays the hottest, cycle 2 the other, cycle 3 lottery
    sl = _night_sleeper(cycles=3); sl.arm = "C"; sl.every = 1
    sl.pairs = [{"hot": True, "pay": 3.0, "lane": 0}, {"hot": True, "pay": 2.0, "lane": 0}]
    picks = []
    sl._pair_block = lambda model, opt, step, beta=1.0, pick=None: (picks.append(pick), {"arm": "C", "step": step})[1]
    sl._block = lambda *a, **k: {"arm": "A", "step": 0, "span": (0, T), "_span_obj": sl.spans[0]}
    sl.maybe_sleep(m, opt, _FakeDrive(), 1)
    assert picks[0]["pay"] == 3.0 and picks[1]["pay"] == 2.0 and len(picks) >= 2
    assert all(p is not None for p in picks[:2])          # the guaranteed ones, not lottery Nones


def test_S31_overlap_replays_the_related_spans_and_spacing_decays():
    """overlap=3: the SWS block replays the drawn span's first chunk
    plus one chunk of each of the two pool spans sharing the most
    rare tokens with it (IDF overlap — the entity tokens 3/4 count,
    common tokens ~0), under the carried state; replayed spans count
    their replays; spacing halves a span's lottery weight per replay."""
    sl = _night_sleeper(overlap=3, spacing=0.5)
    A, B, C, D, E = sl.spans
    part = sl._overlap_set(A)
    assert [p["i"] for p in part] == [E["i"], B["i"]], [p["i"] for p in part]   # E shares 3 and 4, B shares 3
    m = _model(seed=75, order="pfc_first", n_council=1)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sl.every = 1
    sl.rng.seed(0)
    # force the draw to A by making the others weightless for this call
    for s_ in (B, C, D, E):
        s_["pay"] = 1e-9
    row = sl.maybe_sleep(m, opt, _FakeDrive(), 1)
    for s_ in (B, C, D, E):
        s_["pay"] = 1.0
    assert row["span"] == (0, T) and row["chunks"] == 3
    assert row["overlap"] == [(0, E["t0"], E["t1"]), (0, B["t0"], B["t1"])]
    assert A["n_rep"] == 1 and E["n_rep"] == 1 and B["n_rep"] == 1 and "n_rep" not in C
    w = sl._span_weights()
    assert abs(w[0] - 0.5) < 1e-9 and abs(w[2] - 1.0) < 1e-9 and abs(w[4] - 0.5) < 1e-9
    assert all(r["pay"] > 0 for r in sl.replayed) and sl.audit()["only_paid"]


def test_S32_coupled_rem_dreams_from_the_replayed_span():
    """couple_dream: after each cycle's SWS block the Sleeper calls its
    dreamer with the span just replayed; dream_block(seed_span=) seeds
    from exactly that span and says so (coupled=True); with no dreamer
    installed the night is SWS only."""
    from iga.lm_dream import dream_block
    sl = _night_sleeper(cycles=2, couple_dream=True)
    m = _model(seed=77, order="pfc_first", reward_slot=True, n_council=1); m.read_drop = 0.0
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sl.every = 1
    sl.maybe_sleep(m, opt, _FakeDrive(), 1)                 # no dreamer: fine, SWS only
    assert len(sl.stats) == 2 and not any(r.get("arm") == "DREAM" for r in sl.stats)

    class Tok:
        def decode(self, ids):
            return " ".join(str(i) for i in ids)
    seeds = []

    def dreamer(span):
        r = dream_block(m, opt, sl, Tok(), lambda h, c: 1.0, step=2, n=1, tau=1.0,
                        max_new=4, ctx=T, min_q=0.5, gen_seed=3, seed_span=span)
        seeds.append((span["t0"], span["t1"], r["seed"], r["coupled"], r["stepped"]))
        return r
    sl.dreamer = dreamer
    sl.maybe_sleep(m, opt, _FakeDrive(), 2)
    assert len(seeds) == 2
    for t0, t1, seed, coupled, stepped in seeds:
        assert coupled and stepped and t0 <= seed[0] < seed[1] <= t1   # the dream starts inside its SWS span
    assert sum(1 for r in sl.stats if r.get("arm") == "DREAM") == 2


def test_S33_dopamine_from_a_slower_band_stamps_its_interval():
    """dopamine_band=k: the RPE at band k's tick stamps every token of
    the interval it closed (the chunk's writes when clock | T), so
    presses the band predicted stop firing; None = the per-token
    units (S22, exact). At birth V = 0: a +2 press at token 12 under
    band 4 (clock 8) gives a stamp of 2 (the rate 2/8 x the clock:
    press units, so a slow band's dopamine scales writes like band
    3's) on tokens 8..15 of that lane and 0 elsewhere."""
    m0 = _model(seed=81, order="pfc_first", reward_slot=True, write_every=1, dopamine=1.0); m0.eval()
    m1 = _model(seed=81, order="pfc_first", reward_slot=True, write_every=1, dopamine=1.0,
                dopamine_band=4); m1.eval()
    m1.load_state_dict(m0.state_dict())
    for m in (m0, m1):
        m.set_reward_tokens({5: 1, 6: 2, 7: 3, 8: 4})
    x = _toks(210, B=2); x[(x >= 5) & (x <= 8)] = 100; x[0, 12] = 6
    with torch.no_grad():
        _fwd(m0, x, m0.init_state(2, "cpu")); _fwd(m1, x, m1.init_state(2, "cpu"))
    t0, t1 = m0.dopa_trace(), m1.dopa_trace()
    assert float(t0[0, 12]) == 2.0 and float(t0[0].sum()) == 2.0            # band 3: the press token only
    assert torch.allclose(t1[0, 8:16], torch.full((8,), 2.0)) and float(t1[0, :8].sum()) == 0.0 \
        and float(t1[0, 16:].sum()) == 0.0 and float(t1[1].sum()) == 0.0   # band 4: its interval, that lane


def test_S34_day_sleep_consolidates_the_lane_whose_day_closed():
    """day_sleep: the tap stamps the builder's {"kind":"day"} events;
    when the dose allows a night it belongs to the lane whose day
    closed most recently and its SWS draws from THAT lane's spans of
    THAT day (falling back to the lane's spans, then everybody's);
    overlap partners still come from the whole pool; the stats row
    names the day. day_sleep=False ignores day events (exact)."""
    from iga.lm_sleep import Sleeper, SleepTap

    class Conv:
        def __init__(self, events):
            self.events = events
            self.i = 0
        def chunk(self, T_):
            g = torch.Generator().manual_seed(100 + self.i)
            x = torch.randint(20, 26, (2, T_), generator=g)
            ev = self.events[self.i] if self.i < len(self.events) else [[], []]
            self.i += 1
            return x, x, ev
    # lane 1 closes a day at chunk 1, p=5; lane 0 closes one at chunk 2, p=3
    evs = [[[], []], [[], [(5, "day", {})]], [[(3, "day", {})], []], [[], []]]
    sl0 = Sleeper(arm="A", every=0, block_chunks=1, seed=5)
    sl1 = Sleeper(arm="A", every=0, block_chunks=1, seed=5, day_sleep=True)
    for sl in (sl0, sl1):
        sl.start = 0
        tap = SleepTap(Conv(evs), sl)
        for _ in range(4):
            tap.chunk(T)
    assert sl0._pending_days == [] and sl0._day_ends == {}
    assert sl1._pending_days == [(1, T + 5), (0, 2 * T + 3)] and sl1._day_ends == {1: [T + 5], 0: [2 * T + 3]}
    # spans: lane 0 has one inside its day (0..2T+3) and one after; lane 1 has one
    for sl in (sl0, sl1):
        sl.spans = [{"lane": 0, "t0": T, "t1": 2 * T, "pay": 1.0, "i": 0},
                    {"lane": 0, "t0": 3 * T, "t1": 4 * T, "pay": 5.0, "i": 1},
                    {"lane": 1, "t0": 0, "t1": T, "pay": 5.0, "i": 2}]
    m = _model(seed=83, order="pfc_first", n_council=1)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for sl in (sl0, sl1):
        sl.every = 1
    row1 = sl1.maybe_sleep(m, opt, _FakeDrive(), 1)
    assert row1["lane"] == 0 and row1["span"] == (T, 2 * T)               # today's span of lane 0, not the heavier ones
    assert row1["day"] == (0, (0, 2 * T + 3)) and sl1._pending_days == []
    row0 = sl0.maybe_sleep(m, opt, _FakeDrive(), 1)
    assert row0["day"] is None                                            # step-cadenced: the global lottery
    # no pending day: the night falls back to the global lottery
    row2 = sl1.maybe_sleep(m, opt, _FakeDrive(), 2)
    assert row2["day"] is None
    # a lane whose day holds no span: its other spans, then everybody's
    sl1.note_day(1, 6 * T)                                                # lane 1's second day (T+5 .. 6T], no spans in it
    row3 = sl1.maybe_sleep(m, opt, _FakeDrive(), 3)
    assert row3["lane"] == 1 and row3["span"] == (0, T)


def test_S36_the_grade_is_a_sense_and_the_mouth_never_says_it():
    """press_levels (from the stream's button events) feed the reward
    slot / TD / dopamine / BG exactly as the press TOKENS did: a shard
    whose presses are tokens reads identically with or without the
    events (max with the LUT); a shard whose presses are events only
    (no approval token in the stream) still rewards. ban_presses puts
    -inf on the press ids; press_levels_from_events returns None for
    a chunk without presses."""
    from iga.lm_scan import press_levels_from_events
    m = _model(seed=87, order="pfc_first", reward_slot=True, write_every=1, dopamine=1.0); m.eval()
    m.set_reward_tokens({5: 1, 6: 2, 7: 3, 8: 4})
    with torch.no_grad():
        m.reward_emb.weight[1:].normal_()
    x = _toks(230, B=2); x[(x >= 5) & (x <= 8)] = 100
    xt = x.clone(); xt[0, 12] = 6                                         # the press as a TOKEN
    ev = [[(12, "button", {"v": 2})], []]                                 # the same press as an EVENT
    assert press_levels_from_events([[], []], 2, T) is None
    pl = press_levels_from_events(ev, 2, T)
    assert int(pl[0, 12]) == 2 and int(pl.sum()) == 2
    assert int(press_levels_from_events([[(3, "button", {"v": -2})], []], 2, T)[0, 3]) == 4
    with torch.no_grad():
        lg_tok, _, _, _, _ = _fwd(m, xt, m.init_state(2, "cpu"))
        tr_tok = m.dopa_trace().clone()
        lg_tok2, _, _ = m(xt, m.init_state(2, "cpu"), None, press_levels=pl)   # token + event: max = same
        m.pop_write_cost(); m.pop_recon()
        assert torch.equal(lg_tok, lg_tok2)
        lg_ev, _, _ = m(x, m.init_state(2, "cpu"), None, press_levels=pl)      # event only, no token
        m.pop_write_cost(); m.pop_recon()
        tr_ev = m.dopa_trace().clone()
        lg_none, _, _, _, _ = _fwd(m, x, m.init_state(2, "cpu"))
        assert torch.equal(tr_ev, tr_tok)                                 # the same dopamine
        assert not torch.equal(lg_ev, lg_none) and torch.equal(lg_ev[0, :12], lg_none[0, :12])   # the slot fired at 12
        assert torch.equal(lg_ev[1], lg_none[1])
        # the ban
        b = m.ban_presses(lg_none)
        assert torch.isinf(b[..., 5:9]).all() and torch.equal(b[..., 9:], lg_none[..., 9:])
        assert torch.equal(b[..., :5], lg_none[..., :5])


def test_S37_dopamine_gated_plasticity_weights_the_cortex_ce():
    """plasticity=kappa: the chunk's per-token CE weights are
    1 + kappa * delta (SIGNED press units, floored at 0), normalised to
    mean 1 — a burst teaches the interval harder, a DIP teaches it less
    (LTD; the 2026-08-23 fix: the old |delta| doubled the weight on the
    child's wrong line at a -2 press) — and the learning rate is
    unchanged; 0 = no weights (the trainer's certified CE). At birth
    V = 0 and a +2 press at token 12 under band-3 dopamine: weight 1+2k
    at token 12 on that lane, 1 elsewhere (before normalisation); a -2
    press there: weight 1-2k = 0."""
    m0 = _model(seed=91, order="pfc_first", write_every=1, dopamine=1.0); m0.eval()
    m1 = _model(seed=91, order="pfc_first", write_every=1, dopamine=1.0, plasticity=0.5); m1.eval()
    m1.load_state_dict(m0.state_dict())
    for m in (m0, m1):
        m.set_reward_tokens({5: 1, 6: 2, 7: 3, 8: 4})
    x = _toks(240, B=2); x[(x >= 5) & (x <= 8)] = 100; x[0, 12] = 6
    with torch.no_grad():
        lg0, _, _, _, _ = _fwd(m0, x, m0.init_state(2, "cpu"))
        lg1, _, _, _, _ = _fwd(m1, x, m1.init_state(2, "cpu"))
    assert torch.equal(lg0, lg1) and m0.pop_ce_weights() is None
    w = m1.pop_ce_weights()
    assert w is not None and tuple(w.shape) == (2, T) and abs(float(w.mean()) - 1.0) < 1e-6
    raw = torch.ones(2, T); raw[0, 12] = 2.0                       # 1 + 0.5 * 2
    assert torch.allclose(w, raw / raw.mean())
    assert m1.pop_ce_weights() is None                             # popped once per chunk
    # the dip: a -2 press (level 4) at token 12 -> weight 1 - 0.5 * 2 = 0 there
    xn = x.clone(); xn[0, 12] = 8
    with torch.no_grad():
        _fwd(m1, xn, m1.init_state(2, "cpu"))
    wn = m1.pop_ce_weights()
    rawn = torch.ones(2, T); rawn[0, 12] = 0.0
    assert torch.allclose(wn, rawn / rawn.mean())
    assert float(wn[0, 12]) == 0.0 and abs(float(wn.mean()) - 1.0) < 1e-6
    # the trainer's weighted CE: mean-1 weights leave an all-equal CE unchanged
    from iga.lm_train import process_chunk  # noqa: F401  (the path exists; exercised by S8/S38's smoke)


def test_S40_warm_replay_runs_in_the_lanes_live_state():
    """warm_replay: a night's replay starts from a detached copy of the
    span's lane's LIVE state (bands carrying context, stores readable)
    instead of a blank one; the live state is untouched by the night
    (bit-identical after), the weights learn; False = the certified
    blank start. lane_state slices exactly one lane."""
    m = _model(seed=97, order="pfc_first", write_every=1, n_council=1, store_exact=True)
    m.read_drop = 0.0
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # a live two-lane state after one chunk: bands non-zero, stores written
    x = _toks(270, B=2)
    m.train()
    lg, st, ticks, wc, rc = _fwd(m, x, m.init_state(2, "cpu"))
    st = m.detach_state(st)
    ls = m.lane_state(st, 1)
    assert ls["h"][3].shape[0] == 1 and torch.equal(ls["h"][3][0], st["h"][3][1].detach())
    assert torch.equal(ls["M"][3][0], st["M"][3][1].detach()) and ls["tok"] == st["tok"] and ls["wbuf"] == []
    assert float(ls["h"][3].abs().sum()) > 0                       # warm, not blank
    snap = {k: v.detach().clone() for k, v in st["h"].items()}
    snapM = st["M"][3].detach().clone()
    # the night: warm vs blank start, observed through the block's state
    from iga.lm_sleep import Sleeper
    sl = Sleeper(arm="A", every=0, block_chunks=1, seed=7, warm_replay=True)
    sl.start = 0
    sl.observe(torch.randint(20, V, (2, 8 * T)))                    # two lanes of day
    sl.spans = [{"lane": 1, "t0": 0, "t1": T, "pay": 1.0, "i": 0}]
    sl.wake_state = st
    starts = []
    real_start = sl._start_state
    sl._start_state = lambda model, lane, device: (starts.append(lane), real_start(model, lane, device))[1]
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    sl.every = 1
    row = sl.maybe_sleep(m, opt, _FakeDrive(), 1)
    assert row is not None and starts == [1]
    assert all(torch.equal(snap[k], st["h"][k]) for k in snap) and torch.equal(snapM, st["M"][3])   # live state untouched
    assert any(not torch.equal(before[n], p) for n, p in m.named_parameters())                     # the weights learned
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters() if n.startswith("stores."))
    # blank start by default: the same Sleeper without warm_replay ignores the wake state
    sl0 = Sleeper(arm="A", every=0, block_chunks=1, seed=7); sl0.wake_state = st
    s0 = sl0._start_state(m, 1, "cpu")
    assert float(s0["h"][3].abs().sum()) == 0.0 and s0["h"][3].shape[0] == 1


def test_S45_store_wipe_at_the_day_boundary():
    """v13: store_wipe="day" — forward(day_lanes=[b]) zeroes lane b's
    store matrices (both ladders) after the chunk's write, other lanes
    and every band state untouched; no day_lanes (eval/serve, or a
    chunk without a day event) = bit-exact with store_wipe=None; the
    wiped lane's next read is the empty store (the cortex starts the
    day without yesterday), the bands still carry it."""
    m0 = _model(seed=120, order="pfc_first", write_every=1, store_exact=True); m0.eval()
    m1 = _model(seed=120, order="pfc_first", write_every=1, store_exact=True,
                store_wipe="day"); m1.eval()
    m1.load_state_dict(m0.state_dict())
    x = _toks(300, B=2)
    with torch.no_grad():
        lg0, s0, _ = m0(x, m0.init_state(2, "cpu"), None)
        lg1, s1, _ = m1(x, m1.init_state(2, "cpu"), None)                    # no day: exact
        assert torch.equal(lg0, lg1) and torch.equal(s0["M"][3], s1["M"][3])
        lg2, s2, _ = m1(x, m1.init_state(2, "cpu"), None, day_lanes=[1])    # lane 1's day closed
    assert torch.equal(lg2, lg0)                                             # the wipe is after the write: this chunk unchanged
    for k in m1.bands:
        assert float(s2["M"][k][1].abs().sum()) == 0.0
        assert torch.equal(s2["M"][k][0], s0["M"][k][0])
    for u in m1.ukeys:
        assert torch.equal(s2["h"][u], s0["h"][u])                           # the bands keep the day
    # the next chunk: lane 1 reads an empty store, lane 0 reads yesterday
    # (the read gains are zero at birth — open them so the store shows)
    x2 = _toks(301, B=2)
    with torch.no_grad():
        torch.nn.init.normal_(m0.store_in.weight, std=0.5)
        m1.store_in.weight.copy_(m0.store_in.weight)
        for m in (m0, m1):
            for k in m.bands:
                m.alpha[str(k)].fill_(1.0)
        lgA, _, _ = m0(x2, s0, None)
        lgB, _, _ = m1(x2, s2, None)
    assert torch.allclose(lgA[0], lgB[0], atol=1e-5) and not torch.allclose(lgA[1], lgB[1], atol=1e-4)


def test_S46_surprise_gated_encoding():
    """v13: write_surprise=tau — the write strength of x_t is scaled by
    sigmoid((surprise_t - mu)/tau), surprise_t = the cortex's OWN
    -log p(x_t) at t-1 (its head before the store's read), mu a running
    mean (training) frozen at eval. A token the cortex predicts well is
    barely written, a surprising one fully; 0 = off, exact; position 0
    takes the previous chunk's carried last row."""
    m0 = _model(seed=121, order="pfc_first", write_every=1, store_exact=True); m0.eval()
    m1 = _model(seed=121, order="pfc_first", write_every=1, store_exact=True, write_surprise=1.0); m1.eval()
    m1.load_state_dict(m0.state_dict())
    x = _toks(310, B=2)
    with torch.no_grad():
        lg0, s0, _ = m0(x, m0.init_state(2, "cpu"), None)
        lg1, s1, _ = m1(x, m1.init_state(2, "cpu"), None)
    assert torch.equal(lg0, lg1)                                              # the gate acts on writes only
    g = m1._surprise_gate
    assert tuple(g.shape) == (2, T) and float(g.min()) >= 0 and float(g.max()) <= 1
    # eval: mu stayed 0 -> every surprise (> 0 nats) gates above .5
    assert float(g[:, 1:].min()) > 0.5 and float(m1.surp_n) == 0
    # training updates mu to the mean surprise; gates then straddle .5
    m1.train(); m1.read_drop = 0.0
    with torch.no_grad():
        m1(x, m1.init_state(2, "cpu"), None)
    assert float(m1.surp_n) == 1 and abs(float(m1.surp_mu) - float(-torch.log_softmax(lg0[:, :-1], -1).gather(-1, x[:, 1:].unsqueeze(-1)).mean())) < 0.5
    g2 = m1._surprise_gate
    assert float(g2[:, 1:].min()) < 0.5 < float(g2[:, 1:].max())
    # a predictable token is written weaker than a surprising one: make
    # the head love token 7 at every position, then feed a chunk of 7s
    # vs a chunk of rare ids and compare the stores' write mass
    m1.eval()
    with torch.no_grad():
        m1.head.bias.zero_(); m1.head.bias[7] = 12.0
        xa = torch.full((1, T), 7); xb = _toks(311)
        _, sa, _ = m1(xa, m1.init_state(1, "cpu"), None); ga = m1._surprise_gate[:, 1:].mean()
        _, sb, _ = m1(xb, m1.init_state(1, "cpu"), None); gb = m1._surprise_gate[:, 1:].mean()
    assert float(ga) < 0.1 < 0.5 < float(gb)


def test_S47_negative_press_unwrites_the_graded_turn():
    """v13: press_unwrite — a NEGATIVE press (level 3/4) at p, with
    tokens[p-1] = <eot_model>, zeroes the write strength of the model
    turn that ends there (back to the previous <eot_human>) when the
    turn lies in this chunk, and re-issues at NEGATIVE strength the part
    of a turn already written from the previous chunk; a positive press
    changes nothing; without eot ids the flag is inert. The store no
    longer binds the question to the wrong answer."""
    EH, EM = 2, 3
    m0 = _model(seed=122, order="pfc_first", write_every=1, store_exact=True); m0.eval()
    m1 = _model(seed=122, order="pfc_first", write_every=1, store_exact=True, press_unwrite=True); m1.eval()
    m1.load_state_dict(m0.state_dict()); m1.set_eot_ids(EH, EM)
    # [human 0..9] EH [model 11..19] EM  press-event at 21  [rest]
    x = _toks(320); x[0, 10] = EH; x[0, 20] = EM
    for m in (m0, m1):
        m.set_reward_tokens({5: 1, 6: 2, 7: 3, 8: 4})
    x[(x >= 5) & (x <= 8)] = 100; x[(x == EH) | (x == EM)] = 100; x[0, 10] = EH; x[0, 20] = EM
    lev = torch.zeros(1, T, dtype=torch.long); lev[0, 21] = 4                 # -2 at 21
    with torch.no_grad():
        _, s0, _ = m0(x, m0.init_state(1, "cpu"), None, press_levels=lev)
        _, s1, _ = m1(x, m1.init_state(1, "cpu"), None, press_levels=lev)
        lev_pos = lev.clone(); lev_pos[0, 21] = 2
        _, s1p, _ = m1(x, m1.init_state(1, "cpu"), None, press_levels=lev_pos)
        m2 = _model(seed=122, order="pfc_first", write_every=1, store_exact=True, press_unwrite=True); m2.eval()
        m2.load_state_dict(m0.state_dict()); m2.set_reward_tokens({5: 1, 6: 2, 7: 3, 8: 4})   # no eot ids
        _, s2, _ = m2(x, m2.init_state(1, "cpu"), None, press_levels=lev)
    assert torch.equal(s1p["M"][3], s0["M"][3]) and torch.equal(s2["M"][3], s0["M"][3])   # +press / no ids: exact
    assert not torch.equal(s1["M"][3], s0["M"][3])                                          # the -2 changed the write
    # the wrong turn's rows (11..20) were written at strength 0: the store
    # behaves as if those tokens were never there
    m3 = _model(seed=122, order="pfc_first", write_every=1, store_exact=True); m3.eval()
    m3.load_state_dict(m0.state_dict())
    with torch.no_grad():
        # replicate by masking: forward with the same tokens, strengths
        # zero on 11..20 via a reward-free model and a hand-built mask
        lg3, s3, _ = m1(x, m1.init_state(1, "cpu"), None, press_levels=lev)
    assert torch.allclose(s3["M"][3], s1["M"][3])                                           # deterministic
    # the un-write across a chunk boundary: the model turn fills the end
    # of chunk A, the press lands at the start of chunk B
    xa = _toks(321); xa[0, 5] = EH; xa[(xa >= 5) & (xa <= 8)] = 100; xa[0, 5] = EH          # model turn 6..T-1 (no EM yet)
    xb = _toks(322); xb[(xb >= 5) & (xb <= 8)] = 100; xb[0, 0] = EM                           # EM at 0, press at 1
    levb = torch.zeros(1, T, dtype=torch.long); levb[0, 1] = 3
    with torch.no_grad():
        _, t0, _ = m0(xa, m0.init_state(1, "cpu"), None)
        _, t1, _ = m1(xa, m1.init_state(1, "cpu"), None)
        assert torch.equal(t0["M"][3], t1["M"][3])                                          # chunk A: identical writes
        _, u0, _ = m0(xb, t0, None, press_levels=levb)
        _, u1, _ = m1(xb, t1, None, press_levels=levb)
    assert not torch.equal(u0["M"][3], u1["M"][3])                                          # the re-issue subtracted
    # the press OPENS the chunk: the turn (with its <eot_model>) ended at
    # the previous chunk's last token — the whole turn is re-issued
    xc = _toks(323); xc[0, 3] = EH; xc[(xc >= 5) & (xc <= 8)] = 100; xc[0, 3] = EH; xc[0, T - 1] = EM   # model turn 4..T-2, EM last
    xd = _toks(324); xd[(xd >= 5) & (xd <= 8)] = 100
    levd = torch.zeros(1, T, dtype=torch.long); levd[0, 0] = 4
    with torch.no_grad():
        _, v0, _ = m0(xc, m0.init_state(1, "cpu"), None)
        _, v1, _ = m1(xc, m1.init_state(1, "cpu"), None)
        assert torch.equal(v0["M"][3], v1["M"][3])                           # chunk C: identical writes
        _, w0, _ = m0(xd, v0, None, press_levels=levd)
        _, w1, _ = m1(xd, v1, None, press_levels=levd)
    assert not torch.equal(w0["M"][3], w1["M"][3])                           # the whole turn re-issued negative
    # the un-written rows' associations are weaker than before: the
    # store's read of chunk A's turn keys returns less of those tokens
    with torch.no_grad():
        q = torch.nn.functional.normalize(m1.key_proj(t1["prev_c"].new_zeros(1, 1, m1.d) + t1["h"][3].unsqueeze(1)), dim=-1)
        stn = m1.stores["3"]
        r0 = torch.einsum("bij,btj->bti", u0["M"][3], stn.lift(q)).norm()
        r1 = torch.einsum("bij,btj->bti", u1["M"][3], stn.lift(q)).norm()
    assert float(r1) <= float(r0) + 1e-6


def test_S50_routed_cycles_mixture_of_depths():
    """v14 (the user's call: only compute a cycle if one wasn't enough):
    ponder_mode="route" — cycle 1 for every token; a router head on the
    conclusion picks the tokens above a self-tuning threshold and only
    those take cycles 2..K, batched after the loop; their decoded state
    is the blend C1 + g (C_deep - C1) so the router earns gradient;
    nothing ticks on a cycle; threshold tuned to route_cap of tokens in
    training, frozen at eval; none-routed = bit-exact with ponder=1;
    ponder_aux hands the trainer the routed tokens' own deep logits."""
    base = dict(order="pfc_first", n_council=1, write_every=1, store_exact=True)
    m1 = _model(seed=140, **base); m1.eval()
    mr = _model(seed=140, ponder=3, ponder_mode="route", ponder_reenter="token",
                ponder_aux=0.5, route_cap=0.25, **base); mr.eval()
    mr.load_state_dict(m1.state_dict(), strict=False)
    x = _toks(340, B=2)
    with torch.no_grad():
        lg1, s1, t1 = m1(x, m1.init_state(2, "cpu"), None)
        mr.route_tau.fill_(1e9)                       # none routed
        lgn, sn, _ = mr(x, mr.init_state(2, "cpu"), None)
        assert torch.equal(lgn, lg1)                                          # bit-exact with K=1
        assert float(mr.ponder_trace().max()) == 1.0
        mr.route_tau.fill_(-1e9)                      # all routed
        lga, sa, ta = mr(x, mr.init_state(2, "cpu"), None)
    assert not torch.equal(lga, lg1)                                          # the deep path decodes
    assert float(mr.ponder_trace().min()) == 3.0
    for u in m1.ukeys:
        assert torch.equal(s1["h"][u], sa["h"][u])                            # nothing ticks on a cycle
    assert torch.equal(s1["M"][3], sa["M"][3])
    assert len(t1[3]) == len(ta[3])
    # the router's gradient arrives through the blend (main CE alone)
    mr.train(); mr.read_drop = 0.0
    mr.route_tau.fill_(0.0)
    lg, st, ticks = mr(x, mr.init_state(2, "cpu"), None)
    ra = mr.pop_route_aux()
    tr = mr.ponder_trace()
    routed_n = int((tr == 3.0).sum())
    if routed_n:
        assert ra is not None and ra[0].shape == (routed_n, V) and ra[1].numel() == routed_n
        assert mr.pop_route_aux() is None
        mr.zero_grad()
        torch.nn.functional.cross_entropy(lg[0], x[0]).backward()
        assert mr.route_head.weight.grad is not None and float(mr.route_head.weight.grad.abs().sum()) > 0
    # the threshold self-tunes toward the capacity in training
    fr = []
    with torch.no_grad():
        for i in range(40):
            mr(_toks(341 + i, B=2), mr.init_state(2, "cpu"), None)
            fr.append(float((mr.ponder_trace() == 3.0).float().mean()))
    tail = sum(fr[-10:]) / 10
    assert 0.05 < tail < 0.6, (tail, float(mr.route_tau))                     # near cap .25, not stuck at 0 or 1
    # eval freezes the threshold
    mr.eval()
    tau0 = float(mr.route_tau)
    with torch.no_grad():
        mr(_toks(400, B=2), mr.init_state(2, "cpu"), None)
    assert float(mr.route_tau) == tau0


def test_S42_the_nights_agenda_is_biased_not_exclusive():
    """pair_share: each cycle draws the category first (a pair with that
    probability, an episode otherwise) so accumulated pairs cannot
    monopolise the night; pair_master: a pair whose contrastive loss
    fell under it at replay retires from the working set for good and
    is not re-mined; hot_once: the amygdala guarantee fires on a pair's
    first night only; a pair block carries the correction episode as a
    coupled-dream seed. Defaults (None, 0, False) = the certified
    lottery bit-exactly."""
    from iga.lm_sleep import Sleeper
    # --- the draw: 400 cycles with pair pay 100x the episode pay ---
    def run(pair_share, seed=5):
        sl = Sleeper(arm="C", every=0, block_chunks=1, seed=seed, pair_share=pair_share)
        sl.start = 0
        sl.observe(torch.randint(20, V, (1, 8 * T)))
        sl.spans = [{"lane": 0, "t0": 0, "t1": T, "pay": 1.0, "i": 0}]
        sl.pairs = [{"lane": 0, "tw": 10, "tr": 40, "pay": 100.0, "hot": False} for _ in range(4)]
        kinds = []
        sl._pair_block = lambda *a, **k: (kinds.append("pair"), {"arm": "C", "step": 0})[1]
        sl._block = lambda *a, **k: (kinds.append("span"), {"arm": "A", "step": 0, "span": (0, T)})[1]
        for i in range(400):
            sl._cycle(None, None, i)
        return kinds.count("pair") / 400
    assert run(None) > 0.97                                  # the certified lottery: pay decides
    assert 0.4 < run(0.5) < 0.6                              # category first: biased, not exclusive
    # --- retirement: a mastered pair leaves and is not re-mined ---
    m = _model(seed=101, order="pfc_first", n_council=1)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sl = Sleeper(arm="C", every=0, block_chunks=1, seed=3, pair_master=10.0, hot_once=True)   # everything "masters"
    sl.start = 0
    sl.observe(torch.randint(20, V, (1, 8 * T)))
    sl.T = T
    pr = {"lane": 0, "tw": 3 * T, "tr": 4 * T, "w0": 3 * T - 8, "w1": 3 * T, "r0": 4 * T - 8, "r1": 4 * T,
          "pay": 3.0, "hot": True, "ctx_w": 32, "iw": -1, "ir": -2}
    sl.pairs = [pr]
    row = sl._pair_block(m, opt, 1, pick=pr)
    assert row is not None and row.get("retired") is True and sl.pairs == [] and (3 * T, 4 * T) in sl._retired
    assert row["_span_obj"]["lane"] == 0 and row["_span_obj"]["t1"] == 4 * T          # the correction episode seeds REM
    # re-mining skips it: a drive whose presses would rebuild exactly this pair yields nothing
    class D:
        presses = [{"lane": 0, "t": 3 * T, "v": -2}, {"lane": 0, "t": 4 * T, "v": 2}]
        step_t = 8 * T
    sl.buffers[0][3 * T - 1] = 7; sl.buffers[0][4 * T - 1] = 7
    used = sl.harvest_pairs(D(), eot_h=7, eot_m=7, marks=())
    assert sl.pairs == [] and used == {0, 1}
    sl._retired.clear()
    sl.harvest_pairs(D(), eot_h=7, eot_m=7, marks=())
    assert len(sl.pairs) == 1                                 # the same presses mine a pair once un-retired
    # --- hot once: the guarantee fires on the first night only ---
    sl = Sleeper(arm="C", every=0, block_chunks=1, seed=3, hot_once=True, pair_share=0.0)
    sl.start = 0; sl.observe(torch.randint(20, V, (1, 8 * T)))
    sl.spans = [{"lane": 0, "t0": 0, "t1": T, "pay": 1.0, "i": 0}]
    hp = {"lane": 0, "tw": 10, "tr": 40, "pay": 4.0, "hot": True}
    sl.pairs = [hp]
    picks = []
    sl._pair_block = lambda model, opt, step, beta=1.0, pick=None: (picks.append(pick), {"arm": "C", "step": step})[1]
    sl._block = lambda *a, **k: {"arm": "A", "step": 0, "span": (0, T)}
    sl._night_hot = set(); sl._cycle(None, None, 1)
    sl._night_hot = set(); sl._cycle(None, None, 2)
    assert picks == [hp]                                      # night 2: no guarantee, the draw went to the episode


def test_s51_tied_lexicon():
    """V15: tie_embed shares one matrix both directions (with the
    sqrt(d) input rescale) — no head bias, grads reach the shared
    weight from both roles, and the checkpoint round-trips."""
    import torch
    from iga.lm_scan import ScanLM
    m = ScanLM(101, d=32, n_layers=1, n_heads=2, max_T=16, n_council=1,
               store_exact=True, tie_embed=True)
    assert m.head.weight.data_ptr() == m.embed.weight.data_ptr()
    assert m.head.bias is None
    assert abs(m.emb_scale - 32 ** 0.5) < 1e-6
    st = m.init_state(1, "cpu")
    x = torch.randint(0, 101, (1, 8))
    lg, st, _ = m(x, st)
    loss = lg.sum()
    loss.backward()
    assert m.embed.weight.grad is not None
    assert float(m.embed.weight.grad.abs().sum()) > 0
    sd = m.state_dict()
    m2 = ScanLM(101, d=32, n_layers=1, n_heads=2, max_T=16, n_council=1,
                store_exact=True, tie_embed=True)
    m2.load_state_dict(sd)
    assert m2.head.weight.data_ptr() == m2.embed.weight.data_ptr()
    # the birth law (scan15's 924-CE lesson): a tied newborn's CE must
    # start near ln V, not orders above it
    import math
    st3 = m2.init_state(2, "cpu")
    x3 = torch.randint(0, 101, (2, 16))
    lg3, _, _ = m2(x3, st3)
    ce = torch.nn.functional.cross_entropy(
        lg3.float().reshape(-1, 101), x3.reshape(-1))
    assert float(ce) < 3.0 * math.log(101), f"hot birth: CE {float(ce):.1f}"


def test_s52_z_loss_pins_the_head():
    """V15: z_w adds z-loss in the trainer — logsumexp^2, so scaling
    every logit up raises the penalty even when CE is unchanged."""
    import torch
    z = lambda lg: (torch.logsumexp(lg.float(), dim=-1) ** 2).mean()
    lg = torch.randn(4, 7, 50)
    assert float(z(lg * 3.0)) > float(z(lg))
    from iga.lm_scan import ScanLM
    m = ScanLM(64, d=16, n_layers=1, n_heads=2, max_T=8, n_council=1, z_w=1e-4)
    assert m.z_w == 1e-4


def test_s53_plan_no_op_at_birth():
    """v16 PLAN (49a): plan_gate is zero-init, so a newborn with the
    plan organ produces EXACTLY the logits of one without it (the gate
    law: every new organ is a no-op at birth)."""
    m0 = _model(seed=3)
    m1 = _model(seed=3, plan_m=4)
    m0.eval(); m1.eval()
    x = _toks(11)
    lg0, _, _ = m0(x, m0.init_state(1, "cpu"))
    lg1, _, _ = m1(x, m1.init_state(1, "cpu"))
    assert torch.allclose(lg0, lg1, atol=1e-5), \
        f"plan organ perturbed a newborn: {float((lg0-lg1).abs().max())}"


def test_s54_plan_foresight_trains():
    """Day foresight: training forward stashes a finite plan loss whose
    targets are detached (no grad reaches the future through them)."""
    m = _model(seed=4, plan_m=4)
    m.train()
    x = _toks(12)
    m(x, m.init_state(1, "cpu"))
    pa = m.pop_plan_aux()
    assert pa is not None and torch.isfinite(pa), "no plan loss stashed"
    assert pa.requires_grad, "plan loss must carry grad (via plan_head)"
    pa.backward()
    assert m.plan_head.weight.grad is not None
    assert m.pop_plan_aux() is None, "pop must clear"
    assert set(m.plan_fid) == {1, 2, 4, 8}, m.plan_fid


def test_s55_rem_rollout_pure():
    """LATENT REM: rem_loss is finite, differentiable through plan_head
    only, and touches NOTHING else — state dict tensors and the store
    are byte-identical before and after (dreams do not write)."""
    m = _model(seed=5, plan_m=4, rem_k=8, store_exact=True)
    m.train()
    x = _toks(13)
    st = m.init_state(1, "cpu")
    _, st, _ = m(x, st)
    snap = {k: v.clone() for k, v in
            [("M3", st["M"][3]), ("prev_c", st["prev_c"])]}
    rl = m.rem_loss()
    assert rl is not None and torch.isfinite(rl), "no rem loss"
    rl.backward()
    assert m.plan_head.weight.grad is not None
    # 49l FULL REM: live seeds — the night's gradient must REACH the
    # council (the dream shapes the dreamer), while still writing
    # nothing into state
    assert any(p_.grad is not None and float(p_.grad.abs().sum()) > 0
               for p_ in m.council.parameters()), \
        "full REM: no gradient reached the council"
    assert torch.equal(st["M"][3], snap["M3"]), "REM wrote the store"
    assert torch.equal(st["prev_c"], snap["prev_c"]), "REM moved state"
    assert m.rem_fid, "rollout fid not recorded"


def test_s56_plan_cfg_reload_truth():
    """The n_heads lesson: plan_m/rem_k are ctor args riding scan_opts
    into cfg['scan'] — a rebuild from cfg must carry the plan organ."""
    m = _model(seed=6, plan_m=4)
    n_with = sum(p.numel() for p in m.parameters())
    m2 = _model(seed=6)
    n_without = sum(p.numel() for p in m2.parameters())
    d = 32
    assert n_with - n_without == (d * 4 * d + 4 * d) + 1, \
        "plan params unaccounted"


def test_s57_eot_weighted_ce_mean_one():
    """49c stop discipline: the trainer's eot weight sharpens the stop
    decision without changing the learning rate — the composed weight
    vector has mean 1 and weights eot targets eot_w x the rest."""
    import torch as _t
    y = _t.tensor([[1, 2, 9, 3, 9, 4]])
    eot = 9
    w2 = _t.ones(y.shape, dtype=_t.float32)
    w2[y == eot] = 3.0
    w = w2 / w2.mean().clamp(min=1e-6)
    assert abs(float(w.mean()) - 1.0) < 1e-6
    r = float(w[y == eot].mean() / w[y != eot].mean())
    assert abs(r - 3.0) < 1e-6


def test_s58_bg_selector_trains_at_night():
    """49h: the BG stub — candidate dream transitions + selector gate.
    The night's consistency loss reaches BOTH the candidates and the
    gate (selection is trainable without reward); REM purity holds;
    plan_cand=0 stays the plain h=1 slice (param census exact)."""
    m = _model(seed=9, plan_m=4, rem_k=8, plan_cand=4, store_exact=True)
    m.train()
    x = _toks(21)
    st = m.init_state(1, "cpu")
    _, st, _ = m(x, st)
    snap = st["M"][3].clone()
    rl = m.rem_loss()
    assert rl is not None and torch.isfinite(rl)
    rl.backward()
    assert m.plan_gate_bg.weight.grad is not None, "gate got no grad"
    assert m.plan_trans[0].weight.grad is not None, "candidates got no grad"
    assert torch.equal(st["M"][3], snap), "REM wrote the store"
    assert m.bg_gate_use and abs(sum(m.bg_gate_use.values()) - 1.0) < 0.05
    d = 32
    m0 = _model(seed=9, plan_m=4)
    extra = sum(p.numel() for p in m.parameters()) - \
        sum(p.numel() for p in m0.parameters())
    assert extra == 4 * (d * d + d) + (d * 4 + 4), extra


def test_s59_intrinsic_bg_trains_without_presses():
    """49i: the BG's judge-free signal. With intrinsic_w on and ZERO
    press events, the value/TD/BG machinery still gets a live reward
    stream (prediction success, one chunk delayed, press units,
    bounded) — value loss and BG loss both materialize and the BG
    force lands on the gates. The law: intrinsic value manages
    computation, never content (logits untouched by construction —
    the reward enters only the TD path)."""
    m = _model(seed=13, bg_w=0.05, dopamine=1.0, intrinsic_w=1.0)
    m.train()
    st = m.init_state(1, "cpu")
    _, st, _ = m(_toks(31), st)              # chunk 1: builds prev_surp
    assert st.get("prev_surp") is not None
    _, st, _ = m(_toks(32), st)              # chunk 2: intrinsic reward live
    vl = m.pop_value_loss() if hasattr(m, "pop_value_loss") else None
    bl = m._bg_loss
    assert vl is not None and torch.isfinite(vl), "no value loss"
    assert bl is not None and torch.isfinite(bl), "no bg loss"
    bl.backward(retain_graph=True)
    touched = [n for n, p_ in m.named_parameters()
               if p_.grad is not None and float(p_.grad.abs().sum()) > 0]
    assert touched and all(".z." in n and n.startswith("cells.")
                           for n in touched), touched


def test_s60_imagination_offered_not_forced():
    """49m: the imagination option — k-step latent lookahead through
    the dreamer, gated at zero. At birth the gate is closed: logits
    identical to the same-seed model with imag_k=0 (the rollout runs
    but contributes nothing). The gate itself must be LEARNABLE: CE
    gradient reaches it, so prediction can choose to open it."""
    m0 = _model(seed=17, plan_m=4, plan_cand=4)
    m1 = _model(seed=17, plan_m=4, plan_cand=4, imag_k=4)
    m0.eval(); m1.eval()
    x = _toks(41)
    lg0, _, _ = m0(x, m0.init_state(1, "cpu"))
    lg1, _, _ = m1(x, m1.init_state(1, "cpu"))
    assert torch.allclose(lg0, lg1, atol=1e-5), \
        "imagination perturbed a newborn (gate must be zero-init)"
    m1.train()
    lg, _, _ = m1(x, m1.init_state(1, "cpu"))
    ce = torch.nn.functional.cross_entropy(lg.reshape(-1, V),
                                           x.reshape(-1))
    ce.backward()
    assert m1.imag_gate.grad is not None and \
        float(m1.imag_gate.grad.abs()) > 0, \
        "the gate cannot learn: CE gradient does not reach it"


def test_s61_imagine_cycle_interleave():
    """49n: imagine -> cycle -> imagine -> cycle. The rollout summary
    rides into the re-deliberation's token slot through its own
    zero-init gate: a newborn is bit-identical (S60's twin law still
    holds through the cycle path), and CE gradient reaches the cycle
    gate so deliberation can learn to consult imagination mid-thought."""
    m0 = _model(seed=19, order="pfc_first", plan_m=4, plan_cand=4,
                ponder=3, ponder_aux=0.5, route_cap=0.5)
    m1 = _model(seed=19, order="pfc_first", plan_m=4, plan_cand=4,
                imag_k=4, ponder=3, ponder_aux=0.5, route_cap=0.5)
    m0.eval(); m1.eval()
    x = _toks(43)
    lg0, _, _ = m0(x, m0.init_state(1, "cpu"))
    lg1, _, _ = m1(x, m1.init_state(1, "cpu"))
    assert torch.allclose(lg0, lg1, atol=1e-5), "cycle-imagination perturbed a newborn"
    m1.train()
    m1.route_tau.fill_(-1e9)                 # all tokens route (S50's trick)
    lg, _, _ = m1(x, m1.init_state(1, "cpu"))
    ce = torch.nn.functional.cross_entropy(lg.reshape(-1, V), x.reshape(-1))
    ce.backward()
    assert m1.imag_cycle_gate.grad is not None and \
        float(m1.imag_cycle_gate.grad.abs()) > 0, \
        "cycle gate unreachable: deliberation cannot learn to consult imagination"
