"""v5.4 — the complete architecture as an LLM (A19 + A24).

Attention is the hands, the bands are the memory, imagination is the
lookahead, the ledger is the law.

Substrate: a standard decoder-only transformer over each 512-token
window (language, in-window lookup). Riding above it: slow band
latents at clocks 512 / 4096 / 32768 — persistent across the whole
run, injected back as MEMORY TOKENS the transformer attends over.

v5.4 (A24, from the run-4 autopsy): the v5.3 bands were written with
the MEAN of all window hiddens each tick — an unforecastable soup;
fid:4/5 sat below floor all run, the gate vetoed every cross-window
frontier, and the real carry transient died unpaid. Now:
  selective read — each band attention-pools the window with its own
    learned query (it chooses what to look at, not everything);
  closed-by-default write — an exposed-gate slow cell, update gate
    biased shut at init, so the state drifts slowly and the band's
    forward model has something learnable to predict;
  write cost — a small penalty on open gates keeps writes sparse.

Same forward API: (tokens, st, _) -> (logits, st, ticks). Lesion
zeroes the memory tokens. The trainer reads model.pop_write_cost().
"""

import torch
import torch.nn as nn

from .lm_bands import N_BANDS
from .lm_transformer import Block

HYBRID_CLOCKS = {3: 1, 4: 8, 5: 64}   # band idx -> tick every N chunks
                                       # (chunks of 512 -> 512/4k/32k)


class BandMatrix(nn.Module):
    """A28: fast-weights associative store per slow band. The math
    that chose it: a squashing recurrent vector holds ~1-2 facts
    (every write decays all content by (1-z) through tanh) while the
    spans hold 5-15; a delta-rule matrix holds ~d/(2 ln d) pairs with
    crosstalk ~sqrt(n/d), degrading gracefully, and its capacity
    grows with d^2 at scale. Writes are additive (no erasure); the
    timescale ladder becomes per-band DECAY (half-life = the band's
    clock). Write head is dedicated (separate from the read query).
    Cross-chunk detachment breaks write-path gradient, so writes
    learn from an in-chunk write-fidelity loss (read back the just-
    written pair); the read path learns from LM + pay downstream."""

    def __init__(self, d, decay):
        super().__init__()
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wq = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.decay = decay
        self.beta = nn.Parameter(torch.tensor(0.0))  # sigmoid -> 0.5

    def write(self, M, x, stale_ok=False):
        """x: [B, d] this chunk's write selection. Delta rule:
        M <- (1-decay) M + beta (v - M k) k^T. Returns (M', recon).
        stale_ok (A38): the store pass's backward runs from the NEXT
        chunk, after opt.step() has bumped parameter versions in
        place — apply CLONED weights so the saved tensors stay valid.
        Gradient still reaches the parameter leaves through the
        clone; values are one step stale (standard TBPTT)."""
        Wk = self.wk.weight.clone() if stale_ok else self.wk.weight
        Wv = self.wv.weight.clone() if stale_ok else self.wv.weight
        k = nn.functional.normalize(
            nn.functional.linear(x, Wk), dim=-1)
        v = nn.functional.linear(x, Wv)
        pred = torch.einsum("bij,bj->bi", M, k)
        M = (1 - self.decay) * M + torch.sigmoid(self.beta) * \
            torch.einsum("bi,bj->bij", v - pred, k)
        back = torch.einsum("bij,bj->bi", M, k)
        recon = (1 - nn.functional.cosine_similarity(
            back, v, dim=-1)).mean()
        return M, recon

    def read(self, M, h):
        """h: [B, T, d] -> per-position associative read [B, T, d]."""
        q = nn.functional.normalize(self.wq(h), dim=-1)
        r = torch.einsum("bij,btj->bti", M, q)
        return self.out(r)

    def write_keyed(self, M, K, V, s, stale_ok=False):
        """A52 (R4): token-keyed batch write. K [B, T, d] unit keys =
        DETACHED token embeddings (the address is the token's
        identity, not a learned projection); V [B, T, d] content
        (wv applied here); s [B, T] per-pair write strength. One
        (k, v) pair PER POSITION — replaces the one-gist-per-chunk
        softmax selection that made item retrieval impossible.
        Chunkwise-parallel delta rule: predictions against the
        chunk-initial M (same-chunk same-token writes blend). The
        update is the strength-NORMALIZED convex mix of single-pair
        delta steps — a plain sum diverged in minutes (a token with
        n same-chunk occurrences got its correction n times over;
        n*beta*s > 2 overshoots and oscillates; whitespace has
        n in the hundreds -> NaN by step ~1.5k, first r4 pod). The
        normalization also makes tok_u truly price storage: as glue
        strengths fall, each surviving write gets a larger share.
        Returns (M', recon) like write()."""
        Wv = self.wv.weight.clone() if stale_ok else self.wv.weight
        v = nn.functional.linear(V, Wv)
        pred = torch.einsum("bij,btj->bti", M, K)
        upd = torch.einsum(
            "bti,btj->bij", s.unsqueeze(-1) * (v - pred), K)
        denom = s.sum(dim=1).clamp(min=1e-3)
        M = (1 - self.decay) * M + torch.sigmoid(self.beta) * \
            upd / denom.unsqueeze(-1).unsqueeze(-1)
        back = torch.einsum("bij,btj->bti", M, K)
        # A52b: recon weights DETACHED from the strength path. R4
        # proved tok_u games an s-weighted fidelity loss: pushing
        # strength onto frequent easy keys ('the', '=', newlines)
        # minimizes weighted recon while making the store a glue
        # echo. Detached, recon still measures fidelity where mass
        # actually sits, but tok_u learns ONLY from read-usefulness
        # (read-mix logits every position + A38 next-chunk credit).
        w = (s / (s.sum(dim=1, keepdim=True) + 1e-6)).detach()
        recon = ((1 - nn.functional.cosine_similarity(back, v, dim=-1))
                 * w).sum(dim=1).mean()
        return M, recon

    def read_keyed(self, M, q):
        """q [B, T, d] unit queries built from use-site token
        embeddings. No wq: write-key space and read-query space are
        the SAME space by construction — a use of token t addresses
        exactly the slot written at t."""
        r = torch.einsum("bij,btj->bti", M, q)
        return self.out(r)


class LogitStore(nn.Module):
    """A53 (R5): decode-free item store, capacity-sized. The A52b
    bench proved the learned value/decode path (wv in, out out) is
    dead — far-read credit is too sparse/noisy to shape it. Here
    NOTHING between store and prediction is learned but one scalar:
    values are token IDENTITIES (unit embedding rows), and the read
    result is matched straight against the vocabulary and added to
    the LOGITS. Keys are context mixes lifted into a D-dim space by
    a FROZEN random projection (JL: inner products preserved) — D is
    decoupled from model width and sized per band to the measured
    write load (the A52b capacity arithmetic: a delta-rule matrix
    holds ~D pairs against dot-product readout; band 5's ~92-chunk
    pair lifetime needs thousands, and d=256 drowned it)."""

    def __init__(self, d, D, decay, seed):
        super().__init__()
        self.D = D
        self.decay = decay
        self.beta = nn.Parameter(torch.tensor(0.0))
        g = torch.Generator().manual_seed(seed)
        # NONLINEAR lift (random Fourier features). A linear
        # projection of d-dim keys spans a d-dim subspace of R^D —
        # rank, not ambient dimension, sets capacity, so a linear
        # lift buys NOTHING (caught by the capacity law test: 64
        # keys in an effective 32-dim space erased each other).
        # cos(Px+b) images span D dims; gamma sets the kernel
        # bandwidth so orthogonal unit tokens decorrelate to ~0.15
        # while nearby context mixes stay matched.
        self.register_buffer(
            "proj", 1.4 * torch.randn(D, d, generator=g))
        self.register_buffer(
            "phase", 2 * 3.141592653589793
            * torch.rand(D, generator=g))

    def lift(self, x):
        """[.., d] context mix -> unit RFF key in R^D."""
        return nn.functional.normalize(
            torch.cos(nn.functional.linear(x, self.proj)
                      + self.phase), dim=-1)

    def write(self, M, K, V, s, stale_ok=False):
        """M [B, d, D], K [B, T, D] unit lifted keys, V [B, T, d]
        unit identity values (detached), s [B, T] write strength.
        Convex strength-normalized delta rule (A52 NaN law); recon
        weights detached (A52b anti-gaming law)."""
        beta = self.beta.clone() if stale_ok else self.beta
        pred = torch.einsum("bij,btj->bti", M, K)
        upd = torch.einsum("bti,btj->bij",
                           s.unsqueeze(-1) * (V - pred), K)
        denom = s.sum(dim=1).clamp(min=1e-3)
        M = (1 - self.decay) * M + torch.sigmoid(beta) * \
            upd / denom.unsqueeze(-1).unsqueeze(-1)
        back = torch.einsum("bij,btj->bti", M, K)
        w = (s / (s.sum(dim=1, keepdim=True) + 1e-6)).detach()
        recon = ((1 - nn.functional.cosine_similarity(back, V,
                                                      dim=-1))
                 * w).sum(dim=1).mean()
        return M, recon

    def read(self, M, q):
        """q [B, T, D] lifted queries -> retrieved identity vectors
        [B, T, d], ready for the vocabulary match."""
        return torch.einsum("bij,btj->bti", M, q)


class SlowCell(nn.Module):
    """Gated delta-write with the update gate biased closed at init:
    h' = (1-z)*h + z*cand. At init z ~ sigmoid(gate_bias) so the
    state barely moves until training earns the right to write."""

    def __init__(self, d, gate_bias=-2.0):
        super().__init__()
        self.z = nn.Linear(2 * d, d)
        self.cand = nn.Linear(2 * d, d)
        nn.init.constant_(self.z.bias, gate_bias)

    def forward(self, x, h):
        hx = torch.cat([h, x], dim=-1)
        z = torch.sigmoid(self.z(hx))
        c = torch.tanh(self.cand(hx))
        return (1 - z) * h + z * c, z.mean()


class HybridLM(nn.Module):
    def __init__(self, vocab_size, d=128, n_layers=6, n_heads=8,
                 max_T=512, talk=None, widths=None, store="vector",
                 use_xl=True, gate_init=-4.0, read_drop=0.5,
                 gate_mode="scalar", keyed=None):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.max_T = max_T
        self.store = store
        self.use_xl = use_xl   # A36: benched in v6.0 — real one-boundary
                               # reach (A33) but unresolved held-out cost
                               # and large run-variance; revisit at scale
        self.mid = n_layers // 2                    # read injection depth
        self.bands = sorted(HYBRID_CLOCKS)          # [3, 4, 5]
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_T + len(self.bands), d)
        self.blocks = nn.ModuleList(
            [Block(d, n_heads) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)
        self.cells = nn.ModuleDict(
            {str(k): SlowCell(d) for k in self.bands})
        self.read_q = nn.ParameterDict(
            {str(k): nn.Parameter(torch.randn(d) / d ** 0.5)
             for k in self.bands})
        self.pred = nn.ModuleDict(
            {str(k): nn.Linear(d, d) for k in self.bands})
        self.mem_proj = nn.ModuleDict(
            {str(k): nn.Linear(d, d, bias=False) for k in self.bands})
        if store == "matrix":
            # half-life = the band's clock, in chunks
            self.mats = nn.ModuleDict(
                {str(k): BandMatrix(d, 1 - 0.5 ** (1 / HYBRID_CLOCKS[k]))
                 for k in self.bands})
            self.write_q = nn.ParameterDict(
                {str(k): nn.Parameter(torch.randn(d) / d ** 0.5)
                 for k in self.bands})
            # A52 (R4): token-keyed storage. Three consecutive runs
            # (v8.0, R1, R2) put true-memory retrieval at chance,
            # lesion-invariant, while R2 proved the DEMAND signal
            # works. Diagnosis: addressing — learned-soup keys over
            # one softmax gist per chunk cannot fetch one identifier
            # among thousands. keyed="token" writes one pair per
            # POSITION with key = the token's own (detached, unit)
            # embedding, and reads with a query mixed from the last
            # QR tokens' embeddings: tok_u learns which token TYPES
            # are worth storing/asking about (identifiers vs glue),
            # qmix a recency prior. Same space on both sides — a
            # shared rare token bridges use site to definition site
            # with no learned alignment needed.
            self.keyed = keyed
            if keyed == "token":
                self.QR = 8
                self.tok_u = nn.Parameter(torch.zeros(vocab_size))
                self.qmix = nn.Parameter(torch.zeros(self.QR))
            if keyed == "logit":
                # A53 (R5): decode-free capacity-sized stores.
                # QR=64: bridge ceiling 30.2% (A52b curve). KD per
                # band from load = writes/chunk x pair lifetime
                # (band 3 ~2 chunks, band 4 ~12, band 5 ~92).
                # alpha init 0: the bonus is opt-in like the gates;
                # gradient flows at 0. The mid-layer residual read
                # (the dead decode route) is OFF in this mode.
                self.QR = 64
                self.KD = {3: 512, 4: 1024, 5: 2048}
                self.tok_u = nn.Parameter(torch.zeros(vocab_size))
                self.qmix = nn.Parameter(torch.zeros(self.QR))
                self.stores = nn.ModuleDict(
                    {str(k): LogitStore(
                        d, self.KD[k],
                        1 - 0.5 ** (1 / HYBRID_CLOCKS[k]),
                        seed=1000 + k)
                     for k in self.bands})
                self.alpha = nn.ParameterDict(
                    {str(k): nn.Parameter(torch.tensor(0.0))
                     for k in self.bands})
            # A30: reads gated shut at init (sigmoid(-4) ~ 0.018) —
            # v5.6 proved ungated per-position reads crowd out
            # induction formation; the model must opt in. gate_init
            # and read_drop are v6.2 bootstrap knobs (A39): defaults
            # reproduce v6.0/v6.1 exactly.
            self.read_gate = nn.ParameterDict(
                {str(k): nn.Parameter(torch.tensor(float(gate_init)))
                 for k in self.bands})
            # A41 candidate — per-position read gates: the v6.1 82k
            # snapshot showed the DILUTION stall (betas 0.94/0.99 =
            # write path engaged; scalar gates pinned = one knob
            # cannot price reads that help at ask-positions and cost
            # noise at the other 511). gate_mode="position" gives
            # each position its own learned gate (d->1, bias at
            # gate_init): asks can open the vault while chatter
            # keeps it shut. Same protection at init.
            self.gate_mode = gate_mode
            if gate_mode == "entropy":
                # A51 (R2): metamemory — reads flow only where the
                # BLIND path is uncertain. The trainer runs a blind
                # pass (reads off, throwaway state), computes
                # per-position entropy H, and sets entropy_gate =
                # sigmoid(ent_a*(H - ent_tau)) before the real pass.
                # The crutch loop is broken twice: blind CE trains
                # the base every chunk, and confident positions get
                # no read at all.
                self.ent_a = nn.Parameter(torch.tensor(1.0))
                self.ent_tau = nn.Parameter(torch.tensor(2.0))
                self.entropy_gate = None
            if gate_mode == "position":
                self.read_gate_pos = nn.ModuleDict()
                for k in self.bands:
                    lin = nn.Linear(d, 1)
                    nn.init.zeros_(lin.weight)
                    nn.init.constant_(lin.bias, float(gate_init))
                    self.read_gate_pos[str(k)] = lin
        self.read_drop = read_drop
        # A30: Transformer-XL chunk carry — the previous chunk's
        # hiddens as attendable keys. v5.6's autopsy: chunks were
        # processed independently, so ANY boundary-straddling gap
        # (even 48 tokens) was invisible to attention and fell to
        # the store. Attention now owns everything within one chunk
        # of lookback; the store owes only true long range.
        self.xl_tag = nn.Parameter(torch.zeros(d))
        self.lesioned = set()
        self._write_cost = None
        self._recon = None

    def init_state(self, B, device):
        st = {"h": {k: torch.zeros(B, self.d, device=device)
                    for k in self.bands},
              "acc": {k: torch.zeros(B, self.d, device=device)
                      for k in self.bands},
              "cnt": {k: 0 for k in self.bands},
              "pend": {k: None for k in self.bands},
              "chunk": 0}
        if self.store == "matrix":
            if getattr(self, "keyed", None) == "logit":
                st["M"] = {k: torch.zeros(B, self.d, self.KD[k],
                                          device=device)
                           for k in self.bands}
            else:
                st["M"] = {k: torch.zeros(B, self.d, self.d,
                                          device=device)
                           for k in self.bands}
        st["xl"] = None            # per-layer cached hiddens (A30)
        return st

    def detach_state(self, st):
        st["h"] = {k: v.detach() for k, v in st["h"].items()}
        st["acc"] = {k: v.detach() for k, v in st["acc"].items()}
        st["pend"] = {k: (p.detach() if p is not None else None)
                      for k, p in st["pend"].items()}
        # A38: M is deliberately NOT detached here — it carries one
        # write-op of graph across the boundary (inputs were detached
        # at the write site), so the next chunk's read backward can
        # credit the write head. Depth cannot grow: each write starts
        # from M.detach().
        if st.get("xl") is not None:
            st["xl"] = [h.detach() for h in st["xl"]]
        return st

    def _mem_tokens(self, st, B):
        toks = []
        for k in self.bands:
            h = st["h"][k]
            if k in self.lesioned:
                h = torch.zeros_like(h)
            toks.append(self.mem_proj[str(k)](h))
        return torch.stack(toks, dim=1)          # [B, M, d]

    def forward(self, tokens, st, scene_starts=None):
        B, T = tokens.shape
        dev = tokens.device
        M = len(self.bands)
        mem = self._mem_tokens(st, B)
        x = self.embed(tokens) + self.pos(
            torch.arange(M, M + T, device=dev))[None]
        mem = mem + self.pos(torch.arange(M, device=dev))[None]
        x = torch.cat([mem, x], dim=1)           # [B, M+T, d]
        # causal over text; every text position may attend all memory;
        # memory rows attend only themselves
        sq = torch.triu(torch.ones(M + T, M + T, device=dev,
                                   dtype=torch.bool), diagonal=1)
        sq[:M, :] = True
        sq[torch.arange(M + T), torch.arange(M + T)] = False
        xl = st.get("xl") if self.use_xl else None
        if self.training and xl is not None and \
                float(torch.rand(())) < 0.5:
            # A34: XL-dropout — v5.8 proved the carry crowds out
            # induction (same-chunk 0.68 -> 0.31 while straddle
            # tripled); half the chunks train blind so the robust
            # circuit must form, the other half keep the reach
            xl = None
        self._xl_used = xl is not None
        if xl is not None:
            # XL carry (A30): previous chunk's per-layer text hiddens
            # as extra keys — text rows attend all of them (they are
            # wholly past); mem-token rows still attend only self
            xT = xl[0].shape[1]
            left = torch.ones(M + T, xT, device=dev, dtype=torch.bool)
            left[M:, :] = False
            mask = torch.cat([left, sq], dim=1)
        else:
            mask = sq
        new_xl = []
        read_ok = (not self.training) or \
            float(torch.rand(())) >= self.read_drop
        self._reads_used = read_ok and self.store == "matrix"
        for i, b in enumerate(self.blocks):
            if self.use_xl:
                new_xl.append(x[:, M:].detach())
            kv = (xl[i] + self.xl_tag) if xl is not None else None
            x = b(x, mask if xl is not None else sq, kv=kv)
            if self.store == "matrix" and i == self.mid - 1 \
                    and read_ok \
                    and getattr(self, "keyed", None) != "logit":
                # per-position associative reads from LAST chunks'
                # matrices, gated shut at init (A30) + read-dropout
                # (A36: the crowding-out law — half the chunks train
                # storeless so induction must form)
                text = x[:, M:]
                r = torch.zeros_like(text)
                q_tok = None
                if getattr(self, "keyed", None) == "token":
                    # A52: query = unit mix of the last QR tokens'
                    # embeddings — the same space the write keys
                    # live in, so no learned alignment is needed
                    E = nn.functional.normalize(
                        self.embed.weight, dim=-1).detach()
                    Tt = tokens.shape[1]
                    idx = torch.arange(Tt, device=tokens.device)
                    offs = torch.arange(self.QR, device=tokens.device)
                    rel = idx.unsqueeze(1) - offs.unsqueeze(0)
                    wtok = tokens[:, rel.clamp(min=0)]   # [B, T, QR]
                    lgt = self.qmix + self.tok_u[wtok]
                    lgt = lgt.masked_fill(
                        (rel < 0).unsqueeze(0), float("-inf"))
                    mixw = torch.softmax(lgt, dim=-1)
                    q_tok = nn.functional.normalize(torch.einsum(
                        "btr,btrd->btd", mixw, E[wtok]), dim=-1)
                for k in self.bands:
                    if k in self.lesioned:
                        continue
                    gm = getattr(self, "gate_mode", "scalar")
                    if gm == "position":
                        # [B, T, 1] — each position prices its own read
                        g = torch.sigmoid(self.read_gate_pos[str(k)](text))
                    elif gm == "entropy":
                        eg = getattr(self, "entropy_gate", None)
                        if eg is None:
                            g = 0.0          # no uncertainty signal: blind
                        else:
                            g = eg.unsqueeze(-1)     # [B, T, 1]
                    else:
                        g = torch.sigmoid(self.read_gate[str(k)])
                    if q_tok is not None:
                        r = r + g * self.mats[str(k)].read_keyed(
                            st["M"][k], q_tok)
                    else:
                        r = r + g * self.mats[str(k)].read(
                            st["M"][k], text)
                x = torch.cat([x[:, :M], text + r], dim=1)
        st["xl"] = new_xl if self.use_xl else None
        hidden = x[:, M:]                        # text positions
        logits = self.head(self.lnf(hidden))
        lg_E = lg_qd = lg_smask = None
        if self.store == "matrix" and \
                getattr(self, "keyed", None) == "logit":
            # A53 read: query = mix of the last QR tokens' embeddings
            # (window includes the current token); the retrieved
            # identity vectors are matched against the vocabulary and
            # added DIRECTLY to the logits — no learned decode. Reads
            # see the PREVIOUS chunks' M (writes happen below).
            lg_E = nn.functional.normalize(
                self.embed.weight, dim=-1).detach()
            Tt = tokens.shape[1]
            idx = torch.arange(Tt, device=tokens.device)
            offs = torch.arange(self.QR, device=tokens.device)
            rel = idx.unsqueeze(1) - offs.unsqueeze(0)
            wtok = tokens[:, rel.clamp(min=0)]
            lgt = self.qmix + self.tok_u[wtok]
            lgt = lgt.masked_fill((rel < 0).unsqueeze(0),
                                  float("-inf"))
            mix = torch.softmax(lgt, dim=-1)
            lg_qd = torch.einsum("btr,btrd->btd", mix, lg_E[wtok])
            # write-side index tensors (context STRICTLY before each
            # position — the induction shape; position 0 has none:
            # masked to uniform, write strength zeroed below). The
            # write MIXES are built in the write block, twice, for
            # the A38 two-pass credit (uncloned/cloned params).
            relw = idx.unsqueeze(1) - (offs + 1).unsqueeze(0)
            wtokw = tokens[:, relw.clamp(min=0)]
            allw = (relw < 0).all(dim=-1)
            lg_smask = (~allw).view(1, -1).to(lg_qd.dtype)
            if read_ok:
                rsum = None
                for k in self.bands:
                    if k in self.lesioned:
                        continue
                    stn = self.stores[str(k)]
                    r = self.alpha[str(k)] * stn.read(
                        st["M"][k], stn.lift(lg_qd))
                    rsum = r if rsum is None else rsum + r
                if rsum is not None:
                    logits = logits + rsum @ lg_E.t()
        # band updates: each band SELECTS from the window with its own
        # query (A24) instead of receiving the window mean
        ticks = [[] for _ in range(N_BANDS)]
        st["chunk"] += 1
        wcost = []
        for k in self.bands:
            w = torch.softmax(
                hidden @ self.read_q[str(k)] / self.d ** 0.5, dim=1)
            read = torch.einsum("bt,btd->bd", w, hidden)
            st["acc"][k] = st["acc"][k] + read
            st["cnt"][k] += 1
            if st["chunk"] % HYBRID_CLOCKS[k] == 0:
                pooled = st["acc"][k] / max(st["cnt"][k], 1)
                if st["pend"][k] is not None:
                    fid = nn.functional.cosine_similarity(
                        st["pend"][k], pooled, dim=-1)
                    ticks[k].append((T - 1, fid))
                st["h"][k], z = self.cells[str(k)](pooled, st["h"][k])
                wcost.append(z)
                st["pend"][k] = self.pred[str(k)](st["h"][k])
                st["acc"][k] = torch.zeros_like(st["acc"][k])
                st["cnt"][k] = 0
        if wcost:
            self._write_cost = torch.stack(wcost).mean()
        if self.store == "matrix":
            # dedicated write selection + additive write, EVERY chunk
            # (storage is non-destructive; decay is the timescale).
            # A38: the write path learns from NEXT-chunk reads. The
            # stored M keeps exactly one write-op of graph across the
            # boundary (its M input detached, window hiddens detached),
            # so read-success backward at chunk t+1 reaches write_q/
            # wk/wv/beta — the severed credit that left the selector
            # blind (v5.6-v6.0: gist equilibrium, cross bins dead).
            # Two passes over identical math: the recon pass is
            # traversed by THIS chunk's backward, the store pass by
            # the NEXT chunk's — shared nodes would be freed twice.
            recon = []
            h_wr = hidden.detach()
            if getattr(self, "keyed", None) == "logit":
                # A53 writes: value = the token's own IDENTITY under
                # its preceding-context key. Two passes (A38): recon
                # pass with live params feeds THIS chunk's backward;
                # store pass with cloned params feeds the NEXT
                # chunk's read-success backward, version-safe.
                V_id = lg_E[tokens]
                for pass2 in (False, True):
                    tu = self.tok_u.clone() if pass2 else self.tok_u
                    qm = self.qmix.clone() if pass2 else self.qmix
                    lgtw = qm + tu[wtokw]
                    lgtw = lgtw.masked_fill(
                        (relw < 0).unsqueeze(0), float("-inf"))
                    lgtw = lgtw.masked_fill(allw.view(1, -1, 1), 0.0)
                    mixw = torch.softmax(lgtw, dim=-1)
                    k_d = torch.einsum("btr,btrd->btd", mixw,
                                       lg_E[wtokw])
                    sv = torch.sigmoid(tu[tokens]) * lg_smask
                    for k in self.bands:
                        stn = self.stores[str(k)]
                        M_in = st["M"][k].detach()
                        if not pass2:
                            _, rc = stn.write(M_in, stn.lift(k_d),
                                              V_id, sv)
                            recon.append(rc)
                        else:
                            st["M"][k], _ = stn.write(
                                M_in, stn.lift(k_d), V_id, sv,
                                stale_ok=True)
            elif getattr(self, "keyed", None) == "token":
                # A52: one pair per POSITION, key = the token's own
                # unit embedding, write strength from tok_u. The A38
                # two-pass credit structure is preserved: the recon
                # pass feeds THIS chunk's backward (tok_u learns
                # write-fidelity), the stale_ok pass with cloned
                # tok_u feeds the NEXT chunk's read-success backward.
                E = nn.functional.normalize(
                    self.embed.weight, dim=-1).detach()
                Kt = E[tokens]
                s = torch.sigmoid(self.tok_u[tokens])
                s2 = torch.sigmoid(self.tok_u.clone()[tokens])
                for k in self.bands:
                    M_in = st["M"][k].detach()
                    _, rc = self.mats[str(k)].write_keyed(
                        M_in, Kt, h_wr, s)
                    recon.append(rc)
                    st["M"][k], _ = self.mats[str(k)].write_keyed(
                        M_in, Kt, h_wr, s2, stale_ok=True)
            else:
                for k in self.bands:
                    M_in = st["M"][k].detach()
                    w = torch.softmax(
                        h_wr @ self.write_q[str(k)] / self.d ** 0.5,
                        dim=1)
                    sel = torch.einsum("bt,btd->bd", w, h_wr)
                    _, rc = self.mats[str(k)].write(M_in, sel)
                    recon.append(rc)
                    w2 = torch.softmax(
                        h_wr @ self.write_q[str(k)].clone()
                        / self.d ** 0.5, dim=1)
                    sel2 = torch.einsum("bt,btd->bd", w2, h_wr)
                    st["M"][k], _ = self.mats[str(k)].write(
                        M_in, sel2, stale_ok=True)
            self._recon = torch.stack(recon).mean()
        return logits, st, ticks

    def pop_write_cost(self):
        c = self._write_cost
        self._write_cost = None
        return c

    def pop_recon(self):
        c = self._recon
        self._recon = None
        return c

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
