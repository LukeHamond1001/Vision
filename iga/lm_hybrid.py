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
                 use_xl=True, gate_init=-4.0, read_drop=0.5):
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
            # A30: reads gated shut at init (sigmoid(-4) ~ 0.018) —
            # v5.6 proved ungated per-position reads crowd out
            # induction formation; the model must opt in. gate_init
            # and read_drop are v6.2 bootstrap knobs (A39): defaults
            # reproduce v6.0/v6.1 exactly.
            self.read_gate = nn.ParameterDict(
                {str(k): nn.Parameter(torch.tensor(float(gate_init)))
                 for k in self.bands})
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
            st["M"] = {k: torch.zeros(B, self.d, self.d, device=device)
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
            if self.store == "matrix" and i == self.mid - 1 and read_ok:
                # per-position associative reads from LAST chunks'
                # matrices, gated shut at init (A30) + read-dropout
                # (A36: the crowding-out law — half the chunks train
                # storeless so induction must form)
                text = x[:, M:]
                r = torch.zeros_like(text)
                for k in self.bands:
                    if k in self.lesioned:
                        continue
                    g = torch.sigmoid(self.read_gate[str(k)])
                    r = r + g * self.mats[str(k)].read(st["M"][k], text)
                x = torch.cat([x[:, :M], text + r], dim=1)
        st["xl"] = new_xl if self.use_xl else None
        hidden = x[:, M:]                        # text positions
        logits = self.head(self.lnf(hidden))
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
            for k in self.bands:
                M_in = st["M"][k].detach()
                w = torch.softmax(
                    h_wr @ self.write_q[str(k)] / self.d ** 0.5, dim=1)
                sel = torch.einsum("bt,btd->bd", w, h_wr)
                _, rc = self.mats[str(k)].write(M_in, sel)
                recon.append(rc)
                w2 = torch.softmax(
                    h_wr @ self.write_q[str(k)].clone() / self.d ** 0.5,
                    dim=1)
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
