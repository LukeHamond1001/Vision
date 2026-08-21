"""The one-token organism (2026-08-21, the user's design).

The cortex sees ONLY the token that just arrived. Nothing before it
reaches the trunk except through the organs:

  PFC = the bands — a clocked ladder of recurrent states (band 3 ticks
        every token: the working memory of the last few words; bands
        4..K on the ladder in TOKENS), offered to the trunk as attended
        memory slots at every layer (keys/values only — the deep blocks
        transform the token slot alone, which keeps the cost near a
        plain transformer's);
  hippocampus = the content-keyed identity store (LogitStore), whose
        read enters as its own slot AND adds to the logits (the exact
        recall path).

Between tokens the whole COUNCIL — the token's hidden, the store read,
every band — talks through a shallow exchange (full attention among the
slots); each band's write is gated by its own gate and by VETOES from
the other bands; the council's token slot speaks.

Per token t (batched over lanes), order="cortex_first" (the brain's
order: input -> sensory cortex conditioned top-down by PFC -> PFC ->
decision):
    e_t     = embed(x_t)
    c_t     = trunk(e_t | slots [m_3..m_K, r_t])
    S'_t    = council([c_t, r_t, m_3..m_K])
    logits_t = head(lnf(c'_t)) + alpha-weighted store read of q(c'_t)
    r_{t+1} = store_in(read_t)                  the hippocampus slot
    band k, every clock_k tokens: pooled = mean of its council slot
        over the interval; h_k <- (1-g)h_k + g tanh(W[h_k; pooled]),
        g = sigmoid(z) * prod_{j!=k}(1 - veto_{j->k});
        pend_k = pred_k(h_k); fid_k = cos(pend_k(prev), pooled_C - mu_k)
        (the band listens to its council slot; it predicts the cortex
        stream C pooled over its interval, centred per band)
order="pfc_first" (the user's design, ratified 2026-08-21: all input
goes into the PFC; the PFC outputs one bundle of embeddings; the rest
of the neocortex makes the decision): the council (PFC) runs on the
raw embedding with the band and hippocampus slots, the bands write,
and the trunk (neocortex) queries with the council's token slot over
the WHOLE council bundle S'_t as its key/value slots at every layer —
it never sees e_t directly.

Precision law (the user's: bf16 for the neocortex, 32 bit for the
PFC): only the trunk blocks run under bf16 autocast; the council, the
cells, the vetoes, the predictors, the band states, the store, the head
and the loss are fp32 in both orders.

Store writes once per chunk (T tokens): key_t = proj(c'_{t-1}), value
= identity of x_t, the A38 two-pass credit unchanged; reads see the
previous chunks' M — the bands carry the current chunk. Credit: full
backprop through every state within the chunk; across the boundary
the last write of every band and its pend ride as cloned-parameter
graphs on detached inputs (the A38 pattern), as does the store's one
write-op.

Same forward API as HybridLM: (tokens, st, _) -> (logits, st, ticks);
lesioned / store_read_off / mem_off mean the same things. Clocks are
in TOKENS (clock_unit = "token"); horizons = clock.
"""

import torch
import torch.nn as nn

from .lm_bands import N_BANDS
from .lm_hybrid import LogitStore
from .lm_transformer import Block, make_mlp

SCAN_CLOCKS = {3: 1, 4: 8, 5: 64, 6: 512, 7: 4096, 8: 32768}   # tokens


class ScanBlock(nn.Module):
    """Pre-LN block whose attention QUERY is the token slot alone;
    keys/values are the token slot and the organ slots."""

    def __init__(self, d, heads, mlp="gelu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = make_mlp(d, mlp)

    def forward(self, x, slots):
        h = self.ln1(x)                              # [B, 1, d]
        if slots is not None:
            kv = torch.cat([h, self.ln1(slots)], dim=1)
        else:
            kv = h
        a, _ = self.attn(h, kv, kv, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class ScanCell(nn.Module):
    """Gated delta write h' = (1-g) h + g tanh(W[h;x]); g = sigmoid(z)
    times the council's permission p (product of the other bands'
    non-vetoes). cloned=True computes the same values through cloned
    parameters (A38): the graph survives the optimizer's in-place
    update so the NEXT chunk's backward can credit this write."""

    def __init__(self, d, gate_bias):
        super().__init__()
        self.z = nn.Linear(2 * d, d)
        self.cand = nn.Linear(2 * d, d)
        nn.init.constant_(self.z.bias, gate_bias)

    def forward(self, x, h, p=None, cloned=False):
        hx = torch.cat([h, x], dim=-1)
        if cloned:
            z = torch.sigmoid(nn.functional.linear(
                hx, self.z.weight.clone(), self.z.bias.clone()))
            c = torch.tanh(nn.functional.linear(
                hx, self.cand.weight.clone(), self.cand.bias.clone()))
        else:
            z = torch.sigmoid(self.z(hx))
            c = torch.tanh(self.cand(hx))
        g = z if p is None else z * p
        return (1 - g) * h + g * c, g


class ScanLM(nn.Module):
    def __init__(self, vocab_size, d=128, n_layers=6, n_heads=8, max_T=64,
                 clocks=None, n_council=2, gate_init=-2.0,
                 fast_gate=None, veto=True, read_drop=0.5, aux_trunk=0.0,
                 mlp="gelu", band_center=True, order="cortex_first",
                 kd_base=1, slot_every=8, write_every=4, kd_max=4096):
        super().__init__()
        assert order in ("cortex_first", "pfc_first")
        self.order = order
        self.d = d
        self.vocab_size = vocab_size
        self.max_T = max_T
        self.clocks = dict(SCAN_CLOCKS if clocks is None else clocks)
        self.clock_unit = "token"
        self.bands = sorted(self.clocks)
        self.band_w = {k: d for k in self.bands}
        # contract with the trainer / battery / serve room
        self.store, self.keyed = "matrix", "hidden"
        self.band_credit, self.band_center, self.tail_tokens = True, bool(band_center), 0
        self.use_xl = False
        self.attn_kind, self.qk_norm, self.mlp_kind = "scan", False, mlp
        self.autocast_bf16 = False
        self.read_drop = read_drop
        self.aux_trunk = float(aux_trunk)
        self._aux_hidden = None
        if self.aux_trunk > 0:
            self.aux_head = nn.Linear(d, vocab_size)
        self.veto = bool(veto)
        # the council's hippocampus slot refreshes every slot_every
        # tokens (a phrase): a per-token read over every store is a
        # full pass over M per token (measured 5x the trunk at d=128;
        # ~30% of the trunk at 500M with 8). The logit-path read is
        # per token, batched after the scan (it does not feed back).
        self.slot_every = max(1, int(slot_every))
        # stores are written every write_every chunks (the write is a
        # fixed number of passes over M per write, so short chunks
        # amortize it over write_every x T tokens); the keys of the
        # chunks in between are buffered detached, the live chunk's
        # keys carry the recon credit. Reads lag the stream by at most
        # write_every x T tokens — the bands' duty.
        self.write_every = max(1, int(write_every))
        self.TAIL_W = 0

        self.embed = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList(
            [ScanBlock(d, n_heads, mlp=mlp) for _ in range(n_layers)])
        self.council = nn.ModuleList(
            [Block(d, n_heads, mlp=mlp) for _ in range(n_council)])
        # slot tags: 0 = the token, 1 = the hippocampus read, 2.. = bands
        self.slot = nn.Embedding(2 + len(self.bands), d)
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)
        self.store_in = nn.Linear(d, d, bias=False)       # read -> slot
        nn.init.zeros_(self.store_in.weight)               # silent at init
        # the bands. Gate init per band: the per-token band must be an
        # open recurrence (a closed gate shrinks gradient by (1-g) per
        # token: 0.88^64 ~ 3e-4 across a chunk), the slow bands keep
        # the certified closed-by-default gate.
        fast = {3: 1.0, 4: 0.0} if fast_gate is None else dict(fast_gate)
        self.cells = nn.ModuleDict(
            {str(k): ScanCell(d, fast.get(k, gate_init)) for k in self.bands})
        self.pred = nn.ModuleDict(
            {str(k): nn.Linear(d, d) for k in self.bands})
        self.mem_proj = nn.ModuleDict(
            {str(k): nn.Linear(d, d, bias=False) for k in self.bands})
        if self.veto:
            # veto_{j->k} = sigmoid(S'_j . w_k + b): closed at init
            self.veto_w = nn.ParameterDict(
                {str(k): nn.Parameter(torch.randn(d) / d ** 0.5)
                 for k in self.bands})
            self.veto_b = nn.ParameterDict(
                {str(k): nn.Parameter(torch.tensor(-4.0)) for k in self.bands})
        # band_center, PER BAND (2026-08-21 fix): each band's fidelity
        # target is its own council slot pooled over the interval, so the
        # centring mean must be that slot's running mean — one row per
        # band, seeded by the first tick, EMA 0.99 per tick afterwards.
        # (A single mean of the cortex output left a per-band constant in
        # the target and saturated fid at +0.999: no learning signal.)
        nb = max(self.bands) + 1
        self.register_buffer("band_mu", torch.zeros(nb, d))
        self.register_buffer("band_mu_n", torch.zeros(nb))
        # the hippocampus: the certified content-keyed identity store,
        # capacity per band doubling up the ladder; half-life in WRITES
        # (one write per chunk) = the band's clock in chunks
        self.KD = {k: min(int(kd_max), 512 * (2 ** i) * kd_base)
                   for i, k in enumerate(self.bands)}
        self.key_proj = nn.Linear(d, d, bias=False)
        self.query_proj = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.key_proj.weight)
        nn.init.eye_(self.query_proj.weight)
        self.tok_u = nn.Parameter(torch.zeros(vocab_size))
        self.stores = nn.ModuleDict()
        for k in self.bands:
            hl_writes = max(1.0, self.clocks[k] / (max_T * self.write_every))
            self.stores[str(k)] = LogitStore(
                d, self.KD[k], 1 - 0.5 ** (1 / hl_writes), seed=1000 + k)
        self.alpha = nn.ParameterDict(
            {str(k): nn.Parameter(torch.tensor(0.0)) for k in self.bands})
        self.read_gate = nn.ParameterDict(
            {str(k): nn.Parameter(torch.tensor(0.0), requires_grad=False)
             for k in self.bands})                      # battery compat
        self.lesioned = set()
        self.store_read_off = False
        self.mem_off = False
        self._write_cost = None
        self._recon = None
        self._veto_mean = {}

    # ---------------- state ----------------
    def init_state(self, B, device):
        z = lambda: torch.zeros(B, self.d, device=device)
        return {"h": {k: z() for k in self.bands},
                "acc": {k: z() for k in self.bands},
                "acc_c": {k: z() for k in self.bands},    # the cortex stream
                "cnt": {k: 0 for k in self.bands},
                "pend": {k: None for k in self.bands},
                "fresh": {k: False for k in self.bands},
                "M": {k: torch.zeros(B, self.d, self.KD[k], device=device)
                      for k in self.bands},
                "lp": {}, "lh": {}, "lg": {},      # last write: pooled, h before, permission
                "wbuf": [],                        # (h_prev, tokens, smask) since the last write
                "prev_c": z(), "tail": z(), "tok": 0, "chunk": 0, "xl": None}

    def detach_state(self, st):
        # a band that wrote in this chunk keeps its one-op cloned graph
        # for exactly the next chunk (the A38 pattern); pend likewise
        st["h"] = {k: (v if st["fresh"].get(k) else v.detach())
                   for k, v in st["h"].items()}
        st["fresh"] = {k: False for k in st["h"]}
        st["acc"] = {k: v.detach() for k, v in st["acc"].items()}
        st["acc_c"] = {k: v.detach() for k, v in st["acc_c"].items()}
        st["prev_c"] = st["prev_c"].detach()
        st["tail"] = st["tail"].detach()
        # M keeps one write-op of graph for exactly the first chunk
        # that reads it (A38); that chunk's backward consumes the
        # graph, so every later chunk before the next write reads a
        # detached M
        if st.get("M_fresh"):
            st["M_fresh"] = False
        else:
            st["M"] = {k: v.detach() for k, v in st["M"].items()}
        return st

    # ---------------- helpers ----------------
    def _band_slots(self, st, B):
        toks = []
        for k in self.bands:
            h = st["h"][k]
            if k in self.lesioned or self.mem_off:
                h = torch.zeros_like(h)
            toks.append(self.mem_proj[str(k)](h))
        return torch.stack(toks, dim=1)                 # [B, K, d]

    def _council(self, c, r, m, dev):
        """c [B,d] token hidden, r [B,d] store slot, m [B,K,d] bands ->
        S' [B, 2+K, d] after the exchange."""
        S = torch.cat([c.unsqueeze(1), r.unsqueeze(1), m], dim=1)
        S = S + self.slot.weight[None, : S.shape[1]]
        # the PFC is fp32 always (precision law); autocast is switched
        # OFF here so a bf16 trunk never pulls the council with it
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                            enabled=False):
            for b in self.council:
                S = b(S, None)
        return S

    def _trunk(self, e, slots, dev):
        x = e.unsqueeze(1)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                            enabled=self.autocast_bf16):
            for b in self.blocks:
                x = b(x, slots)
        x = x.float() if self.autocast_bf16 else x
        return x[:, 0]

    def _read(self, st, q_hidden, read_ok):
        """the hippocampus read: identity-space vectors from the
        previous chunks' M, alpha-weighted; q_hidden [B, d] (one token)
        or [B, T, d] (the chunk, batched); None when reads are off."""
        if not read_ok or self.store_read_off:
            return None
        one = q_hidden.dim() == 2
        q = nn.functional.normalize(self.query_proj(q_hidden), dim=-1)
        if one:
            q = q.unsqueeze(1)
        rsum = None
        for k in self.bands:
            if k in self.lesioned:
                continue
            stn = self.stores[str(k)]
            r = self.alpha[str(k)] * stn.read(st["M"][k], stn.lift(q))
            rsum = r if rsum is None else rsum + r
        if rsum is not None and one:
            rsum = rsum.squeeze(1)
        return rsum

    # ---------------- forward ----------------
    def forward(self, tokens, st, scene_starts=None):
        B, T = tokens.shape
        dev = tokens.device
        K = len(self.bands)
        read_ok = (not self.training) or float(torch.rand(())) >= self.read_drop
        self._reads_used = read_ok
        lg_E = nn.functional.normalize(self.embed.weight, dim=-1).detach()
        ticks = [[] for _ in range(max(N_BANDS, max(self.bands) + 1))]
        wcost, vetoes = [], {k: [] for k in self.bands}
        emb = self.embed(tokens)                         # [B, T, d]
        for key in ("lp", "lh", "lg"):
            st.setdefault(key, {})
        st.setdefault("wbuf", [])
        m = self._band_slots(st, B)
        rd0 = self._read(st, st["prev_c"], read_ok) if st["chunk"] > 0 else None
        r_slot = self.store_in(rd0) if rd0 is not None \
            else torch.zeros(B, self.d, device=dev)
        cs, s0s, bundles, tick_log = [], [], [], []
        seg_start = {k: 0 for k in self.bands}   # chunk-local start of each band's open interval
        for t in range(T):
            e = emb[:, t]
            if self.order == "cortex_first":
                c = self._trunk(e, torch.cat([m, r_slot.unsqueeze(1)], 1), dev)
                S = self._council(c, r_slot, m, dev)
                cs.append(S[:, 0])
            else:
                # PFC first: the council deliberates on the raw token; the
                # neocortex is OFF the recurrent path and decodes the whole
                # chunk's bundles at once after the loop (same math, one
                # batched call — the per-token decoder was launch-bound)
                S = self._council(e, r_slot, m, dev)
                bundles.append(S[:, 1:])
            s0 = S[:, 0]
            s0s.append(s0)
            if (t + 1) % self.slot_every == 0 and t + 1 < T:
                # the hippocampus, queried by what the council concluded
                rd = self._read(st, s0, read_ok)
                r_slot = self.store_in(rd) if rd is not None \
                    else torch.zeros_like(s0)
            # the bands LISTEN through their council slots every token and
            # must PREDICT the cortex stream (the fidelity target = the
            # neocortex output pooled over the interval — input-driven,
            # the certified hybrid's target; a band's own council slot is
            # mostly an echo of its state and was self-predictable, fid
            # saturating at +0.98 even centred per band). The fidelity is
            # scored after the decoder has run (C is not known in the loop)
            st["tok"] += 1
            ticked = False
            for i, k in enumerate(self.bands):
                s_k = S[:, 2 + i]
                st["acc"][k] = st["acc"][k] + s_k
                st["cnt"][k] += 1
                if st["tok"] % self.clocks[k] == 0:
                    pooled = st["acc"][k] / max(st["cnt"][k], 1)
                    if st["pend"][k] is not None:
                        tick_log.append((t, k, st["pend"][k], seg_start[k],
                                         st["cnt"][k]))
                    seg_start[k] = t + 1
                    p = None
                    if self.veto and K > 1:
                        others = torch.cat([S[:, 2 + j] for j in range(K) if j != i], 0)
                        v = torch.sigmoid(others @ self.veto_w[str(k)]
                                          + self.veto_b[str(k)])
                        v = v.view(K - 1, B)
                        p = torch.prod(1 - v, dim=0).unsqueeze(-1)  # [B, 1]
                        vetoes[k].append(v.detach().mean())
                    cell = self.cells[str(k)]
                    st["lp"][k] = pooled.detach()
                    st["lh"][k] = st["h"][k].detach()
                    st["lg"][k] = p.detach() if p is not None else None
                    h_new, g = cell(pooled, st["h"][k], p)
                    st["h"][k] = h_new
                    st["fresh"][k] = True
                    st["pend"][k] = self.pred[str(k)](h_new)
                    if self.clocks[k] > 1:
                        wcost.append(g.mean())
                    st["acc"][k] = torch.zeros_like(st["acc"][k])
                    st["cnt"][k] = 0
                    ticked = True
            if ticked:
                m = self._band_slots(st, B)
        S0 = torch.stack(s0s, dim=1)                       # [B, T, d] the PFC's conclusions
        if self.order == "cortex_first":
            C = torch.stack(cs, dim=1)                     # [B, T, d]
        else:
            Bd = torch.stack(bundles, dim=1)               # [B, T, 1+K, d]
            C = self._trunk(S0.reshape(B * T, self.d),
                            Bd.reshape(B * T, Bd.shape[2], self.d), dev)
            C = C.reshape(B, T, self.d)
        # ---- band fidelity: pend (from the previous tick) vs the cortex
        # stream pooled over the interval, centred per band; intervals
        # that began in earlier chunks carry their partial sum in acc_c
        for (t, k, pend, s_from, cnt) in tick_log:
            pooled_c = (st["acc_c"][k] + C[:, s_from:t + 1].sum(1)) / max(cnt, 1)
            target = pooled_c
            if self.band_center:
                # TRAINING centres by the batch mean at this very tick:
                # a lagging running mean leaves the training DRIFT of the
                # cortex mean in the residual — one direction shared by
                # every lane, matched by a predictor bias (scan1 measured
                # fid:5 = +1.000 at step 1300). The instantaneous mean
                # cancels it exactly; what remains is how THIS lane's
                # context differs from the others — the signal a band
                # can only carry by remembering. Eval/serve (B may be 1,
                # no drift) centre by the per-band running mean.
                if self.training and B > 1:
                    target = pooled_c - pooled_c.detach().mean(0, keepdim=True)
                else:
                    target = pooled_c - self.band_mu[k]
                if self.training:
                    with torch.no_grad():
                        pm = pooled_c.detach().mean(0)
                        if self.band_mu_n[k] == 0:
                            self.band_mu[k] = pm
                        else:
                            self.band_mu[k].mul_(0.99).add_(0.01 * pm)
                        self.band_mu_n[k] += 1
            fid = nn.functional.cosine_similarity(pend, target, dim=-1)
            ticks[k].append((t, fid))
            st["acc_c"][k] = torch.zeros_like(st["acc_c"][k])
        for k in self.bands:                               # the open intervals' partial sums
            if seg_start[k] < T:
                st["acc_c"][k] = st["acc_c"][k] + C[:, seg_start[k]:].sum(1)
        logits = self.head(self.lnf(C))
        self._aux_hidden = None
        # the hippocampus is a PFC organ: keyed and queried by the
        # council's token slot (in cortex_first that is the cortex output)
        R = self._read(st, S0, read_ok)                   # batched, [B, T, d]
        if R is not None:
            if self.aux_trunk > 0 and self.training:
                self._aux_hidden = C
            logits = logits + R @ lg_E.t()
        if wcost:
            self._write_cost = torch.stack(wcost).mean()
        self._veto_mean = {k: float(torch.stack(v).mean()) for k, v in vetoes.items() if v}
        # ---- the carried band graphs (A38): the last write of each band
        # that ticked, recomputed through cloned params on detached
        # inputs, so the next chunk's CE and fidelity can credit it
        # without touching this chunk's graph
        for k in self.bands:
            if st["fresh"][k] and k in st["lp"]:
                cell = self.cells[str(k)]
                h_c, _ = cell(st["lp"][k], st["lh"][k], st["lg"][k], cloned=True)
                st["h"][k] = h_c
                # pend through its OWN cloned graph: h's graph is consumed
                # by the next chunk's backward, pend's by the band's next
                # tick, which may be chunks later
                h_c2, _ = cell(st["lp"][k], st["lh"][k], st["lg"][k], cloned=True)
                pr = self.pred[str(k)]
                st["pend"][k] = nn.functional.linear(
                    h_c2, pr.weight.clone(), pr.bias.clone())
        # ---- store writes, once per chunk: key_t = proj(s'_{t-1}) — the
        # PFC's conclusion about the previous token — value = identity of
        # x_t; recon pass live, store pass cloned
        h_prev_live = torch.cat([st["prev_c"].unsqueeze(1), S0[:, :-1]], dim=1)
        h_prev_det = h_prev_live.detach()
        smask = torch.ones(B, T, device=dev)
        if st["chunk"] == 0:
            smask[:, 0] = 0.0                              # nothing before
        st["prev_c"] = S0[:, -1].detach()
        st["chunk"] += 1
        if st["chunk"] % self.write_every != 0:
            st["wbuf"].append((h_prev_det, tokens, smask))
            self._recon = None
            return logits, st, ticks
        buf = st["wbuf"]; st["wbuf"] = []
        toks_all = torch.cat([b[1] for b in buf] + [tokens], dim=1)
        sm_all = torch.cat([b[2] for b in buf] + [smask], dim=1)
        hp_det = torch.cat([b[0] for b in buf] + [h_prev_det], dim=1)
        hp_live = torch.cat([b[0] for b in buf] + [h_prev_live], dim=1)
        V_id = lg_E[toks_all]
        recon = []
        for pass2 in (False, True):
            tu = self.tok_u.clone() if pass2 else self.tok_u
            Wk = self.key_proj.weight.clone() if pass2 else self.key_proj.weight
            hp = hp_det if pass2 else hp_live
            k_d = nn.functional.normalize(hp @ Wk.t(), dim=-1)
            sv = torch.sigmoid(tu[toks_all]) * sm_all
            for k in self.bands:
                stn = self.stores[str(k)]
                M_in = st["M"][k].detach()
                if not pass2:
                    _, rc = stn.write(M_in, stn.lift(k_d), V_id, sv)
                    recon.append(rc)
                else:
                    st["M"][k], _ = stn.write(M_in, stn.lift(k_d), V_id, sv,
                                              stale_ok=True)
        self._recon = torch.stack(recon).mean()
        st["M_fresh"] = True
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
