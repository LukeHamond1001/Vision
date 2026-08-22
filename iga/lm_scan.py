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


def _lift_read(M, q, proj, phase, alpha):
    """one band's hippocampus read: alpha * M . lift(q); M [B, d, D],
    q [B, T, d] unit queries, proj [D, d], phase [D] -> [B, T, d].
    Same math as LogitStore.lift + LogitStore.read (a free function so
    torch.compile can fuse the elementwise tail; see compile_read)."""
    k = nn.functional.normalize(torch.cos(nn.functional.linear(q, proj) + phase), dim=-1)
    return alpha * torch.einsum("bij,btj->bti", M, k)


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
                 kd_base=1, slot_every=8, write_every=4, kd_max=4096,
                 compile_council=False, compile_mode="default", compile_read=False,
                 store_exact=False, register=None, reward_slot=False, value_gamma=0.9,
                 dopamine=0.0, bg_w=0.0, dopamine_band=None):
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
        # REGISTERS (2026-08-22): register={band: n} gives a band n
        # working-memory slots — n units, each with its own cell, gate,
        # veto, predictor and council slot, all ticking on the band's
        # clock; lesioning the band removes every unit. Unit keys: the
        # band int for the first unit (bit-exact with register=None
        # everywhere), "<k>r<j>" for the extras. The hippocampus stays
        # per band. Band 3 carried +1.79 of the bands' +1.95 nats at 15M
        # tokens as a single 512-vector — this is its capacity lever.
        self.register = {int(k): int(v) for k, v in (register or {}).items()}
        self.units = [(k, j) for k in self.bands for j in range(self.register.get(k, 1))]
        self.ukeys = [k if j == 0 else f"{k}r{j}" for k, j in self.units]
        self.unit_band = {u: k for u, (k, j) in zip(self.ukeys, self.units)}
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
        # PHASE 2 (2026-08-22, the user's design): rewards grounded in the
        # PFC. reward_slot adds a council slot fed by the press token at
        # this position (none / +1 / +2 / -1 / -2: a 5-level embedding) —
        # input only, never a target, seen by the decoder only through
        # the PFC bundle. Value heads V_u(h_u) per unit are trained by TD
        # across the ladder at each unit's own tick cadence:
        #   V(h_prev_tick) -> R(rewards since that tick) + gamma V(h_now)
        # (target detached), gamma per TICK, so band 3 values the next
        # ~10 tokens and band 8 the next ~10 hours — secondary reinforcers
        # form at each timescale. The TD gradient flows into the band
        # states (the PFC learns to carry value); pop_value_loss() hands
        # the trainer the loss (VALUE_W, default 0 = off, bit-exact).
        self.reward_slot = bool(reward_slot)
        self.n_fixed = 3 if reward_slot else 2           # token, [reward,] hippocampus
        self.slot = nn.Embedding(self.n_fixed + len(self.units), d)
        self.reward_emb = nn.Embedding(5, d)
        nn.init.zeros_(self.reward_emb.weight)            # silent at init
        self.register_buffer("reward_lut", torch.zeros(vocab_size, dtype=torch.long))
        self.register_buffer("reward_val", torch.tensor([0.0, 1.0, 2.0, -1.0, -2.0]))
        self.value_gamma = float(value_gamma)
        self._value_loss = None
        self._bg_loss = None
        self._gate_trace = None
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
            {str(u): ScanCell(d, fast.get(self.unit_band[u], gate_init)) for u in self.ukeys})
        self.pred = nn.ModuleDict(
            {str(u): nn.Linear(d, d) for u in self.ukeys})
        self.value = nn.ModuleDict(
            {str(u): nn.Linear(d, 1) for u in self.ukeys})
        for head in self.value.values():                  # V = 0 at birth: the RPE IS the reward
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        # DOPAMINE (Phase 2, step 2): the reward prediction error of the
        # per-token units (band 3: delta_t = R_t + gamma V(h_now) -
        # V(h_prev), detached) scales the hippocampus's write strength
        # at that token, s_t <- min(1, s_t (1 + kappa |delta_t|)) — a
        # surprising reward burns the episode in harder; an expected one
        # does nothing. kappa = 0 (default) is exact.
        self.dopamine = float(dopamine)
        # which band's RPE is the dopamine: None = the per-token (clock
        # 1) units, whose 10-token horizon cannot predict a press ~740
        # tokens away, so every press stays a full surprise (scan8:
        # value AUC .49 at band 3, .56/.71 at bands 5/6). A slower
        # band's tick closes an interval: its |delta| stamps every
        # token of that interval (the chunk's writes, when the clock
        # divides T) — reward that the band predicted no longer fires.
        self.dopamine_band = None if dopamine_band is None else int(dopamine_band)
        # Basal ganglia (the actor; the value heads are the critic): the
        # band gates g = sigmoid(z) * permission learn from the reward
        # prediction error at their own tick, not only from BPTT inside
        # the chunk — delta > 0 (the update led somewhere better than
        # expected) pulls the gate that made it toward open, delta < 0
        # toward shut, |delta|-weighted (PBWM Go/NoGo; a DA burst
        # strengthens Go, a dip NoGo). The regression runs on a gate
        # recomputed from DETACHED inputs, so the force lands on the
        # gate and veto parameters only — never on the council (the
        # scan2 lesson). bg_w = 0 (default) is exact.
        self.bg_w = float(bg_w)
        # eval-time lesions for the battery (exact when False)
        self.reward_off = False          # the reward slot zeroed
        self.veto_off = False            # permissions = 1 (no lateral vetoes)
        self.windowless = True           # tokens stream; max_T is the batching unit, not a window
        self.mem_proj = nn.ModuleDict(
            {str(u): nn.Linear(d, d, bias=False) for u in self.ukeys})
        if self.veto:
            # veto_{j->k} = sigmoid(S'_j . w_k + b): closed at init
            self.veto_w = nn.ParameterDict(
                {str(u): nn.Parameter(torch.randn(d) / d ** 0.5)
                 for u in self.ukeys})
            self.veto_b = nn.ParameterDict(
                {str(u): nn.Parameter(torch.tensor(-4.0)) for u in self.ukeys})
        # band_center, PER BAND (2026-08-21 fix): each band's fidelity
        # target is its own council slot pooled over the interval, so the
        # centring mean must be that slot's running mean — one row per
        # band, seeded by the first tick, EMA 0.99 per tick afterwards.
        # (A single mean of the cortex output left a per-band constant in
        # the target and saturated fid at +0.999: no learning signal.)
        nb = max(self.bands) + 1
        self.register_buffer("band_mu", torch.zeros(nb, d))
        self.register_buffer("band_mu_n", torch.zeros(nb))
        # batched band bookkeeping (2026-08-22): the six bands' per-token
        # work runs as single tensor ops on [B, K, d] — one add for the
        # accumulators, one bmm for the slots, one matmul for every veto —
        # instead of six Python-level passes (the step was launch-bound:
        # ~1000 kernels per token, scan1/scan2 at 2.2-3.4k tok/s)
        self.register_buffer("band_idx", torch.arange(len(self.units)))
        self.register_buffer("veto_eye", torch.eye(len(self.units)))
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
        self.store_exact = bool(store_exact)
        for stn in self.stores.values():
            stn.exact = self.store_exact
        self.lesioned = set()
        self.store_read_off = False
        self.mem_off = False
        self._write_cost = None
        self._recon = None
        self._veto_mean = {}
        # opt-in torch.compile of the council's blocks — the one fixed-
        # shape, control-flow-free function called 64x per chunk; on
        # CUDA the fused kernels (and, with mode="reduce-overhead",
        # graph replay) cut the launch count the step is bound by.
        # Default off = the eager path, bit-exact. (2026-08-22)
        self.compile_council = bool(compile_council)
        if compile_council or compile_read:
            # the read is compiled per band shape x batch size (6 bands x
            # train/eval/collapse batches > dynamo's default 8 variants,
            # after which it silently falls back to eager — scan6's
            # heartbeat, 2026-08-22)
            from torch import _dynamo as _dyn
            _dyn.config.cache_size_limit = max(_dyn.config.cache_size_limit, 64)
        self._council_fn = (torch.compile(self._council_blocks, dynamic=False, mode=compile_mode)
                            if compile_council else self._council_blocks)
        # same treatment for one band's hippocampus read (lift + read +
        # alpha: ~9 small ops per band per token at slot_every=1)
        self.compile_read = bool(compile_read)
        self._lift_read_fn = (torch.compile(_lift_read, dynamic=False, mode=compile_mode)
                              if compile_read else _lift_read)

    # ---------------- state ----------------
    def init_state(self, B, device):
        z = lambda: torch.zeros(B, self.d, device=device)
        return {"h": {u: z() for u in self.ukeys},
                "acc": {u: z() for u in self.ukeys},
                "acc_c": {u: z() for u in self.ukeys},    # the cortex stream
                "cnt": {u: 0 for u in self.ukeys},
                "pend": {u: None for u in self.ukeys},
                "fresh": {u: False for u in self.ukeys},
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
        h_all = torch.stack([st["h"][u] for u in self.ukeys], dim=1)   # [B, U, d]
        return self._slots_from(h_all, self._mem_W())

    def _mem_W(self):
        # the unit projections stacked once per chunk: [U, d, d]
        return torch.stack([self.mem_proj[str(u)].weight for u in self.ukeys])

    def _slots_from(self, h_all, W):
        """h_all [B, K, d] band states -> slots [B, K, d] = h_k W_k^T,
        lesioned / mem_off bands silenced (same values as the per-band
        Linear path, in one bmm)."""
        if self.mem_off or self.lesioned:
            keep = torch.tensor([0.0 if (self.mem_off or self.unit_band[u] in self.lesioned) else 1.0
                                 for u in self.ukeys], device=h_all.device)
            h_all = h_all * keep[None, :, None]
        return torch.bmm(h_all.transpose(0, 1), W.transpose(1, 2)).transpose(0, 1)

    def _council(self, c, r, m, dev, rw=None):
        """c [B,d] token hidden, r [B,d] store slot, m [B,K,d] bands,
        rw [B,d] the reward slot (reward_slot only) -> S' [B, n_fixed+K, d]
        after the exchange."""
        parts = [c.unsqueeze(1)]
        if rw is not None:
            parts.append(rw.unsqueeze(1))
        parts += [r.unsqueeze(1), m]
        S = torch.cat(parts, dim=1)
        S = S + self.slot.weight[None, : S.shape[1]]
        # the PFC is fp32 always (precision law); autocast is switched
        # OFF here so a bf16 trunk never pulls the council with it
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                            enabled=False):
            S = self._council_fn(S)
        return S

    def _council_blocks(self, S):
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
            r = self._lift_read_fn(st["M"][k], q, stn.proj, stn.phase, self.alpha[str(k)])
            rsum = r if rsum is None else rsum + r
        if rsum is not None and one:
            rsum = rsum.squeeze(1)
        return rsum

    # ---------------- forward ----------------
    def forward(self, tokens, st, scene_starts=None):
        B, T = tokens.shape
        dev = tokens.device
        K = len(self.units)                      # council band slots = units
        read_ok = (not self.training) or float(torch.rand(())) >= self.read_drop
        self._reads_used = read_ok
        lg_E = nn.functional.normalize(self.embed.weight, dim=-1).detach()
        ticks = [[] for _ in range(max(N_BANDS, max(self.bands) + 1))]
        wcost, vetoes = [], {u: [] for u in self.ukeys}
        emb = self.embed(tokens)                         # [B, T, d]
        lev = self.reward_lut[tokens]                    # [B, T] press level per token
        rew = self.reward_val[lev]                       # [B, T] its value
        rew_cum = torch.cat([rew.new_zeros(B, 1), rew.cumsum(1)], dim=1)   # [B, T+1]
        rslots = self.reward_emb(lev) if self.reward_slot else None        # [B, T, d]
        if rslots is not None and self.reward_off:
            rslots = torch.zeros_like(rslots)
        gtr = []                                         # band-3 gate means per token (the BG trace)
        bg_on = self.bg_w > 0
        v_pairs = {u: [] for u in self.ukeys}            # (h_prev, R, h_now) per tick
        rpe = [None] * T                                 # |RPE| per token (dopamine), summed over clock-1 units
        v_from = {u: 0 for u in self.ukeys}              # chunk-local start of the open TD interval
        st.setdefault("R_carry", {u: rew.new_zeros(B) for u in self.ukeys})   # rewards since the last tick, earlier chunks
        nf = self.n_fixed
        for key in ("lp", "lh", "lg"):
            st.setdefault(key, {})
        st.setdefault("wbuf", [])
        W_mem = self._mem_W()
        h_all = torch.stack([st["h"][u] for u in self.ukeys], dim=1)    # [B, U, d]
        acc_all = torch.stack([st["acc"][u] for u in self.ukeys], dim=1)  # [B, U, d]
        if self.veto and K > 1:
            W_veto = torch.stack([self.veto_w[str(u)] for u in self.ukeys])   # [U, d]
            b_veto = torch.stack([self.veto_b[str(u)] for u in self.ukeys])   # [U]
            veto_sum = None
        m = self._slots_from(h_all, W_mem)
        rd0 = self._read(st, st["prev_c"], read_ok) if st["chunk"] > 0 else None
        r_slot = self.store_in(rd0) if rd0 is not None \
            else torch.zeros(B, self.d, device=dev)
        cs, s0s, bundles, tick_log = [], [], [], []
        seg_start = {u: 0 for u in self.ukeys}   # chunk-local start of each unit's open interval
        for t in range(T):
            e = emb[:, t]
            rw_t = rslots[:, t] if rslots is not None else None
            if self.order == "cortex_first":
                c = self._trunk(e, torch.cat([m, r_slot.unsqueeze(1)], 1), dev)
                S = (self._council(c, r_slot, m, dev) if rw_t is None
                     else self._council(c, r_slot, m, dev, rw_t))
                cs.append(S[:, 0])
            else:
                # PFC first: the council deliberates on the raw token; the
                # neocortex is OFF the recurrent path and decodes the whole
                # chunk's bundles at once after the loop (same math, one
                # batched call — the per-token decoder was launch-bound)
                S = (self._council(e, r_slot, m, dev) if rw_t is None
                     else self._council(e, r_slot, m, dev, rw_t))
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
            acc_all = acc_all + S[:, nf:]                    # every band listens, one add
            due = []
            for i, u in enumerate(self.ukeys):
                st["cnt"][u] += 1
                if st["tok"] % self.clocks[self.unit_band[u]] == 0:
                    due.append(i)
            if due:
                P = None
                P_bg = None
                if self.veto and K > 1 and not self.veto_off:
                    # every veto at once: v[b, j, k] = sigmoid(S'_j . w_k + b_k);
                    # permission_k = prod_{j != k} (1 - v[b, j, k])
                    v = torch.sigmoid(S[:, nf:] @ W_veto.t() + b_veto)     # [B, K, K]
                    P = torch.prod(1 - v * (1 - self.veto_eye), dim=1)     # [B, K]
                    vs = v.detach().sum(0)
                    veto_sum = vs if veto_sum is None else veto_sum + vs
                    for i in due:
                        vetoes[self.ukeys[i]].append(None)               # counted below
                    if bg_on:
                        # the same permissions from a detached council:
                        # the BG force reaches the veto weights, not S
                        v_bg = torch.sigmoid(S[:, nf:].detach() @ W_veto.t() + b_veto)
                        P_bg = torch.prod(1 - v_bg * (1 - self.veto_eye), dim=1)
                for i in due:
                    u = self.ukeys[i]; k = self.unit_band[u]
                    pooled = acc_all[:, i] / max(st["cnt"][u], 1)
                    if st["pend"][u] is not None:
                        tick_log.append((t, u, st["pend"][u], seg_start[u],
                                         st["cnt"][u]))
                    seg_start[u] = t + 1
                    p = P[:, i].unsqueeze(-1) if P is not None else None   # [B, 1]
                    cell = self.cells[str(u)]
                    st["lp"][u] = pooled.detach()
                    st["lh"][u] = st["h"][u].detach()
                    st["lg"][u] = p.detach() if p is not None else None
                    h_prev = st["h"][u]
                    h_new, g = cell(pooled, h_prev, p)
                    st["h"][u] = h_new
                    st["fresh"][u] = True
                    # the TD pair for this tick: the state held since the
                    # last tick (live, or the carried one-op graph at the
                    # chunk start) predicts the rewards of the interval
                    # plus the discounted value of the new state
                    # the interval's reward RATE (sum / clock): scale-free
                    # across the ladder — a band-8 tick sums ~32k tokens of
                    # presses, and a raw sum there would make one TD term
                    # ~1e5 x CE; band 3 (clock 1) is unchanged, so the
                    # dopamine trace is the raw press value
                    R = (st["R_carry"][u] + (rew_cum[:, t + 1] - rew_cum[:, v_from[u]])) / float(self.clocks[k])
                    g_bg = None
                    if bg_on:
                        z_bg = torch.sigmoid(cell.z(torch.cat([h_prev.detach(), pooled.detach()], dim=-1)))
                        g_bg = z_bg if P_bg is None else z_bg * P_bg[:, i].unsqueeze(-1)
                    v_pairs[u].append((h_prev, R, h_new, g_bg))
                    if self.clocks[k] == 1 and len(gtr) == t:
                        gtr.append(g.detach())
                    if self.dopamine > 0 and (k == self.dopamine_band if self.dopamine_band is not None
                                              else self.clocks[k] == 1):
                        with torch.no_grad():
                            hv = self.value[str(u)]
                            d_t = R + self.value_gamma * hv(h_new).squeeze(-1) - hv(h_prev).squeeze(-1)
                        d_abs = d_t.abs()
                        for t2 in range(v_from[u], t + 1):          # the interval this tick closed
                            rpe[t2] = d_abs if rpe[t2] is None else rpe[t2] + d_abs
                    st["R_carry"][u] = rew.new_zeros(B)
                    v_from[u] = t + 1
                    # the prediction for the fidelity loss comes from the
                    # same tick on DETACHED inputs (the certified band_credit
                    # semantics): fidelity trains the band's cell and
                    # predictor only — it never bends the PFC or the veto.
                    # The state path above stays live, so CE credit still
                    # flows into what the council tells the band. (scan2
                    # measured the live variant 0.7 nats behind at step 2500
                    # once the target was honest.)
                    h_fid, _ = cell(st["lp"][u], st["lh"][u], st["lg"][u])
                    st["pend"][u] = self.pred[str(u)](h_fid)
                    if self.clocks[k] > 1:
                        wcost.append(g.mean())
                    st["cnt"][u] = 0
                    h_all = h_all.index_copy(1, self.band_idx[i:i + 1], h_new.unsqueeze(1))
                    acc_all = acc_all.index_fill(1, self.band_idx[i:i + 1], 0.0)
                m = self._slots_from(h_all, W_mem)
        for i, u in enumerate(self.ukeys):                 # the open accumulators carry on
            st["acc"][u] = acc_all[:, i]
        # ---- value, TD across the ladder (per unit, per tick) ----
        vl, bgl = [], []
        for u in self.ukeys:
            if not v_pairs[u]:
                continue
            hp = torch.stack([a for a, _, _, _ in v_pairs[u]], dim=1)      # [B, n, d]
            Rr = torch.stack([b for _, b, _, _ in v_pairs[u]], dim=1)      # [B, n]
            hn = torch.stack([c for _, _, c, _ in v_pairs[u]], dim=1)
            head = self.value[str(u)]
            v_prev = head(hp).squeeze(-1)
            with torch.no_grad():
                v_next = head(hn).squeeze(-1)
            td = Rr + self.value_gamma * v_next - v_prev                    # [B, n]
            vl.append((td ** 2).mean())
            if bg_on:
                gb = torch.stack([q for _, _, _, q in v_pairs[u]], dim=1)  # [B, n, d] gates on detached inputs
                delta = td.detach()
                target = (delta > 0).float().unsqueeze(-1).expand_as(gb)
                bce = nn.functional.binary_cross_entropy(
                    gb.clamp(1e-6, 1 - 1e-6), target, reduction="none")
                bgl.append((delta.abs().unsqueeze(-1) * bce).mean())
        self._value_loss = torch.stack(vl).mean() if vl else None
        self._bg_loss = self.bg_w * torch.stack(bgl).mean() if bgl else None
        self._gate_trace = torch.stack(gtr, dim=1).mean(-1) if gtr else None   # [B, T]
        for u in self.ukeys:                               # the open TD intervals carry on
            st["R_carry"][u] = st["R_carry"][u] + (rew_cum[:, T] - rew_cum[:, v_from[u]])
        S0 = torch.stack(s0s, dim=1)                       # [B, T, d] the PFC's conclusions
        if self.order == "cortex_first":
            C = torch.stack(cs, dim=1)                     # [B, T, d]
        else:
            Bd = torch.stack(bundles, dim=1)               # [B, T, 1+K, d]
            C = self._trunk(S0.reshape(B * T, self.d),
                            Bd.reshape(B * T, Bd.shape[2], self.d), dev)
            C = C.reshape(B, T, self.d)
        # ---- band fidelity, batched per band: pend (from the previous
        # tick) vs the cortex stream pooled over the interval, centred;
        # intervals that began in earlier chunks carry their partial sum
        # in acc_c. One (t_last, fid[B, n_ticks]) entry per band per chunk
        # — every band weighs the same in the fidelity loss (the hybrid's
        # one-tick-per-chunk semantics; per-tick entries had let band 3's
        # 63 ticks outweigh band 5's one), and the scoring is a handful of
        # ops per band instead of ~18 per tick (the step is launch-bound)
        Ccum = torch.cat([C.new_zeros(B, 1, self.d), C.cumsum(1)], dim=1)  # [B, T+1, d]
        for u in self.ukeys:
            k = self.unit_band[u]                                 # band_mu / ticks are per band
            ent = [(t, pend, sf, cnt) for (t, uu, pend, sf, cnt) in tick_log if uu == u]
            if not ent:
                continue
            ts = [e[0] for e in ent]
            hi = torch.tensor([t + 1 for t in ts], device=dev)
            lo = torch.tensor([e[2] for e in ent], device=dev)
            cnts = torch.tensor([float(max(e[3], 1)) for e in ent], device=dev)
            sums = Ccum[:, hi] - Ccum[:, lo]                          # [B, n, d]
            if ent[0][2] == 0:                                        # the carried interval
                sums = torch.cat([(sums[:, :1] + st["acc_c"][u].unsqueeze(1)),
                                  sums[:, 1:]], dim=1)
            # the target is DETACHED: the band predicts the cortex; the
            # cortex is not bent toward being predictable (a live target
            # pulled the decoder and the council with a force unrelated to
            # CE — in the hybrid the anisotropy floor kept it tiny, here the
            # honest centred target made it full-strength)
            pooled_c = (sums / cnts[None, :, None]).detach()
            pend_all = torch.stack([e[1] for e in ent], dim=1)        # [B, n, d]
            target = pooled_c
            if self.band_center:
                # TRAINING centres by the batch mean at each tick: a
                # lagging running mean leaves the training DRIFT of the
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
                    with torch.no_grad():                             # EMA 0.99 per tick, in order
                        pm = pooled_c.detach().mean(0)                # [n, d]
                        n = pm.shape[0]
                        w = 0.01 * torch.pow(0.99, torch.arange(n - 1, -1, -1,
                                                                device=dev, dtype=pm.dtype))
                        if self.band_mu_n[k] == 0:
                            w[0] = 0.99 ** (n - 1)                    # seeded by the first tick
                            self.band_mu[k] = (w[:, None] * pm).sum(0)
                        else:
                            self.band_mu[k] = (0.99 ** n) * self.band_mu[k] + (w[:, None] * pm).sum(0)
                        self.band_mu_n[k] += n
            fid = nn.functional.cosine_similarity(pend_all, target, dim=-1)   # [B, n]
            ticks[k].append((ts[-1], fid))                    # units of one band share its entry list
            st["acc_c"][u] = torch.zeros_like(st["acc_c"][u])
        for u in self.ukeys:                               # the open intervals' partial sums
            if seg_start[u] < T:
                st["acc_c"][u] = st["acc_c"][u] + (Ccum[:, T] - Ccum[:, seg_start[u]])
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
        if self.veto and K > 1 and veto_sum is not None:
            off = veto_sum * (1 - self.veto_eye)              # [K(j), K(k)], diagonal dropped
            vm = off.sum(0) / (B * T * (K - 1))
            self._veto_mean = {}
            for i, u in enumerate(self.ukeys):
                if vetoes[u]:
                    k = self.unit_band[u]
                    self._veto_mean.setdefault(k, []).append(float(vm[i]))
            self._veto_mean = {k: sum(v) / len(v) for k, v in self._veto_mean.items()}
        else:
            self._veto_mean = {}
        # ---- the carried band graphs (A38): the last write of each band
        # that ticked, recomputed through cloned params on detached
        # inputs, so the next chunk's CE and fidelity can credit it
        # without touching this chunk's graph
        for k in self.ukeys:
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
        if self.dopamine > 0:
            # the dopamine gain per token rides the write mask: values > 1
            # are applied after the strength sigmoid (clamped to 1 below)
            dopa = torch.stack([torch.zeros(B, device=dev) if r is None else r for r in rpe], dim=1)
            smask = smask * (1.0 + self.dopamine * dopa)
        self._dopa_trace = None if self.dopamine <= 0 else dopa.detach()
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
            sv = (torch.sigmoid(tu[toks_all]) * sm_all).clamp(max=1.0)
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

    def dopa_trace(self):
        """[B, T] |RPE| per token of the last chunk (dopamine), or None."""
        return getattr(self, "_dopa_trace", None)

    def pop_value_loss(self):
        c = self._value_loss
        self._value_loss = None
        return c

    def pop_bg_loss(self):
        """The basal-ganglia actor term, already weighted by bg_w (None at bg_w=0)."""
        c = getattr(self, "_bg_loss", None)
        self._bg_loss = None
        return c

    def gate_trace(self):
        """[B, T] mean band-3 gate per token of the last chunk, or None."""
        return getattr(self, "_gate_trace", None)

    def set_reward_tokens(self, levels):
        """levels: {token_id: level} with level 1..4 = <+1>, <+2>, <-1>, <-2>."""
        self.reward_lut.zero_()
        for tid, lv in levels.items():
            if tid is not None and 0 <= int(tid) < self.reward_lut.shape[0]:
                self.reward_lut[int(tid)] = int(lv)

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
