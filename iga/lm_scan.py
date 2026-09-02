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

import math
import torch
import torch.nn as nn

from .lm_bands import N_BANDS
from .lm_hybrid import LogitStore
from .lm_transformer import Block, make_mlp

SCAN_CLOCKS = {3: 1, 4: 8, 5: 64, 6: 512, 7: 4096, 8: 32768}   # tokens


def _extra(rw, cx):
    """the council's optional slots as positional extras: () / (rw,) /
    (rw, cx) — a 4-arg council (tests, older monkeypatches) keeps
    working when neither is present."""
    if cx is not None:
        return (rw, cx)
    if rw is not None:
        return (rw,)
    return ()


R_MAX = 4.0      # the saturating reward: presses per interval, bounded like dopamine's range


def press_levels_from_events(events, B, T, device=None):
    """[B, T] long: the press level (1..4 = +1 +2 -1 -2) at each position
    from a chunk's button events (per lane: (p, kind, d)); None when the
    chunk holds no press — so a chunk without presses costs nothing and
    reads bit-identically to the LUT path."""
    lev = None
    for lane, evs in enumerate(events or ()):
        for p, kind, d in evs:
            if kind != "button" or not (0 <= p < T):
                continue
            v = int(d.get("v", 0))
            if v == 0:
                continue
            if lev is None:
                lev = torch.zeros(B, T, dtype=torch.long, device=device)
            lev[lane, p] = v if v > 0 else 2 - v
    return lev


def affect_from_events(events, x, hold, eot_ids=None, device=None):
    """[B, T] float: the caretaker's face as a continuous sense — each
    button event's level held until the next event or a turn end (an
    eot token relaxes it to 0), carried across chunks by `hold` (a
    per-lane list this call updates). None when every lane is still,
    so a faceless chunk reads bit-identically to the plain path."""
    B, T = x.shape
    af = torch.zeros(B, T, dtype=torch.float32)
    xs = x.tolist() if eot_ids else None
    eots = set(int(e) for e in (eot_ids or ()))
    any_ = False
    for lane in range(B):
        lv = float(hold[lane]) if lane < len(hold) else 0.0
        evs = {}
        for p, kind, d in (events[lane] if events and lane < len(events) else ()):
            if kind == "button" and 0 <= p < T:
                evs[int(p)] = float(d.get("v", 0))
        row = af[lane]
        for t_ in range(T):
            if t_ in evs:
                lv = evs[t_]
            if xs is not None and xs[lane][t_] in eots:
                lv = 0.0
            row[t_] = lv
        if lv != 0.0 or bool(row.any()):
            any_ = True
        if lane < len(hold):
            hold[lane] = lv
    return af.to(device) if any_ else None


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
                 fast_gate=None, read_drop=0.5, aux_trunk=0.0,
                 mlp="gelu", band_center=True, order="cortex_first",
                 kd_base=1, slot_every=8, write_every=4, kd_max=4096,
                 compile_council=False, compile_mode="default", compile_read=False,
                 store_exact=False, register=None, reward_slot=False, value_gamma=0.9,
                 dopamine=0.0, bg_w=0.0, dopamine_band=None,
                 plasticity=0.0,
                 ponder=1, ponder_w=0.01, ponder_mode="route", ponder_aux=0.0,
                 store_wipe=None, write_surprise=0.0, press_unwrite=False,
                 ponder_reenter="token", route_cap=0.125, tie_embed=False,
                 z_w=0.0, plan_m=0, rem_k=32, plan_cand=0,
                 intrinsic_w=0.0, imag_k=0, keyed_content=False, **legacy):
        # removed organs (v16 refactor, Plan 49): vetoes (convicted),
        # ctx store + ctx_sparse (inert x4), novelty loop (unused).
        # Old checkpoints' cfgs still carry the kwargs — accepted ONLY
        # when off, so scan15 keeps loading and a hot value fails loud.
        _dead = {"veto", "ctx_store", "ctx_sparse", "novelty_w"}
        for k_, v_ in legacy.items():
            assert k_ in _dead, f"unknown ctor arg {k_!r}"
            assert not v_, f"removed organ {k_!r} passed as {v_!r}"

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
        # THE STATE-VALUED HIPPOCAMPUS (the user's call, 2026-08-22): a
        # second store ladder whose VALUES are the PFC's own conclusions
        # (unit S0_t) under the same keys (S0_{t-1}), written at the same
        # dopamine-scaled strength with the same per-band decay, read
        # every slot_every tokens by the current conclusion and fed to
        # the council as one more slot — episodic CONTEXT recall ("what
        # did I conclude the last time I was here"), where the token
        # store recalls only which token followed. ctx_store=False is
        # exact (no slot, no stores).


        # THREE MORE FROM THE HUMAN (the user's call, 2026-08-22 23:30):
        # plasticity (dopamine-gated cortical learning): the chunk's
        # per-token CE weights are 1 + plasticity * |delta| (press units),
        # normalised to mean 1 — surprising moments teach the cortex
        # harder (DA-gated LTP); 0 = exact. ctx_sparse (pattern
        # separation, dentate gyrus -> CA3): the state store's lifted
        # keys keep only their ctx_sparse largest entries (k-winners-
        # take-all), so similar conclusions get near-orthogonal keys;
        # the token store's keys stay dense (its value IS generalising
        # across similar contexts); 0 = exact. novelty_w (the
        # hippocampus -> VTA loop): a context with nothing stored near
        # it — a low state-store read norm — is NOVEL and is written
        # harder, s_t x (1 + novelty_w * novelty_t); 0 = exact.
        # PONDERING (the user's design, 2026-08-23): the organism sets its
        # own cortex rate. Per token the council may run up to `ponder`
        # cycles on its own output (weight-tied); after each cycle the
        # basal ganglia's GO gate (a linear on the token slot) gives the
        # probability of speaking now; the bundle the cortex decodes is
        # v16: ponder_mode="route" ONLY (halt/force removed in the Plan-49
        # refactor; route certified in v14 — the router spends up to K-1
        # extra council cycles on ~route_cap of tokens, tau = quantile-EMA
        # frozen at eval, blend C1 + g(C_deep - C1), deep CE at ponder_aux).
        # ROUTED CYCLES (v14, the user's call: "only calculate a cycle if
        # it can't do it in one"). ponder_mode="route": every token gets
        # cycle 1; a router head on the conclusion picks the tokens whose
        # logit clears a threshold and ONLY THOSE take cycles 2..K —
        # batched AFTER the token loop (legal because a cycle ticks
        # nothing and the stores only change at chunk boundaries, so the
        # re-reads are identical to in-loop). The routed tokens' decoded
        # state is the blend C1 + g (C_deep - C1), g = sigmoid(router
        # logit): the router earns gradient through g (mixture-of-depths).
        # The threshold self-tunes to route_cap of tokens (a quantile
        # EMA, frozen at eval — causal and serve-identical). ponder_aux
        # adds the routed tokens' own deep CE so the deep path stays
        # competent. Cost: 1 + route_cap x (K-1) council passes and
        # 1 + route_cap trunk passes. ponder=1 exact.
        if ponder_mode in ("halt", "force") and int(ponder) <= 1:
            ponder_mode = "route"     # legacy no-op modes at K=1
        assert ponder_mode == "route", \
            f"removed ponder mode {ponder_mode!r} (v16: route only)"
        assert ponder_reenter in ("null", "token")
        if ponder_mode == "route":
            assert ponder_reenter == "token", "routing recycles the bundle"
        self.ponder_reenter = ponder_reenter
        self.ponder = max(1, int(ponder))
        self.ponder_w = float(ponder_w)
        self.ponder_mode = ponder_mode
        self.ponder_aux = float(ponder_aux)
        # v16 PLAN (49a): the PFC looks forward — m plan vectors per
        # token, horizons log-spaced (1,2,4,8...)[:m]; plan_gate
        # zero-init = no-op at birth (the gate law). The h=1 slice of
        # plan_head doubles as the REM transition (weight-tied: the
        # planner IS the dreamer).
        self.plan_m = int(plan_m)
        self.rem_k = int(rem_k)
        self.plan_cand = int(plan_cand)
        self.imag_k = int(imag_k)
        # 49i: intrinsic reward — prediction success in press units.
        # BG's judge-free signal (the HPC law applied to wanting): the
        # gates' internal actions CAUSE prediction success, so dopamine
        # over intrinsic RPE carries real causal gradients (the v14
        # conviction applied only to external reward on forced tokens).
        # LAW: intrinsic value manages COMPUTATION, never content — it
        # never touches token logits.
        self.intrinsic_w = float(intrinsic_w)
        self.plan_horizons = tuple(2 ** i for i in range(self.plan_m))
        if self.plan_m > 0:
            self.plan_fid = {}          # h -> EMA cosine (day foresight)
            self.rem_fid = {}           # n -> EMA cosine (closed-loop)
        self._plan_aux = None
        self._last_C = None
        self.route_cap = float(route_cap)
        if self.ponder > 1:
            self.null_emb = nn.Parameter(torch.randn(d) * 0.02)
            if self.ponder_mode == "route":
                self.route_head = nn.Linear(d, 1)
                nn.init.normal_(self.route_head.weight, std=0.01)   # a whisper of spread: the threshold needs a distribution to sit in
                nn.init.zeros_(self.route_head.bias)
                self.register_buffer("route_tau", torch.zeros(()))   # the logit threshold, tuned to route_cap
        self._ponder_loss = None
        self._ponder_trace = None
        self._aux_logits = None
        # v13 — THE HIPPOCAMPUS, BEST SHOT (the user's call, 2026-08-23):
        #   store_wipe="day": in TRAINING the stores are wiped at each
        #     lane's day boundary (the builder's {"kind":"day"} events,
        #     handed in as forward(day_lanes=)), the band states kept —
        #     old traces never carry the cortex through; the cortex still
        #     learns to use the store within a day. Eval/serve never pass
        #     day_lanes: the life's store persists.
        #   write_surprise=tau (nats): SURPRISE-GATED ENCODING (the NE /
        #     CA1-mismatch organ) — the write strength of x_t is scaled by
        #     sigmoid((surprise_t - mu) / tau), surprise_t = the cortex's
        #     OWN -log p(x_t) (its head before the store's read is added),
        #     mu a running mean. The store holds what the cortex cannot
        #     predict — the CLS complement — and never substitutes for
        #     what it already knows; 0 = off, exact.
        #   press_unwrite: a NEGATIVE press un-writes the graded utterance
        #     — the store would otherwise bind (question-context -> the
        #     wrong answer) and hand the mistake back at the next ask (the
        #     A67 incumbent disease, built into memory). Positions of the
        #     graded turn still in this chunk are written at strength 0;
        #     the part already written from the previous chunk is
        #     re-issued at NEGATIVE strength (the delta rule subtracts).
        #     Needs the turn markers (set_eot_ids) and press levels.
        assert store_wipe in (None, "day")
        self.store_wipe = store_wipe
        self.write_surprise = float(write_surprise)
        self.press_unwrite = bool(press_unwrite)
        self.register_buffer("surp_mu", torch.zeros(()))
        self.register_buffer("surp_n", torch.zeros(()))
        self.eot_ids = None
        self.plasticity = float(plasticity)

        self._ce_weights = None

        self.n_fixed = 2 + int(bool(reward_slot))   # token, [reward,] hippocampus
        self.slot = nn.Embedding(self.n_fixed + len(self.units), d)
        self.reward_emb = nn.Embedding(5, d)
        # LIVE_BODY §1 — the caretaker's face as a CONTINUOUS sense: its
        # level (tonic) and its change (phasic) enter the trunk input as a
        # zero-init modulation, never as a word. Zero at birth: exact.
        self.affect_in = nn.Sequential(nn.Linear(2, max(8, d // 4)), nn.GELU(),
                                       nn.Linear(max(8, d // 4), d))
        nn.init.zeros_(self.affect_in[2].weight); nn.init.zeros_(self.affect_in[2].bias)
        # LIVE_BODY §1 — ITS FACE: a forecast of the caretaker's face for
        # the word about to be said; its expression, its model of you,
        # and over a finished sentence its conscience. Zero at birth.
        self.face_head = nn.Linear(d, 1)
        nn.init.zeros_(self.face_head.weight); nn.init.zeros_(self.face_head.bias)
        self._face = None
        self._face_feat = None
        self._face_loss = None
        self.store_slot_gain = 1.0
        nn.init.zeros_(self.reward_emb.weight)            # silent at init
        self.register_buffer("reward_lut", torch.zeros(vocab_size, dtype=torch.long))
        self.register_buffer("reward_val", torch.tensor([0.0, 1.0, 2.0, -1.0, -2.0]))
        self.value_gamma = float(value_gamma)
        self._value_loss = None
        self._bg_loss = None
        self._gate_trace = None
        self.lnf = nn.LayerNorm(d)
        # v15: tie_embed shares one lexicon both directions (the size
        # class's standard — SmolLM2/Gemma tie; big models untie at ~1%
        # share). Rescale: the tied input read is scaled by sqrt(d) so
        # the embedding can serve both roles at head-friendly norm.
        self.tie_embed = bool(tie_embed)
        self.z_w = float(z_w)      # z-loss (frontier head insurance):
                                   # z_w * logsumexp(logits)^2, in the trainer
        self.head = nn.Linear(d, vocab_size, bias=not tie_embed)
        if tie_embed:
            self.head.weight = self.embed.weight
            self.emb_scale = d ** 0.5
            # the tied matrix serves two masters: init at N(0, 1/sqrt(d))
            # so the sqrt(d)-scaled INPUT read is unit-variance and the
            # raw OUTPUT logits start O(1) — the original-Transformer
            # pairing. (Default N(0,1) rows made birth CE ~900: logits
            # 32x too hot through both roles.)
            nn.init.normal_(self.embed.weight, std=d ** -0.5)
        else:
            self.emb_scale = 1.0
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
        self.windowless = True           # tokens stream; max_T is the batching unit, not a window
        self.mem_proj = nn.ModuleDict(
            {str(u): nn.Linear(d, d, bias=False) for u in self.ukeys})
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
        # the hippocampus: the certified content-keyed identity store,
        # capacity per band doubling up the ladder; half-life in WRITES
        # (one write per chunk) = the band's clock in chunks
        self.KD = {k: min(int(kd_max), 512 * (2 ** i) * kd_base)
                   for i, k in enumerate(self.bands)}
        self.key_proj = nn.Linear(d, d, bias=False)
        # THE GOAL ORGAN (49uu): a wake-surviving pursuit slot read by
        # content attention, joined through a zero-init gate — silent
        # at birth, speaks only if training proves it useful.
        self.goal_query = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.goal_query.weight)
        self.goal_gate = nn.Parameter(torch.tensor(0.0))
        self.query_proj = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.key_proj.weight)
        nn.init.eye_(self.query_proj.weight)
        # CONTENT KEYS (LIVE_BODY.md): the hippocampus keyed by the words
        # themselves, not by a learned projection of the state — recall
        # becomes a mechanism a body with no training already has
        self.keyed_content = bool(keyed_content)
        self.kc_w, self.kc_decay = 8, 0.7
        self.kc_skip = 11            # turn marks and press marks are not words: ids below this stay out of the bag
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
        if self.plan_m > 0:
            # created LAST: plan init must not shift the RNG stream of
            # the shared modules (S53 twin law)
            self.plan_head = nn.Linear(d, self.plan_m * d)
            self.plan_gate = nn.Parameter(torch.zeros(1))
            if self.plan_cand > 0:
                # 49h: the BG stub — candidate dream dynamics + a
                # selector gate, trained at night by which candidate
                # stays true (prediction-as-value; the router's law one
                # level up). Live phase re-targets the same gate with
                # the critic's value. Created after plan_head (RNG law).
                self.plan_trans = nn.ModuleList(
                    [nn.Linear(d, d) for _ in range(self.plan_cand)])
                self.plan_gate_bg = nn.Linear(d, self.plan_cand)
                self.bg_gate_use = {}      # candidate -> usage EMA
                if self.imag_k > 0:
                    # 49m: imagination OFFERED to the PFC (the user) —
                    # a k-step latent lookahead through the dreamer,
                    # added to the renderer through a zero-init gate;
                    # CE opens it only if imagining helps. Its
                    # magnitude is the vital: does it choose to
                    # imagine? Day gradients reach the dreamer here.
                    self.imag_gate = nn.Parameter(torch.zeros(1))
                    self.imag_cycle_gate = nn.Parameter(torch.zeros(1))

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
                "CM": None,
                "lp": {}, "lh": {}, "lg": {},      # last write: pooled, h before, permission
                "wbuf": [],                        # (h_prev, tokens, smask) since the last write
                "G": torch.zeros(B, 4, self.d, device=device),
                "prev_c": z(), "tail": z(), "tok": 0, "chunk": 0, "xl": None}

    def lane_state(self, st, b):
        """A detached B=1 copy of lane b's live state — the warm cortex a
        night's replay runs in (the bands carrying that lane's context,
        both stores readable); what the replay does to the copy is
        discarded, the weights keep the learning."""
        def cut(v):
            if torch.is_tensor(v):
                return v[b:b + 1].detach().clone() if v.dim() >= 1 and v.shape[0] > b else v.detach().clone()
            if isinstance(v, dict):
                return {k: cut(x) for k, x in v.items()}
            if isinstance(v, tuple):
                return tuple(cut(x) for x in v)
            if isinstance(v, list):
                return [cut(x) if torch.is_tensor(x) else
                        (tuple(cut(y) if torch.is_tensor(y) else y for y in x) if isinstance(x, tuple) else x)
                        for x in v]
            return v
        out = {k: cut(v) for k, v in st.items()}
        out["fresh"] = {u: False for u in self.ukeys}
        out["M_fresh"] = False
        if out.get("CM") is not None:
            out["CM_fresh"] = False
        out["wbuf"] = []
        return out

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
        if st.get("CM") is not None:
            if st.get("CM_fresh"):
                st["CM_fresh"] = False
            else:
                st["CM"] = {k: v.detach() for k, v in st["CM"].items()}
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

    def _council(self, c, r, m, dev, rw=None, cx=None):
        """c [B,d] token hidden, r [B,d] store slot, m [B,K,d] bands,
        rw [B,d] the reward slot (reward_slot only), cx [B,d] the context
        slot (ctx_store only) -> S' [B, n_fixed+K, d] after the exchange."""
        parts = [c.unsqueeze(1)]
        if rw is not None:
            parts.append(rw.unsqueeze(1))
        parts.append(r.unsqueeze(1))
        if cx is not None:
            parts.append(cx.unsqueeze(1))
        parts.append(m)
        S = torch.cat(parts, dim=1)
        S = S + self.slot.weight[None, : S.shape[1]]
        # the PFC is fp32 always (precision law); autocast is switched
        # OFF here so a bf16 trunk never pulls the council with it
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                            enabled=False):
            S = self._council_fn(S)
        return S

    def _council_recycle(self, S_prev, r, dev, rw=None, cx=None, im=None):
        """one more round over the council's own bundle (ponder_reenter=
        "token"): token slot = S_prev[0] + NULL (the re-cycle marker), the
        read slots fresh (+ their tags), the band slots as discussed.
        im (49n): the imagined-future summary rides INTO the token slot
        — the re-deliberation happens with the rollout in the room."""
        tok0 = S_prev[:, 0] + self.null_emb.unsqueeze(0)
        if im is not None:
            tok0 = tok0 + im
        parts = [tok0.unsqueeze(1)]
        i = 1
        if rw is not None:
            parts.append((rw + self.slot.weight[i]).unsqueeze(1)); i += 1
        parts.append((r + self.slot.weight[i]).unsqueeze(1)); i += 1
        if cx is not None:
            parts.append((cx + self.slot.weight[i]).unsqueeze(1)); i += 1
        parts.append(S_prev[:, i:])
        S = torch.cat(parts, dim=1)
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

    def _content_bags(self, tokens, st):
        """content keys: a recency-weighted bag of the last kc_w token
        embeddings, unit norm, detached. Returns (incl, excl): incl[t]
        includes x_t (the read query at t: what follows this?), excl[t]
        ends at x_{t-1} (the write key at t, whose value is x_t) — so
        query(t) == key(t+1) by construction. The tail rides in st."""
        B, T = tokens.shape
        w = self.kc_w
        E = self.embed.weight.detach()
        tail = st.get("tail_toks")
        if tail is None or tail.shape[0] != B:
            tail = torch.full((B, w - 1), -1, dtype=torch.long, device=tokens.device)
        seq = torch.cat([tail, tokens], dim=1)                       # [B, w-1+T]
        valid = (seq >= int(getattr(self, "kc_skip", 0))).to(E.dtype).unsqueeze(-1)   # specials and pads carry no content
        e = E[seq.clamp(min=0)] * valid                              # [B, w-1+T, d]
        acc = torch.zeros(B, T, self.d, device=tokens.device, dtype=E.dtype)
        for i in range(w):
            acc = acc + (self.kc_decay ** i) * e[:, w - 1 - i: w - 1 - i + T]
        incl = nn.functional.normalize(acc, dim=-1)
        prev = st.get("bag_prev")
        if prev is None or prev.shape[0] != B:
            prev = torch.zeros(B, self.d, device=tokens.device, dtype=E.dtype)
        excl = torch.cat([prev.unsqueeze(1).to(incl.dtype), incl[:, :-1]], dim=1)
        st["tail_toks"] = seq[:, -(w - 1):].detach()
        st["bag_prev"] = incl[:, -1].detach()
        return incl, excl

    def reset_bag(self, st):
        """a new turn begins: the ear's words are keyed by the ear's words,
        never by whatever the mouth was babbling before them"""
        if isinstance(st, dict):
            st.pop("tail_toks", None)
            st.pop("bag_prev", None)

    def _read(self, st, q_hidden, read_ok):
        """the hippocampus read: identity-space vectors from the
        previous chunks' M, alpha-weighted; q_hidden [B, d] (one token)
        or [B, T, d] (the chunk, batched); None when reads are off."""
        if not read_ok or self.store_read_off:
            return None
        one = q_hidden.dim() == 2
        q = q_hidden if self.keyed_content else \
            nn.functional.normalize(self.query_proj(q_hidden), dim=-1)
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

    def _read_flat(self, st, q, lane_idx, read_ok):
        """the token store read for a flat routed set: q [n, d]
        conclusions, lane_idx [n] their lanes. Grouped BY LANE — never
        M[lane_idx] (that would materialise a [n, d, KD] copy of the
        store per routed position: ~2 GB per band at 32 lanes)."""
        if not read_ok or self.store_read_off:
            return None
        qn = q if self.keyed_content else \
            nn.functional.normalize(self.query_proj(q), dim=-1)     # [n, d]
        out = None
        for b in torch.unique(lane_idx).tolist():
            sel = (lane_idx == b)
            qb = qn[sel].unsqueeze(0)                                # [1, nb, d]
            rb = None
            for k in self.bands:
                if k in self.lesioned:
                    continue
                stn = self.stores[str(k)]
                r = self._lift_read_fn(st["M"][k][b:b + 1], qb, stn.proj, stn.phase, self.alpha[str(k)])
                rb = r if rb is None else rb + r
            if rb is None:
                return None
            if out is None:
                out = q.new_zeros(q.shape[0], self.d)
            out = out.index_copy(0, torch.nonzero(sel).flatten(), rb.squeeze(0))
        return out

    def pop_plan_aux(self):
        """v16 day foresight loss (49a) from the last forward, or None."""
        out = self._plan_aux
        self._plan_aux = None
        return out

    def plan_step(self, x):
        """One dream step. plan_cand=0: the h=1 plan slice. Otherwise
        the BG-stub mixture: softmax(gate(x)) over candidate
        transitions — the selector chooses the dynamics to think with,
        and the night's consistency loss teaches it which choices stay
        true."""
        d = self.d
        if self.plan_cand <= 0:
            return nn.functional.linear(
                x, self.plan_head.weight[:d], self.plan_head.bias[:d])
        g = torch.softmax(self.plan_gate_bg(x), dim=-1)      # [N, C]
        y = torch.stack([t(x) for t in self.plan_trans], dim=-1)
        with torch.no_grad():
            use = g.mean(dim=tuple(range(g.dim() - 1)))
            for c in range(self.plan_cand):
                f = float(use[c])
                self.bg_gate_use[c] = 0.98 * self.bg_gate_use.get(c, f) \
                    + 0.02 * f
        return (y * g.unsqueeze(-2)).sum(-1)

    def rem_loss(self, k=None):
        """v16 LATENT REM, 49f final (the user): the night is PLAIN.
        The hippocampus hands over every state it captured this chunk;
        the predictor (the h=1 plan slice — the planner IS the dreamer)
        rolls each one closed-loop k steps; the recorded wake trajectory
        scores every dream. Uniform over captures — no salience
        weighting, no tiers, no emotional tagging. Captures beyond 16
        per lane are evenly subsampled (deterministic). No capture
        record -> one rollout from the chunk start. The variance hinge
        stays purely as a collapse guard. Day training is untouched:
        foresight every token, CE and every other loss as normal.
        No tokens, store untouched, bands untouched."""
        if self.plan_m == 0 or self._last_C is None:
            return None
        C = self._last_C                        # [B, T, d] detached wake
        Cl = getattr(self, "_last_C_live", None)
        if Cl is None or Cl.shape != C.shape:
            Cl = C                              # fallback: dreamer-only
        B, T, d = C.shape
        k = min(int(k or self.rem_k), T - 2)
        if k < 2:
            return None
        sv = getattr(self, "_last_sv", None)
        have_sv = sv is not None and sv.shape[0] == B and sv.shape[1] == T
        lanes, t0s, ws = [], [], []
        max_t0 = T - k - 1
        CAP = 16
        for lane in range(B):
            if have_sv:
                idx = torch.nonzero(
                    sv[lane, :max_t0 + 1].abs() > 0).flatten()
            else:
                idx = torch.zeros(1, dtype=torch.long, device=C.device)
            if idx.numel() == 0:
                idx = torch.zeros(1, dtype=torch.long, device=C.device)
            if idx.numel() > CAP:               # evenly subsampled
                pick = torch.linspace(0, idx.numel() - 1, CAP,
                                      device=C.device).long()
                idx = idx[pick]
            # pad to CAP with cyclic repeats at weight 0: every chunk
            # presents the SAME [B*CAP] seed batch, so compiled graphs
            # never see a new shape (school's short days under-filled
            # the cap -> variable N -> 256 dynamo recompiles -> host
            # RAM bloat -> OOM-kill at the step-500 save)
            real = idx.numel()
            if real < CAP:
                rep = torch.arange(CAP, device=C.device) % real
                idx = idx[rep]
            lanes.extend([lane] * CAP)
            t0s.extend(idx.tolist())
            ws.extend([1.0] * real + [0.0] * (CAP - real))
        li = torch.tensor(lanes, device=C.device)
        ti = torch.tensor(t0s, device=C.device)
        w = torch.tensor(ws, device=C.device)
        wsum = w.sum().clamp_min(1.0)
        c = Cl[li, ti]                          # [B*CAP, d] LIVE seeds (49l)
        loss = C.new_zeros(())
        rolled = []
        cos = None
        for n in range(1, k + 1):
            c = self.plan_step(c)
            cos = nn.functional.cosine_similarity(c, C[li, ti + n],
                                                  dim=-1)
            loss = loss + ((1.0 - cos) * w).sum() / wsum
            rolled.append(c)
        f = float(((cos * w).sum() / wsum).detach())
        self.rem_fid[k] = 0.98 * self.rem_fid.get(k, f) + 0.02 * f
        R = torch.stack(rolled, dim=1)          # [B*CAP, k, d]
        wake_std = C[:, 1:k + 1].std(dim=1).mean().detach()
        r_std = (R.std(dim=1).mean(-1) * w).sum() / wsum
        var_hinge = torch.relu(0.5 * wake_std - r_std)
        return loss / k + var_hinge

    def pop_route_aux(self):
        """route mode with ponder_aux > 0: (deep logits [n, V], flat
        B*T indices) of the routed tokens from the last training
        forward, or None; the trainer adds ponder_aux x their CE."""
        out = getattr(self, "_route_aux", None)
        self._route_aux = None
        return out

    def pop_ponder_loss(self):
        """ponder_w x (expected cycles - 1), or None (ponder = 1)."""
        c = self._ponder_loss
        self._ponder_loss = None
        return c

    def ponder_trace(self):
        """[B, T] expected council cycles per token of the last chunk, or None."""
        return self._ponder_trace

    def pop_ce_weights(self):
        """[B, T] per-token CE weights (mean 1) for the last chunk, or
        None when plasticity is 0."""
        w = self._ce_weights
        self._ce_weights = None
        return w

    # ---------------- forward ----------------
    def forward(self, tokens, st, scene_starts=None, press_levels=None, day_lanes=None,
                affect=None, face_target=None):
        """press_levels [B, T] long (optional): the press level at each
        position from the stream's button EVENTS (1..4 = +1 +2 -1 -2) —
        the grade as a sense. Combined by max with the token LUT, so a
        shard whose presses are tokens reads identically with or
        without it, and a shard whose presses are events only (no
        approval tokens in the stream) still rewards."""
        B, T = tokens.shape
        dev = tokens.device
        K = len(self.units)                      # council band slots = units
        read_ok = (not self.training) or float(torch.rand(())) >= self.read_drop
        self._reads_used = read_ok
        lg_E = nn.functional.normalize(self.embed.weight, dim=-1).detach()
        ticks = [[] for _ in range(max(N_BANDS, max(self.bands) + 1))]
        wcost = []
        emb = self.embed(tokens) * self.emb_scale        # [B, T, d] (tied: sqrt(d) read)
        if affect is not None and hasattr(self, "affect_in"):
            a_ = affect.to(emb.device, emb.dtype).clamp(-6.0, 6.0) / 6.0     # [B, T]
            prev = st.get("affect_prev")
            if prev is None or prev.shape[0] != B:
                prev = a_[:, :1]
            a_prev = torch.cat([prev.to(a_.dtype), a_[:, :-1]], dim=1)
            emb = emb + self.affect_in(torch.stack([a_, a_ - a_prev], dim=-1))
            st["affect_prev"] = a_[:, -1:].detach()
        lev = self.reward_lut[tokens]                    # [B, T] press level per token
        if press_levels is not None:
            lev = torch.maximum(lev, press_levels.to(lev.device, lev.dtype))
        rew = self.reward_val[lev]                       # [B, T] its value
        if self.intrinsic_w > 0:
            ps = st.get("prev_surp")
            if ps is not None and ps.shape == rew.shape:
                # 49i: one chunk delayed (logits exist only after the
                # decode); bounded like a press
                rew = rew + (self.intrinsic_w
                             * (self.surp_mu - ps)).clamp(-2.0, 2.0)
        rew_cum = torch.cat([rew.new_zeros(B, 1), rew.cumsum(1)], dim=1)   # [B, T+1]
        rslots = self.reward_emb(lev) if self.reward_slot else None        # [B, T, d]
        if rslots is not None and self.reward_off:
            rslots = torch.zeros_like(rslots)
        gtr = []                                         # band-3 gate means per token (the BG trace)
        bg_on = self.bg_w > 0
        v_pairs = {u: [] for u in self.ukeys}            # (h_prev, R, h_now) per tick
        rpe = [None] * T                                 # |RPE| per token (dopamine), summed over clock-1 units
        rpe_s = [None] * T                               # the SIGNED RPE per token (plasticity: dips are LTD)
        v_from = {u: 0 for u in self.ukeys}              # chunk-local start of the open TD interval
        st.setdefault("R_carry", {u: rew.new_zeros(B) for u in self.ukeys})   # rewards since the last tick, earlier chunks
        nf = self.n_fixed
        for key in ("lp", "lh", "lg"):
            st.setdefault(key, {})
        st.setdefault("wbuf", [])
        W_mem = self._mem_W()
        h_all = torch.stack([st["h"][u] for u in self.ukeys], dim=1)    # [B, U, d]
        acc_all = torch.stack([st["acc"][u] for u in self.ukeys], dim=1)  # [B, U, d]
        m = self._slots_from(h_all, W_mem)
        kc = self.keyed_content
        if kc:
            q0 = st.get("bag_prev")
            rd0 = self._read(st, q0, read_ok) if (st["chunk"] > 0 and q0 is not None) else None
            bag_incl, bag_excl = self._content_bags(tokens, st)
        else:
            rd0 = self._read(st, st["prev_c"], read_ok) if st["chunk"] > 0 else None
        r_slot = self.store_in(rd0 * getattr(self, "store_slot_gain", 1.0)) if rd0 is not None \
            else torch.zeros(B, self.d, device=dev)
        c_slot = None
        cs, s0s, bundles, tick_log = [], [], [], []
        seg_start = {u: 0 for u in self.ukeys}   # chunk-local start of each unit's open interval
        for t in range(T):
            e = emb[:, t]
            rw_t = rslots[:, t] if rslots is not None else None
            if self.order == "cortex_first":
                c = self._trunk(e, torch.cat([m, r_slot.unsqueeze(1)], 1), dev)
                S = self._council(c, r_slot, m, dev, *_extra(rw_t, c_slot))
                cs.append(S[:, 0])
            else:
                # PFC first: the council deliberates on the raw token; the
                # neocortex is OFF the recurrent path and decodes the whole
                # chunk's bundles at once after the loop (same math, one
                # batched call — the per-token decoder was launch-bound)
                S = self._council(e, r_slot, m, dev, *_extra(rw_t, c_slot))
                bundles.append(S[:, 1:])
            s0 = S[:, 0]
            s0s.append(s0)
            if (t + 1) % self.slot_every == 0 and t + 1 < T:
                # the hippocampus, queried by what the council concluded
                rd = self._read(st, s0, read_ok)
                r_slot = self.store_in(rd * getattr(self, "store_slot_gain", 1.0)) if rd is not None \
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
                    # the interval's reward in PRESS UNITS, SATURATING: the
                    # sum of presses since the last tick, clamped to +-R_MAX
                    # (a band-8 interval holds ~44 presses; unclamped, one
                    # TD term was ~1e5 x CE; as a RATE (sum / clock, the
                    # S28 fix) the slow bands' TD signal shrank to ~1e-5 and
                    # the value heads never left zero — scan9: value AUC at
                    # chance at 37M tokens, |RPE| at presses RISING. One
                    # unit for the value, the dopamine stamp and the BG
                    # weight: presses, bounded like dopamine's dynamic range).
                    R = (st["R_carry"][u] + (rew_cum[:, t + 1] - rew_cum[:, v_from[u]])).clamp(-R_MAX, R_MAX)
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
                        d_abs = d_t.abs()                       # press units already
                        for t2 in range(v_from[u], t + 1):          # the interval this tick closed
                            rpe[t2] = d_abs if rpe[t2] is None else rpe[t2] + d_abs
                            rpe_s[t2] = d_t if rpe_s[t2] is None else rpe_s[t2] + d_t
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
                delta = td.detach()                                            # press units
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
        self._route_deep = None
        if self.ponder > 1 and self.ponder_mode == "route" and self.order != "cortex_first":
            # ROUTED CYCLES, batched after the loop (a cycle ticks nothing
            # and the stores only change at chunk boundaries, so the
            # re-reads here are identical to in-loop ones)
            r_log = self.route_head(S0).squeeze(-1)                      # [B, T]
            with torch.no_grad():
                routed = r_log > self.route_tau
                if self.training:
                    self.route_tau.add_(0.05 * (routed.float().mean() - self.route_cap))
            tr = torch.ones(B, T, device=dev)
            if bool(routed.any()):
                idx = torch.nonzero(routed.reshape(-1)).flatten()        # flat B*T
                lane_idx = torch.div(idx, T, rounding_mode="floor")
                s_r = S0.reshape(B * T, self.d)[idx]                     # [n, d]
                S_prev = torch.cat([s_r.unsqueeze(1),
                                    Bd.reshape(B * T, Bd.shape[2], self.d)[idx]], dim=1)
                rw_r = (rslots.reshape(B * T, self.d)[idx]
                        if rslots is not None else None)
                for _cyc in range(self.ponder - 1):
                    rdb = self._read_flat(st, bag_incl.reshape(B * T, self.d)[idx] if kc else s_r,
                                          lane_idx, read_ok)
                    r_b = self.store_in(rdb * getattr(self, "store_slot_gain", 1.0)) if rdb is not None else torch.zeros_like(s_r)
                    c_b = None
                    im_b = None
                    if self.plan_cand > 0 and self.imag_k > 0:
                        # 49n: imagine -> cycle -> imagine -> cycle.
                        # Each re-deliberation sees a fresh rollout
                        # from the CURRENT revised thought, through a
                        # zero-init gate of its own.
                        rr = s_r
                        acc_ = None
                        for _ in range(self.imag_k):
                            rr = self.plan_step(rr)
                            acc_ = rr if acc_ is None else acc_ + rr
                        im_b = self.imag_cycle_gate * (acc_ / self.imag_k)
                    S_prev = self._council_recycle(S_prev, r_b, dev, rw_r, c_b, im=im_b)
                    s_r = S_prev[:, 0]
                C_deep = self._trunk(s_r, S_prev[:, 1:], dev).float()    # [n, d]
                g = torch.sigmoid(r_log.reshape(-1)[idx]).unsqueeze(-1)  # the router's gradient path
                C_flat = C.reshape(B * T, self.d)
                mixed = (1.0 - g) * C_flat[idx] + g * C_deep
                C = C_flat.index_copy(0, idx, mixed.to(C_flat.dtype)).reshape(B, T, self.d)
                tr.reshape(-1).index_fill_(0, idx, float(self.ponder))
                self._route_deep = (C_deep, idx)
            self._ponder_loss = None
            self._ponder_trace = tr
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
        if self.plan_m > 0:
            # PLAN (49a): the PFC emits its predicted next working
            # states; their mean conditions the renderer through a
            # zero-init gate. Day foresight: plan_j(t) vs the actual
            # working state h_j steps later (stop-grad targets).
            with torch.autocast(device_type=dev.type,
                                dtype=torch.bfloat16, enabled=False):
                plan = self.plan_head(C).view(B, T, self.plan_m, self.d)
                C = C + self.plan_gate * plan.mean(2)
                if self.training:
                    terms = []
                    tgt_all = C.detach()
                    for j, h in enumerate(self.plan_horizons):
                        if T <= h:
                            continue
                        cos = nn.functional.cosine_similarity(
                            plan[:, :-h, j], tgt_all[:, h:], dim=-1)
                        terms.append(1.0 - cos.mean())
                        f = float(cos.mean().detach())
                        self.plan_fid[h] = (0.99 * self.plan_fid.get(h, f)
                                            + 0.01 * f)
                    if terms:
                        self._plan_aux = torch.stack(terms).mean()
                    self._last_C = tgt_all
                    # 49l FULL REM (the user): dream seeds stay LIVE —
                    # the night's error backpropagates through the
                    # rollout INTO the council: be the kind of mind
                    # whose moments roll forward truly. Targets remain
                    # stop-grad (the future judges, and is never
                    # pulled toward the dream).
                    self._last_C_live = C
                    self._rem_day = bool(day_lanes)
                if self.plan_cand > 0 and self.imag_k > 0:
                    r = C.reshape(B * T, self.d)
                    roll = []
                    for _ in range(self.imag_k):
                        r = self.plan_step(r)
                        roll.append(r)
                    imag = torch.stack(roll).mean(0).reshape(B, T,
                                                             self.d)
                    C = C + self.imag_gate * imag
        logits = self.head(self.lnf(C))
        logits_own = logits                               # the cortex's belief before memory
        if hasattr(self, "face_head"):
            feat_f = self.lnf(C)
            face = self.face_head(feat_f).squeeze(-1) * 6.0            # [B, T], face units
            self._face = face.detach()
            self._face_feat = feat_f.detach()
            if face_target is not None:
                self._face_loss = nn.functional.mse_loss(
                    face, face_target.to(face.device, face.dtype).clamp(-6.0, 6.0))
        G = st.get("G")
        if G is not None and G.abs().sum() > 0:
            # THE GOAL ORGAN speaks in identity space, like the store:
            # the attended pursuit vector is matched straight against
            # the vocabulary, so the contribution generalizes by
            # construction; goal_query learns WHEN to attend, the
            # zero-init gate learns HOW LOUD (and receives gradient
            # even at zero because the read is always computed).
            qg = nn.functional.normalize(self.goal_query(C), dim=-1)
            Gn = nn.functional.normalize(G, dim=-1)
            att = torch.softmax(qg @ Gn.transpose(1, 2) * 8.0, dim=-1)
            goal_read = att @ Gn
            logits = logits + self.goal_gate * (goal_read @ lg_E.t())
        self._aux_hidden = None
        # the hippocampus is a PFC organ: keyed and queried by the
        # council's token slot (in cortex_first that is the cortex output)
        R = self._read(st, bag_incl if kc else S0, read_ok)   # batched, [B, T, d]
        rd_full = None
        if R is not None:
            if self.aux_trunk > 0 and self.training:
                self._aux_hidden = C
            rd_full = R @ lg_E.t()
            # A READ THAT CAN STEER: when the trunk is unsure of the next
            # word (high entropy), the hippocampus speaks louder —
            # read_beta = 0 is exactly the trained body (the store's
            # vote of ~2 logits lost to a confident trunk every time)
            rb = float(getattr(self, "read_beta", 0.0) or 0.0)
            if rb > 0.0:
                with torch.no_grad():
                    pz = torch.softmax(logits.float(), dim=-1)
                    ent = -(pz * (pz + 1e-9).log()).sum(-1, keepdim=True)
                    gain = 1.0 + rb * ent / math.log(max(2, logits.shape[-1]))
                rd_full = rd_full * gain.to(rd_full.dtype)
            _sb = getattr(self, "store_boost", 1.0)
            if _sb != 1.0:
                # 49q sparse boost: amplify only memory's top suggestions
                # per position — the recall megaphone without shouting
                # over ordinary language (flat boost cost: CE 6.9->17.7
                # at 8x; sparse keeps the background whisper intact)
                k = 8
                thr = rd_full.topk(k, dim=-1).values[..., -1:]
                mask = (rd_full >= thr).float()
                _vmin = float(getattr(self, "store_boost_min", 0.0) or 0.0)
                if _vmin > 0.0:
                    # memory speaks up only when it is sure: a faint vote is
                    # left as a whisper, not amplified into a sentence
                    mask = mask * (rd_full > _vmin).float()
                logits = logits + rd_full + (_sb - 1.0) * rd_full * mask
            else:
                logits = logits + rd_full
        self._aux_logits = None
        self._route_aux = None
        if getattr(self, "_route_deep", None) is not None and self.ponder_aux > 0 and self.training:
            C_deep, idx = self._route_deep
            lg_deep = self.head(self.lnf(C_deep))
            if rd_full is not None:
                lg_deep = lg_deep + rd_full.reshape(B * T, -1)[idx]
            self._route_aux = (lg_deep, idx)
        if wcost:
            self._write_cost = torch.stack(wcost).mean()
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
        if kc:
            h_prev_live = bag_excl.to(S0.dtype)            # the words before x_t, as they are
        else:
            h_prev_live = torch.cat([st["prev_c"].unsqueeze(1), S0[:, :-1]], dim=1)
        h_prev_det = h_prev_live.detach()
        smask = torch.ones(B, T, device=dev)
        if getattr(self, "store_write_off", False):
            smask = smask * 0.0                            # the ear writes, the mouth does not
        if st["chunk"] == 0:
            smask[:, 0] = 0.0                              # nothing before
        dopa = None
        if self.dopamine > 0 or self.plasticity > 0:
            dopa = torch.stack([torch.zeros(B, device=dev) if r is None else r for r in rpe], dim=1)
        if self.dopamine > 0:
            # the dopamine gain per token rides the write mask: values > 1
            # are applied after the strength sigmoid (clamped to 1 below)
            smask = smask * (1.0 + self.dopamine * dopa)
        self._dopa_trace = None if dopa is None else dopa.detach()
        if self.plasticity > 0:
            # SIGN-AWARE (2026-08-23): a dopamine BURST teaches the
            # interval harder, a DIP teaches it less (LTD) — the old
            # 1 + k|delta| doubled the weight on the interval holding the
            # child's wrong line at a -2 press (the mistake learned
            # harder; only the night's pairs undid it). Weights floor at
            # 0 and keep mean 1.
            dopa_s = torch.stack([torch.zeros(B, device=dev) if r is None else r for r in rpe_s], dim=1)
            w = (1.0 + self.plasticity * dopa_s.detach()).clamp(min=0.0)
            self._ce_weights = w / w.mean().clamp(min=1e-6)   # mean 1: the learning rate is unchanged
        if self.write_surprise > 0 or self.intrinsic_w > 0:
            # surprise of x_t under the cortex's own head at t-1 (position
            # 0 uses the previous chunk's last row, carried detached)
            with torch.no_grad():
                lo = logits_own.detach().float()
                sp = -torch.log_softmax(lo[:, :-1], -1).gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)   # [B, T-1]
                pl = st.get("prev_logits")
                if pl is not None:
                    s0_ = -torch.log_softmax(pl.float(), -1).gather(-1, tokens[:, :1]).squeeze(-1)
                else:
                    s0_ = torch.full((B,), float(self.surp_mu), device=dev)
                surp = torch.cat([s0_.unsqueeze(1), sp], dim=1)                                             # [B, T]
                if self.training:
                    m_ = surp.mean()
                    if float(self.surp_n) == 0:
                        self.surp_mu.copy_(m_)
                    else:
                        self.surp_mu.mul_(0.99).add_(0.01 * m_)
                    self.surp_n.add_(1)
                st["prev_surp"] = surp                    # 49i intrinsic
                st["prev_logits"] = lo[:, -1]
                if self.write_surprise > 0:
                    gate = torch.sigmoid(
                        (surp - self.surp_mu) / self.write_surprise)
            if self.write_surprise > 0:
                self._surprise_gate = gate
                smask = smask * gate
        unwrite = None
        if self.press_unwrite and self.eot_ids is not None:
            # the graded model turn before each NEGATIVE press (levels 3/4):
            # press at p, tokens[p-1] = <eot_model>, the turn runs back to
            # the previous <eot_human>; in-chunk positions -> strength 0;
            # the part in the previous chunk -> a negative re-issue
            eot_h, eot_m = self.eot_ids
            neg = (lev >= 3)
            if bool(neg.any()):
                tok_cpu = tokens.detach().cpu()
                neg_cpu = neg.cpu()
                prev_w = st.get("prev_write")
                tkp_all = prev_w[1].cpu() if prev_w is not None else None
                for b in range(B):
                    for p in torch.nonzero(neg_cpu[b]).flatten().tolist():
                        j = p - 1
                        in_prev = False
                        if j >= 0:
                            if int(tok_cpu[b, j]) != eot_m:
                                continue                   # no model turn ends here
                            start = None
                            for q_ in range(j - 1, -1, -1):
                                if int(tok_cpu[b, q_]) in (eot_h, eot_m):
                                    start = q_ + 1
                                    break
                            in_prev = start is None        # the turn began in the previous chunk
                            smask[b, (start or 0):j + 1] = 0.0
                        else:
                            # the press opens the chunk: the turn ended exactly at
                            # the previous chunk's last token
                            if tkp_all is None or int(tkp_all[b, -1]) != eot_m:
                                continue
                            in_prev = True
                        if in_prev and tkp_all is not None:
                            tkp = tkp_all[b]
                            n_p = tkp.shape[0]
                            q_ = n_p - 2 if j < 0 else n_p - 1      # skip the turn's own <eot_model>
                            ps = 0
                            while q_ >= 0:
                                if int(tkp[q_]) in (eot_h, eot_m):
                                    ps = q_ + 1
                                    break
                                q_ -= 1
                            if unwrite is None:
                                unwrite = []
                            unwrite.append((b, ps, n_p))
        st["prev_c"] = S0[:, -1].detach()
        st["chunk"] += 1
        if st["chunk"] % self.write_every != 0:
            st["wbuf"].append((h_prev_det, tokens, smask, S0.detach()))
            self._recon = None
            if self.store_wipe == "day" and day_lanes:
                self.wipe_stores(st, day_lanes)
            return logits, st, ticks
        buf = st["wbuf"]; st["wbuf"] = []
        toks_all = torch.cat([b[1] for b in buf] + [tokens], dim=1)
        sm_all = torch.cat([b[2] for b in buf] + [smask], dim=1)
        hp_det = torch.cat([b[0] for b in buf] + [h_prev_det], dim=1)
        hp_live = torch.cat([b[0] for b in buf] + [h_prev_live], dim=1)
        n_neg = 0
        if unwrite:
            # re-issue the previous chunk's graded-turn rows at NEGATIVE
            # strength (detached keys: an un-write teaches nothing)
            hp_p, tk_p, sv_p = st["prev_write"]
            sm_neg = torch.zeros_like(sv_p)
            for b, ps, pe in unwrite:
                sm_neg[b, ps:pe] = -1.0
            rows = sm_neg.abs().sum(0) > 0
            if bool(rows.any()):
                keep = torch.nonzero(rows).flatten()
                n_neg = int(keep.numel())
                toks_all = torch.cat([tk_p[:, keep], toks_all], dim=1)
                sm_all = torch.cat([(sv_p * sm_neg)[:, keep], sm_all], dim=1)
                hp_det = torch.cat([hp_p[:, keep], hp_det], dim=1)
                hp_live = torch.cat([hp_p[:, keep], hp_live], dim=1)
        V_id = lg_E[toks_all]
        recon = []
        for pass2 in (False, True):
            tu = self.tok_u.clone() if pass2 else self.tok_u
            Wk = self.key_proj.weight.clone() if pass2 else self.key_proj.weight
            hp = hp_det if pass2 else hp_live
            k_d = hp if kc else nn.functional.normalize(hp @ Wk.t(), dim=-1)
            sv = (torch.sigmoid(tu[toks_all]) * sm_all).clamp(max=1.0)
            if not pass2 and self.plan_m > 0 and self.training:
                # REM seed (49b): the live chunk's write strengths — the
                # dream starts where the day wrote hardest
                self._last_sv = sv[:, -tokens.shape[1]:].detach()
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
        if self.press_unwrite:
            # what this write put in (for a later un-write): the live
            # chunk's rows only, strengths as applied (sign included)
            n_new = tokens.shape[1] * (len(buf) + 1)
            st["prev_write"] = (hp_det[:, -n_new:].detach(), toks_all[:, -n_new:],
                                sm_all[:, -n_new:].detach())
        if self.store_wipe == "day" and day_lanes:
            self.wipe_stores(st, day_lanes)            # the day closed: its traces go, the bands stay
        return logits, st, ticks

    def dopa_trace(self):
        """[B, T] |RPE| per token of the last chunk (dopamine), or None."""
        return getattr(self, "_dopa_trace", None)

    def read_value(self, st):
        """serve-time read-only gauge: each unit's value-head estimate
        of upcoming press-reward from the CURRENT state, press units.
        This is its own felt expectation, not a proxy."""
        out = {}
        with torch.no_grad():
            for u in self.ukeys:
                try:
                    out[str(u)] = float(
                        self.value[str(u)](st["h"][u]).squeeze(-1).mean())
                except Exception:
                    pass
        return out

    def pop_value_loss(self):
        c = self._value_loss
        self._value_loss = None
        return c

    def pop_face(self):
        """its face per token of the last forward (a forecast of the
        caretaker's face, face units) and the detached features it was
        read from — for the serve's online lesson at every token."""
        return self._face, self._face_feat

    def pop_face_loss(self):
        c = self._face_loss
        self._face_loss = None
        return c

    def pop_bg_loss(self):
        """The basal-ganglia actor term, already weighted by bg_w (None at bg_w=0)."""
        c = getattr(self, "_bg_loss", None)
        self._bg_loss = None
        return c

    def gate_trace(self):
        """[B, T] mean band-3 gate per token of the last chunk, or None."""
        return getattr(self, "_gate_trace", None)

    def set_eot_ids(self, eot_human, eot_model):
        """the turn markers (press_unwrite walks back over the graded
        model turn); None leaves un-writes off."""
        if eot_human is None or eot_model is None:
            self.eot_ids = None
        else:
            self.eot_ids = (int(eot_human), int(eot_model))

    def wipe_stores(self, st, lanes):
        """store_wipe: zero the given lanes' store matrices (both ladders)
        and any buffered keys of theirs; band states untouched."""
        if not lanes:
            return
        idx = torch.as_tensor(sorted(set(int(b) for b in lanes)), device=st["prev_c"].device)
        for k in self.bands:
            st["M"][k] = st["M"][k].index_fill(0, idx, 0.0)
            if st.get("CM") is not None:
                st["CM"][k] = st["CM"][k].index_fill(0, idx, 0.0)
        st["wbuf"] = [(hp, tk, sm.index_fill(0, idx, 0.0), s0) for (hp, tk, sm, s0) in st["wbuf"]]
        if st.get("prev_write") is not None:
            hp, tk, sv = st["prev_write"]
            st["prev_write"] = (hp, tk, sv.index_fill(0, idx, 0.0))

    def set_reward_tokens(self, levels):
        """levels: {token_id: level} with level 1..4 = <+1>, <+2>, <-1>, <-2>."""
        self.reward_lut.zero_()
        for tid, lv in levels.items():
            if tid is not None and 0 <= int(tid) < self.reward_lut.shape[0]:
                self.reward_lut[int(tid)] = int(lv)

    def ban_presses(self, logits):
        """A64: the mouth never says a grade — the press ids (the LUT's
        nonzero rows) get -inf in any sampled/greedy path."""
        ids = torch.nonzero(self.reward_lut, as_tuple=False).flatten()
        if ids.numel():
            logits = logits.clone()
            logits[..., ids] = float("-inf")
        return logits

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
