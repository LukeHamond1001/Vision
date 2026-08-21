# Memory math: why the organs are passengers, measured

2026-08-21. Equations are the ones in `iga/lm_hybrid.py`; numbers are
from the 78M raised life's weights (`v94sp`, step 296k) on 49k tokens
of UltraChat held-out dialogue in its own tokenizer
(`results/evidence/memory_math_v94sp.json`, script
`memory_math.py`). Nothing here is a guess about a bigger model — it
is what this architecture learned at 78M over 296k steps, and the
mechanisms scale unless something breaks them.

## 1. The two questions

An organ is load-bearing iff (a) the objective cannot be met without
it at a weight that matters (**necessity**) and (b) the organ's path
to that objective is learnable before the cortex finds a cheaper one
(**ability**). Both are measurable.

## 2. Necessity exists, and it is dense

Attention sees one chunk of T = 2048 tokens (no XL carry). So the
first tokens of every chunk are predicted with no context unless an
organ carries it. CE by position in the chunk, state carried across
chunks:

| position | 0–16 | 16–64 | 64–256 | 256–1024 | 1024–2048 |
|---|---|---|---|---|---|
| CE (nats) | **5.06** | 4.27 | 3.90 | 3.80 | 3.51 |

The boundary deficit is 1.55 nats on the first 16 tokens, 0.76 on the
next 48, 0.39 on the next 192. Integrated over the chunk it is ~0.18
nats per token — about **5% of the total CE (3.68)** — available to
any organ that carries the previous chunk's tail. That is a large,
dense gradient, present at every chunk boundary, with no cast asks
needed. The belief that "nothing beyond the window matters for
next-token prediction" is false by 5%.

## 3. What the organs recover of it

Same stream, same state carried, each removal applied on the read
path (ΔCE vs base, nats):

| removal | 0–16 | 16–64 | 64–256 | all |
|---|---|---|---|---|
| thread off (`mem_off`: band tokens zeroed, stores on) | **−0.004** | −0.002 | −0.001 | −0.001 |
| stores off (`store_read_off`) | +0.006 | +0.004 | +0.003 | +0.001 |
| both off (`lesioned` all) | +0.002 | +0.003 | +0.002 | 0.000 |

The band memory tokens contribute **nothing** — removing them makes
the boundary very slightly *better* (they are noise to the cortex).
The stores contribute +0.006 nats at the first 16 tokens: **0.4% of
the deficit**. The organs built for the boundary do not meet it.

## 4. The store: why it became a bigram cache (ability, failed)

As implemented, at position t the key is a mix over the 64 preceding
tokens,

    a_{t,r} = softmax_r( qmix[r] + tok_u[x_{t−r}] ),   r = 0..63
    mix_t   = normalize( Σ_r a_{t,r} · E[x_{t−r}] )
    key_t   = normalize( cos(P · mix_t + φ) )          (frozen RFF lift, R^D)

The write at t stores (key from the context STRICTLY before t, value =
identity of x_t) with strength s_t = σ(tok_u[x_t]); the read at t
queries with the context up to and including x_t and adds
α_band · M · q to the logits. This is the induction shape: "what
followed this context before". It is a good design *if the key can be
an entity when it needs to be.*

Measured: `qmix` softmax = **[0.9992, 0.0007, 0, 0, …]**, entropy
0.006 nats (uniform would be 4.16). The key is the single preceding
token. Writes are "x_{t−1} → x_t", reads are "x_t → what followed x_t
before": a bigram cache. And `tok_u` — which sets both a token's
salience in keys and its write strength — is **−0.86 for colours**
against a mean of −0.37: the model learned to *suppress* the cast's
answer words from the store.

Why gradient descent goes there, in three steps:

1. **Two objectives, one key.** Induction (repeated bigrams within a
   document) pays at a large fraction of positions; entity recall pays
   at ~1 ask per 350 tokens. `qmix` is a single global vector over
   offsets — one mixing pattern for every position — so the key can
   be "the previous token" or "the salient entity in the window", not
   both. The dense objective wins the race.
2. **Saturation locks it.** ∂L/∂qmix_r carries the softmax Jacobian
   a_r(1 − a_r); at a_0 = 0.9992 that factor is 0.0008. Every later
   gradient toward an entity key is multiplied by ~10⁻³. The local
   optimum is a trap by construction of the softmax.
3. **Interference makes the cast toxic.** Under a bigram key, "was →
   red" and "was → blue" collide in the delta rule; storing colours
   *hurts* the cache, so their write strength and salience are
   pushed down (tok_u colours −0.86). The store un-learns the very
   values the demo needs.

The fix is not a wider store or a longer horizon; it is a key that
can differ by position: **content-keyed** — key_t = lift(W_k · h_t)
with h_t the trunk's hidden at t (the attention layers can put
"Mira" and "key" into h_t at the plant and at the ask, and the
previous token into it elsewhere). That is the fast-weight /
linear-attention key; the RFF lift, the identity values, the delta
rule and the A38 two-pass credit all stay. Capacity is unchanged
(~D pairs per band; band 5's D ≥ 2048 against ~100–200 live episodic
facts per 131k tokens).

## 5. The bands: why the cortex never reads them (ability, failed)

Per chunk, band k reads the chunk with its own query and accumulates:

    w = softmax(hidden · q_k / √d),  read_k = Σ_t w_t hidden_t,  acc_k += read_k

Every `clock[k]` chunks: pooled = mean(acc), fidelity
fid_k = cos(pred_k(h_k), pooled), then h_k ← SlowCell(pooled, h_k)
(gated delta write, gate bias −2 at init), pend_k ← pred_k(h_k). The
cortex sees m_k = mem_proj_k(h_k) as one attended token.

Measured: the SlowCell gate biases after 296k steps are **−1.62 /
−2.02 / −2.03** for bands 3/4/5 (init −2.0): bands 4 and 5 received
no effective gradient in a third of a million steps. The read gates
are at exactly σ(−2) = 0.119, untouched. Three mechanisms:

1. **The fixed point.** The only content-shaping gradient into h_k
   is CE through attention on m_k. A constant token carries no
   information, so attention learns to ignore it; ignored, it gets no
   gradient; with no gradient its content stays constant. fid_k and
   the write cost keep the machinery alive, not useful.
2. **The fidelity target is the wrong statistic.** pooled is a mean
   of 2048 hidden states; transformer hiddens share a dominant
   direction, so any two chunk means have cos ≈ 0.97 — band 3's
   fid:3 ≈ 0.97 is the anisotropy floor, not content. And a mean over
   2048 positions cannot carry *the last sentence before the
   boundary*, which is what the first 16 tokens of the next chunk
   need (section 2). The band summarizes the wrong thing for the one
   dense need that exists.
3. **Tick starvation.** Band k updates 1/clock[k] as often; with Adam
   each update moves a bounded step, so total movement scales with
   the update count: band 4 gets 1/8 of band 3's, band 5 1/64, band 6
   1/512. The band-lr arm (3×) is this law seen from the other side
   (fid:4 .17 → .41 at zero CE cost). Compensation by lr alone is
   capped by Adam stability well below ×64; the rest must come from
   more ticks (shorter window × faster clocks — the conveyor; longer
   lives) or a slower-band objective that does not need many ticks.

Why the XL carry is not simply switched back on: it was built (A30)
and removed by evidence (A33 — it tripled cross-boundary recall and
*halved* in-chunk binding: "an easier path crowds out the
generalizing circuit"). Cross-chunk duty went to the bands and the
store, which — section 3 — never took it up.

## 6. What the math prescribes, in order

1. **Content-keyed store** (section 4) — removes the one-key-two-jobs
   conflict at the root. Gate at d=128 on the episodic shards with
   the same instruments (in-ctx / short / b3, the boundary meter);
   ship only on a win.
2. **A tail organ for the boundary** — the dense need is the last
   ~64 tokens before the cut. Cheapest: band 3's read weighted to the
   tail (its query is learned; a position-aware read can select the
   end of the chunk), or one extra memory token = the mean of the
   last 64 hiddens. Measured against the 0.18 nats/token available.
   This is what makes Demo 1's organ *necessary* at every boundary,
   without shrinking the window.
3. **Band objectives that carry content**: predict the *next chunk's
   early hiddens* (what the boundary needs), not the chunk mean; and
   per-band lr within Adam's limits, with ticks made up by the
   conveyor or longer lives for bands 5–6.
4. **The necessity meter in the battery** (`probe: boundary`, built
   tonight): deficit at 0–16 / 16–64 and each removal's delta. An
   organ counts as load-bearing when its removal raises the early
   CE; the number to watch is the fraction of the deficit recovered.

## 7. What this means for the demos

On the evidence, the architecture as certified would give two null
removal demos at 500M on this diet: the stores are a bigram cache
that avoids the cast, and the band tokens are unread. Tonight's four
minis test whether the episodic diet, the modern trunk, the band lr
and the forced window change that at 78M — with the boundary meter
and the store-health probe reporting every beat. If they do not, the
content-keyed store and the tail organ are the program before the
$180 is spent, and the served-life claim (sleep + economy + weights,
the organs that do work) is what V10.1 can honestly carry alone.
