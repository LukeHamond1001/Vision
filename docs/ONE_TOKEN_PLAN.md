# The one-token organism — plan (2026-08-21, user-ratified design)

The user's spec, verbatim intent: *all input goes into the PFC with its
time bands; the PFC outputs one thing — a bundle of embeddings — and
those feed the rest of the neocortex (the standard net), which makes
the output token. One model at a time, one iteration at a time. bf16
for the neocortex, fp32 for the PFC. Pod: RTX 2000 Ada or 4090.*

## 1. The organism (iga/lm_scan.py, order = pfc_first)

```
token x_t --> e_t (fp32)
   |
   v
PFC (fp32): slots [e_t, r_t, m_3 .. m_8] + slot embeddings
            -> n_council attention blocks -> bundle S'_t  (8 slots x d)
   |                                   r_t  = hippocampus read slot
   |                                   m_k  = band k's state, projected
   v
NEOCORTEX (bf16 autocast): n_layers ScanBlocks; query = S'_t[0],
            keys/values at every layer = the whole PFC bundle S'_t.
            It never sees x_t directly.
   |
   v
head (fp32) -> logits + hippocampus logit read (fp32) -> next token
```

Between tokens (all fp32): band k accumulates its PFC slot S'_t[k];
ticks every clock_k TOKENS with the veto permission from the other
bands' slots (band 3 every token, 4: 8, 5: 64, 6: 512, 7: 4096,
8: 32768); the hippocampus (content-keyed LogitStore ladder, KD
512..4096) is written every `write_every` chunks from the neocortex
hidden and its read is refreshed into the PFC slot every `slot_every`
tokens. Full BPTT inside the T=64 chunk; A38 cloned carry across it.

Two code changes vs commit 7faee62:
- **precision law**: trunk blocks under bf16 autocast; council, cells,
  vetoes, predictors, mem_proj, band states, store, head, logits, loss
  in fp32 (the council used to share the trunk's autocast). Law S9.
- **pfc_first feeds the bundle**: `_trunk(S'[:,0], S'[:,1:])` (it used
  to pass no slots). Law S7 updated: the neocortex still never sees e_t.

## 2. One model at a time

Every iteration = ONE config, ONE pod lifetime, ONE change from the
previous iteration, read before the next. No batteries.

Iteration 1 (the design as specified):
  arch=scan order=pfc_first d=512 n_layers=8 n_council=2 T=64 lanes=32
  clocks 3:1,4:8,5:64,6:512,7:4096,8:32768 slot_every=8 write_every=4
  precision bf16 (trunk) / fp32 (PFC) lr 1e-4 warmup 1000 lesion-every 2
  shard: mini_epi_l32 (32 lives, ~200M tokens) from the prep volume's
  sources; eval: mini_eval_epi (2 unseen lives).

Pre-registered reads (heartbeat rows on results-v10, mini_hb_scan1.jsonl):
  R1 language: eval CE at 12M tokens vs the hybrid mini's 5.91 at the
     same d/L/token count (mule-1 row, step 1500). Bar: <= 6.2.
     Larger gap = the bands are not carrying the context yet.
  R2 dependence: lesion deltas (bands 3..8 off, store read off) on CE
     and on recall-by-gap. Bar by 100M tokens: band 3 and 4 deltas
     > +0.05 CE; store-read-off delta > 0 on in-ctx/short recall.
  R3 throughput: >= 8k tok/s at 32 lanes on the 4090. Below it, the
     next iteration is CUDA graphs, nothing else.
  R4 health: no NaN; veto mean in (0.05, 0.95); write cost and fid
     channels finite and moving.

Iteration 2+ candidates (pick ONE from iteration 1's read, in order):
  a. cortex_first control at matched tokens — only if R1 fails by
     > 0.3 (does the strict order itself cost language?)
  b. slot_every=1 (the hippocampus into the PFC every token)
  c. n_council=4 (a deeper PFC)
  d. 64 lanes (the per-token step is launch-bound; lanes are ~free)
  e. CUDA-graph capture of the token step (R3)
  f. bands 9/10 (262k / 2M tokens) for the 500M ladder

Endgame: the 500M (d=1280 / 20L / T=64 / 64 lanes) on an H100 after
>= 2 iterations pass R1-R4; cost re-estimated from the 4090's measured
tok/s (1.5-2x a transformer's per-token cost is the prior). That
launch is a separate, explicit go.

## 3. Pod

RTX 4090 (24 GB, $0.34/hr, EU-RO-1) on the PREP network volume
2o9gtwzkhd — the corpus, shards, sources and the 24 GB ship tar are
already there. Fallback: RTX 2000 Ada (16 GB, $0.50/hr). Boot by
dockerEntrypoint from a SHA-pinned raw URL (scripts/pod_scan.sh);
heartbeats + rows pushed to results-v10; the pod removes itself when
the job ends so an iteration costs ~$2-3 (~6-8 h at 8-15k tok/s).
Launcher: scripts/launch_pod.sh (secrets read at run time from
~/.runpod_key and `gh auth token`; never stored in the repo).

## 4. Timeline

- Aug 21 night: code + laws (S7, S9) + plumbing + launcher; boot the
  4090; build mini_epi_l32; iteration 1 running.
- Aug 22: read iteration 1 (R1-R4); launch iteration 2.
- Aug 23: iteration 3 or the 500M decision.

## 5. Ledger of what was lost today

- mule-1 (4gzvaq7jrokt31) ended after "ship tar ready 24G" 22:17 UTC;
  its last lines never pushed. Its hybrid mini OOM'd at step ~1500 on
  the 16 GB card (store read einsum at T=2048). The l4 shard, the
  corpus and the tar live on 2o9gtwzkhd (verify on the next boot).
- local gate arms scan/scanpfc/win64 died at step ~120 with the
  session restart — no result; iteration 1 replaces them.
- the scratch launch scripts (launch_mule2.sh, swap_mule.sh) were
  wiped with the session; replaced by scripts/launch_pod.sh in-repo.

## 6. Iteration 1 smoke (23:39 UTC) and the iteration-2 decision

Pod affvbon31255ol (4090 secure, $0.74/hr — the $0.34 was the
community-cloud quote; the volume's DC prices secure). Shard
scan_epi_l32 built in 5 min: 200M tokens, 32 lives x 6.25M, 765k
events, 19.9k corrections. Smoke at the exact config: **2166 tok/s,
peak 17.9 GiB, 67.0M params**, holds 23.2 -> lam 0.0108, CE 9.84 ->
6.81 in 40 steps. R3 FAILS at the smoke (bar 8k): ~25 h / ~$19 for
the 200M run.

Why: the per-token loop is kernel-launch bound and the 8-block
neocortex runs inside it. In the user's order the neocortex is NOT on
the recurrent path — band states depend only on the council outputs;
the only feedback from the cortex is the hippocampus query (and the
fidelity target, which is a loss term). So:

Iteration 2 (pre-registered now, built while scan1 runs its first
beat): the HIPPOCAMPUS IS A PFC ORGAN — keys, slot-refresh queries
and logit-read queries all come from the council's token slot S'_t[0]
(in cortex_first that is what c_out already was, so both orders
agree) — and the NEOCORTEX RUNS ONCE PER CHUNK over the 64 bundles
(batched B x T, same math, a parity law). The band fidelity targets
are computed after the decoder from C's interval slices (carried
partial sums across the boundary). Expected: ~2x fewer launches; the
council loop (2 blocks + cells per token) is the remaining recurrent
cost and the CUDA-graph candidate after that.

scan1 keeps running until its first row (step 6000, 12.3M tokens —
the R1 read) lands, then scan2 replaces it (one model at a time).

## 7. scan1 in flight (step 1300, 23:56 UTC): the drift artifact

scan1 trains at 2.77k tok/s (CE 6.51 at 2.7M tokens). Its fid channels:
fid:3 +0.84, fid:4 +0.39, **fid:5 +1.000**. Band 5 ticks once per
chunk; its target was the chunk-mean of the cortex minus a LAGGING
running mean — during early training the cortex mean drifts, the
residual is the drift direction, identical in every lane, and a
predictor bias matches it. The EMA centring re-created the anisotropy
floor the hybrid's band_center was meant to remove. Fixed for scan2
(commit c6ba686): in training the target is centred by the batch mean
at the tick (the drift cancels exactly; the residual is how this
lane's context differs from the others — carryable only by memory);
the per-band running mean stays for eval/serve. S10 extended.

scan2 = iteration 2 = commit c6ba686: batched decoder + PFC-keyed
hippocampus + instantaneous centring. Everything else identical to
scan1. Reads: R3 first (tok/s at the smoke), then R1 against scan1's
step-6000 row (the one A/B we get for free: cortex-keyed vs PFC-keyed
hippocampus, per-token vs batched decoder being exact).

## 8. Iteration 3 (named with the user, 00:20 UTC): the hippocampus in the PFC cycle

Option 4 (the cortex attending over past PFC outputs) is OUT: cortex
in the brain reads its own recent activity, the PFC biases it, the
hippocampus binds the immediate past — a PFC log in the cortex would
relieve the pressure that makes the PFC learn to guide. The user's
bet stands: no crutch.

Iteration 3 = scan2 + two cadence knobs, nothing else:
  write_every 4 -> 1   (the hippocampus written every chunk; per
                        token afterwards if the cost allows)
  slot_every  8 -> 1   (the PFC reads the hippocampus every token)
The only change between scan2 and scan3, so a CE move is attributable.
A 2-second (8-token) cortex self-window is the ONLY brain-faithful
cortex-side fallback and is user-gated; not built.

## 9. scan1's first row (step 6000, 12.3M tokens, 02:10 UTC) — evidence results/evidence/scan1/

  R1  eval CE 6.192 (bar 6.2: PASS; hybrid mini 5.91 at the same point,
      gap +0.28). Recall bins equal the hybrid's (in-ctx .195 vs .20,
      short .205 vs .196, b3 .218, b4 .188) — both at floor this early.
  R2  (early, from the boundary probe) bands off = +0.54 nats over the
      chunk. The hybrid's lifetime all-bands removal cost +0.26 PERCENT.
      The organism depends on its bands by construction, and they carry
      real information already. Chunk-boundary deficit 0.08 (hybrid
      1.55). Store alpha 0.045 -> 0.20: the hippocampus vote is used.
  R3  2.1-2.7k tok/s (FAIL, known) — the battery at 2 lanes took ~45
      min at HBC=16000; scan2 runs HBC=4000.
  R4  clean: no NaN, collapse sampled .99 / greedy .19, entropy 6.9;
      the drift artifact (fid:5 1.000, fid:6 .998 at the record) is
      the thing scan2 removes.
scan1 killed at step ~7000 (train CE 4.85), ~2.6 h, ~$1.9. scan2 =
pod qa1m3h985uaytt, sha e37e88e, launched 02:13 UTC.

## 10. scan2 smoke (02:14 UTC): the decoder was not the bottleneck

scan2 smoke: 2188 tok/s (scan1 2166), peak 14.9 GiB (17.9), holds 91
-> lam 0.0027. The batched decoder removed 80% of the loop's kernel
launches and changed nothing: the cost is in the council+band loop's
per-op overhead and the drive economy's Python (scan1 ran 2840 tok/s
at step 100 and decayed to ~1800 as its ledger filled to the 200k
cap, then recovered to 2700 when the segment reset it). The batching
stays (bit-exact, cheaper in memory, the structure we want).

scan2 runs to its second beat (step 12000 ~ 04:45 UTC: the first
per-band lesion pass = R2 proper, and R1 vs scan1's 6.192). Meanwhile,
locally: count the ops per token with the profiler, batch the six
bands' per-token work into single tensors (acc/cnt/slots/vetoes),
measure; scan3 = hippocampus cadence + the bit-exact speedups.

## 11. The speed work, measured properly (03:10 UTC)

My first profiles ran the DEFAULT order (cortex_first, per-token
decoder) — wrong organism; corrected. At the pod's order (pfc_first):
~97 forward ops per token, ~21k kernels per step with backward, ~29 us
each on the 4090 = the 0.6 s/step observed (3.4k tok/s). Shares:
council 29%, hippocampus read 8%, loop bookkeeping 60% (the fidelity
scored tick by tick was ~20 ops/token by itself).

Commit d74b356: bands batched (one add / one bmm / one veto matmul),
fidelity scored per band in one shot, opt-in compile_council (S13).
Ops per step 21.0k -> 17.2k. One SEMANTIC correction rides along and
is ledgered as a fix, not a design change: the fidelity loss now has
one entry per band per chunk (the hybrid's equal-weight semantics);
the per-tick port had let band 3's 63 ticks outweigh band 5's one
(63:1), starving the slow bands' objective.

scan3 = cadence (write_every 1, slot_every 1) + d74b356 + compile_
council (default mode; if the pod's torch.compile fails, the smoke
dies in ~2 min and scan3 relaunches without it). The pod's smoke
reports tok/s: scan2 3.4k is the reference. Launch after scan2's
step-12000 row (~05:00 UTC) — never two pods on the volume.

## 12. scan2 behind scan1 on train CE (02:46 UTC) — the fidelity coupling

scan2 train CE 6.87 at step 2500 vs scan1 6.17 at 2300. Diagnosis,
verified by law S14: once the fidelity target was honest (batch-mean
centred), the fidelity loss — satisfied for free in scan1 by the drift
artifact — pulled the PFC and the decoder through two live paths the
port had left open: the tick's pend (live pooled -> council) and the
target (live cortex output). The certified hybrid trained the cell and
predictor on DETACHED inputs (band_credit) and its live target sat at
the anisotropy floor, so neither force mattered there. Fix (0929965):
the fidelity prediction comes from the tick on detached inputs and the
target is detached — fidelity trains the band's cell and predictor
only; CE credit still flows into the council through the live state.

scan2's step-6000 eval row decides the cut: eval CE >> 6.19 confirms
the drag -> scan2 killed at once and scan3 launched (cadence + this
fix + the speed work + compile_council); eval CE ~ 6.19 -> the gap was
lam/noise, scan2 runs to its step-12000 lesion pass first.

## 13. scan2 cut at step 3400 (03:05 UTC); scan3 launched

scan2 was flat: train CE 6.90 / 6.87 / 6.87 at steps 1500 / 2500 / 3400
(scan1: 6.5 -> 5.8 over the same steps). Cut without waiting for the
eval row (~$0.7, 50 min). Evidence results/evidence/scan2/.
scan3 = pod et5a01eby0eb9p, sha 31184a2: write_every 1 + slot_every 1
(the hippocampus in the PFC cycle), fidelity trains the band only
(0929965), batched bands/fidelity, compile_council (default mode).
Reads: the smoke's tok/s (scan2 3.4k at step 100 = reference; compile
may fail -> relaunch without it), then train CE vs scan1 at matched
steps (6.5 @1300, 6.17 @2300, 5.77 @3300, 5.02 @6000), the step-6000
eval row vs 6.192 (R1), the step-12000 lesion pass (R2: per band and
STORE OFF — the hippocampus question).

## 14. scan3 step 1300 (03:21 UTC): the hippocampus in the cycle

scan3 train CE 4.255 at step 1300 (2.7M tokens); scan1 6.505, scan2
6.900 at the same step; the hybrid mini (full 2048-token attention)
reached 4.38 at 12M tokens. Speed 2.6k tok/s (the read every token
costs ~58 forward ops; compile_read a66dfeb is ready for scan4).
fid:3 = +0.43 honest (batch-centred; scan2 0.001): the band-3
predictor now forecasts the per-lane deviation of the next cortex
output. Leakage check: the logit read and the slot reads use st["M"]
BEFORE this chunk's write block — only previous chunks' pairs.
Pending: the step-6000 eval row (unseen lives, ~04:35 UTC) and the
step-12000 lesion pass (STORE OFF must now cost nats).

## 15. Two bugs found while scan3 runs (04:35 UTC) — scan4's reasons

1. THE THROUGHPUT DECAY was the drive ledger: a Python list capped at
   200k with `del ledger[:drop]` per settle. The scan settles hundreds
   of holds per step, so once capped every step shifted 200k entries
   hundreds of times (scan1/scan2/scan3 all decayed 2.7k -> 1.5k tok/s
   by step ~4000; scan1 recovered at its segment boundary = a fresh
   drive). Now a deque(maxlen): identical entries, order and
   ledger_base (S16), O(1).
2. SCAN ECONOMY HORIZONS: the scan branch set the economy's horizon to
   the clock itself in tokens — band 3's holds were due after ONE token
   (< a step) and expired unpaid every step; the hybrid's rule is
   max(4 x clock, 512) tokens. Fixed (S8). This changes the press
   economy's holds, not the model.
scan4 = scan3's model exactly + these two fixes + compile_read. Plan:
read scan3's step-6000 eval row (R1), then relaunch as scan4 and take
the step-12000 lesion pass (R2, store off) there, at full speed.

## 16. scan3's step-6000 row (05:15 UTC) — R1 passed, the HPC hypothesis falsified

  R1  eval CE 4.547 on unseen lives (scan1 6.192; hybrid mini with a
      2048-token window 5.91). PASS by 1.65 nats.
  R2  bands off +0.97 nats (scan1 +0.54). STORE OFF +0.003: the
      hippocampus is inert even written every chunk and read every
      token. The cadence hypothesis is FALSIFIED; scan3's gain over
      scan1 is the fidelity fix freeing the PFC. Recall bins at floor
      (in-ctx .196). Entropy 6.9 -> 1.2, greedy distinct-3 .58.
Why the store is inert (from its write rule, LogitStore.write): the
update is the strength-NORMALISED average of the chunk's pairs — one
fact is written at 1/64 strength per chunk; the store accumulates
frequent pairs (the hybrid's bigram-cache finding, same cause) and
cannot hold a one-shot item at readable magnitude. Next HPC
hypothesis (iteration 5): exact one-shot binding — the sequential
delta rule at full per-pair strength via DeltaNet's chunkwise-exact
algorithm (the A52 overshoot handled exactly, not by averaging),
decay per band unchanged.
scan3 killed at step 6700; scan4 = pod zm2a7e7cw5ntnv, sha 86b5011:
scan3's model + the ledger/horizon fixes + compile_read. Reads: the
speed at step 100 and at step 4000+ (no decay now), the step-12000
per-band lesion pass.

## 17. Iteration 5 built (05:40 UTC): the exact hippocampus

Commit 9336d3e: LogitStore.write_exact — the sequential delta rule
M_t = M_{t-1} + b_t (v_t - M_{t-1} k_t) k_t^T computed exactly for the
chunk ((I + L) U = b (V - K M0^T), M_T = M0 + U^T K; a 64 x 64
triangular solve, same cost as the averaged rule). ScanLM(store_exact
=True); default off keeps the hybrid bit-exact. Laws S17 (equals the
token-by-token rule to 1e-8; repeated keys without overshoot; one item
among 64 pairs read back at > 0.6 vs < 0.25 averaged) and S18 (in the
organism). Init strength b_t = sigmoid(beta=0) x sigmoid(tok_u=0) =
0.25 per pair (learnable; a stronger init is a later knob).

scan4 (05:24 UTC, zm2a7e7cw5ntnv): step 100 at 2.7k tok/s, holds 1.5k
(scan3 7.1k — the horizon fix). Read: the rate at step 4400 (scan3 had
decayed to 1.5k), the step-6000 row (R1 with the corrected economy).
Then scan5 = scan4 + store_exact; its store-off number is the
hippocampus verdict for this hypothesis.

## 18. Leak test and copyability (06:10 UTC)

A local d=32/1-layer organism with scan3's cadence reached train CE
0.30 by step 1500 on the local gate shard. Leak test on uniformly
random tokens: CE = log V (6.253) under every cadence — nothing about
token t reaches its own logits. The cause is the DATA: the local gate
shard (life_gate_bio_epi) is 72% copyable at k=4/w=256 (95% precision
on matches); its eval shard 10-26%. scripts/copyable.py measures it;
pod_scan.sh now reports the train and eval shards' copyability at boot
so every CE number carries its context. scan3's eval gain is not
copying: the hybrid it beat by 1.4 nats had a 2048-token attention
window and could copy anything within it.
The pod's throughput decay has no local reproduction (drive, sleeper,
prophet all bounded; the deque fixed only the capped-ledger part);
IGA_TIMING=1 (3b4125f) prints per-component ms/step on the pod's step
lines — scan5 runs with it.

## 19. scan4 cut at step 3000 (06:20 UTC); scan5 launched

scan4 reproduced scan3 exactly (CE 3.707 @2200, 3.351 @3000) and
decayed exactly the same (1,726 tok/s @3000) with its ledger at 48k —
the capped ledger was never the cause. Nothing more to learn from it;
cut. scan5 = pod obs1x8bc9psc3k, sha 6160c76: scan4's model +
store_exact (the exact one-shot hippocampus) + IGA_TIMING=1 (per-
component ms/step and gc counters on every step line) + copyability
of both shards at boot. Reads: the smoke, the timing lines at steps
100 / 1300 / 3000 (where the seconds go and whether gc2 grows), train
CE vs scan3's trace (4.26 @1300, 3.71 @2200, 3.35 @3000, 2.97 @6000),
the step-6000 row (R1 + recall bins), the step-12000 lesion pass
(STORE OFF — the verdict on one-shot binding).

## 20. THE DECAY FOUND (06:30 UTC): Python's garbage collector

scan5's first timing line: bwd 318 ms, fwd 113, everything else ~0 —
and 2,203,647 tracked Python objects with 15 full collections by step
100. The conveyor holds the shard's 765k event dicts; every gen-2
collection walks all of them (~0.4 s), and the collector fires more
often as the ledger, readings and holds grow — the cost lands inside
whatever phase allocates (fwd/bwd), which is why no component ever
showed it and no local run reproduced it at scale. Fix 51f6fb8:
gc.collect() + gc.freeze() once before the loop (the events, model and
optimizer live for the whole run) and gen-2 threshold x5. Local check:
tracked objects 370k -> ~600, step 167 -> 100 ms. scan5 cut; scan5b =
pod 162vgnlfdvjco8, sha 51f6fb8 — the same model (store_exact) with
the fix. Shards' copyability (pod): train 5.0% @k4/w256 (15% @w4096),
eval 14.5% — the CE numbers are language, not copying.

## 21. scan5b timing (07:05 UTC) — the rest of the decay, itemised

Steps 2100-2400, ms/step: bwd 381->432, fwd 217->242, detach 110->133,
sweep 20-26, opt 11, prophet 3, loss/events 5; total 748->846; gc2
every ~4 steps; tracked objects 298k->309k (post-freeze).
  - detach_readings rebuilt every readings list every step (only the
    fresh tail carries a graph) and sweep rebuilt them again for the
    cutoff prune: now O(new) + an amortised prune (2f3c8b6, S19 exact).
  - fwd/bwd creep: GC (gen-2 every ~4 steps over ~300k objects) and,
    likely, caching-allocator fragmentation (the tick pattern varies
    the allocation sequence; the pod ran without expandable segments)
    -> pod_scan.sh sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments.
  - still O(all readings per key) per hold in the sweep's rs filter
    (~20 ms) — a bisect on the time-ordered list; later.
scan5b keeps running (a relaunch would cost more than it saves); the
fixes ride scan6. CE trace scan5b = scan3's (4.257 @1300, 3.699 @2200).

## 22. scan5b's step-6000 row (08:05 UTC): THE HIPPOCAMPUS IS LOAD-BEARING

Exact one-shot store (store_exact), 12.3M tokens, unseen lives:
  STORE OFF +0.099 nats (scan3's averaged store: +0.003) — 33x.
  recall in-ctx .297 / short .297 (scan3 .196/.196; hybrid .20/.196):
  the first lift off the floor in the program (+50% relative).
  eval CE 4.475 (scan3 4.547); bands off +1.01; alpha .18 -> .23;
  tok_u .039 -> .054 (write strengths growing); entropy .84; greedy
  distinct-3 .42 (sampled .92).
The one-shot-binding hypothesis holds at first read. Next: the step-
12000 lesion pass (~09:15 UTC) — per band and store-off at 24.6M.
Evidence results/evidence/scan5b/.

## 23. scan5b's lesion pass (step 7500 = 15.4M tokens, 09:20 UTC): both demos exist

dCE when removed / in-ctx recall (base .389): band 3 +1.79 / .249;
bands 4,5,6,7,8 +.016 / +.014 / +.021 / +.024 / +.025 (all positive,
ordered by timescale, small); ALL BANDS +1.95 / .143; HIPPOCAMPUS
+.114 / .263 (a third of in-context recall); thread +1.77. Eval CE
4.347. The one-token organism depends on its PFC (band 3 carries the
context) and on its hippocampus. Open: the slow bands carry little at
15M tokens (fid 4-8 at .03-.05) — the PFC-depth / register question.
Next: scan5b to its step-12000 pass (do the store and slow-band deltas
GROW with tokens — the 500M sizing trend), then scan6 = readings fixes
+ council 4 (the PFC boost, measured against scan5b at matched steps).

## 24. scan5b step 12000 (24.6M tokens, 10:40 UTC): the hippocampus compounds

            6000    7500    12000
eval CE     4.475   4.347   4.163
in-ctx      .297    .318    .599
short       .297    .239    .384
store off   +.099   +.128   +.232      alpha .23 -> .29 -> .49
bands off   +1.01   +1.13   +.97
Recall doubled in 12M tokens; the read gain doubled. (The 500M hybrid
flash never armed its binder in 5B tokens.) scan5b runs to its step-
18000 lesion pass (~12:20 UTC) for the per-band trend at 37M tokens —
the input for the 500M band ladder — then scan6 = readings/sweep
fixes + expandable segments + council 4 (sha c87fa53), read against
this trace at matched steps.

## 25. The next two levers, built (11:20 UTC)

- scan6 = council 4 (PFC depth; SCAN_OPTS n_council 4) + the readings/
  sweep fixes + expandable segments (sha a7fd049 or later).
- scan7 = register {3: 4} (a7fd049: band 3 as four working-memory
  units, S20) — the PFC's fast capacity; or both if scan6 is a clear
  win. Hippocampus strength/KD after that.
Each is read against scan5b's trace at matched steps: eval CE 4.475 /
4.347 / 4.163 at 6000 / 7500 / 12000; in-ctx .297 / .318 / .599;
store off .099 / .128 / .232; the 7500 lesion pass (band 3 +1.79,
bands 4-8 +.014..+.025, store +.114).

## 26. scan5b step 18000 (36.9M tokens, 12:45 UTC): everything compounds, the ladder is ordered

dCE removed    @15.4M   @36.9M        in-ctx without it @36.9M (base .491)
hippocampus    +.114    +.315 (3x)    .184
band 3         +1.79    +1.53         .208
band 4         +.016    +.025         .461
band 5         +.014    +.035         .457
band 6         +.021    +.053         .396
band 7         +.024    +.061         .379
band 8         +.025    +.066         .372
all bands      +1.95    +1.89         .150
eval CE 4.347 -> 4.080; b4-gap recall .151 -> .292. The slow bands
doubled-tripled and are strictly ordered by timescale; load shifts
from band 3 to the slower bands and the store. scan5b killed at step
18500 (~$7.5, 6.3 h); scan6 launched = council 4 + readings/sweep
fixes + expandable segments (sha a7fd049).

## 27. scan6 (council 4) vs scan5b (council 2) at step 6000 (14:35 UTC)

eval CE 4.441 vs 4.475; in-ctx recall .417 vs .297 (+40%); short .292
vs .297; b4 .165 vs .10; store off .101 vs .099; bands off 1.02 vs
1.01; entropy 1.43 vs .84. 73.3M params (+9%), 2.46k tok/s (-7%),
flat (the readings fixes hold: detach/sweep 1 ms). A deeper PFC buys
recall more than CE. scan6 runs to 7500 (lesions) and 12000 (scan5b:
4.163 / .599 / .232); then scan7 = register {3:4} on the council-2
base, the two PFC levers measured independently before combining.

## 28. scan6 at 7500 (15:55 UTC): council 4 vs 2, lesion pass

eval CE 4.309 vs 4.347; in-ctx .38 vs .318 (lesion base .468 vs .389);
short .283 vs .239; b4 .178 vs .151; store off .129 vs .128; lesions
b3 +1.73 vs +1.79, bands 4-8 identical, all bands +1.90 vs +1.95,
store +.116 vs +.114. Depth buys recall (+20-40%) and a little CE;
the organ dependence is unchanged. User's framing (15:45): the PFC
should hold most of the parameters because it does attention's job —
agreed for INTEGRATION (depth; recurrent LMs put all params in the
recurrent path), with the correction that attention's RETRIEVAL is the
hippocampus's state (KD, free in params), and the binding constraint is
sequential time (PFC blocks run per token; the decoder is batched).
The depth curve 2 -> 4 -> 8 decides the 500M split (PFC 40-50%
expected); scan8 = council 8.

## 29. Phase 2 named (user, 16:00 UTC): rewards grounded in the PFC

The user's next step: ground reward in the PFC, let higher layers
learn to make lower layers show reward, then dopamine — complex
rewards form. Mapping: (1) value heads on the PFC state (bands +
council) that TRAIN it (today the prophet reads band states as a
spectator and presses settle on the cortex's readings); the decoder
stays value-blind. (2) TD across the ladder: band k's value predicts
the discounted sum the faster bands see — secondary reinforcers form
at each timescale. (3) Dopamine = reward prediction error gating
plasticity: RPE scales the hippocampus's per-pair write strength
(tok_u / beta) and the band gates at that moment. Caveat from scan2:
a value gradient into the PFC is the kind of force that stalled
training when the fidelity loss bent the council — small, measured
against CE, confined if needed. Reads: prophet AUC per band, recall
of pressed vs unpressed facts, the press-pay ledger. Order: after the
three 78M levers settle the 500M shape. This is the ratified two-
button design (graded +/- primary, band-built secondaries).

## 30. The reward slot (user, 16:20 UTC) — Phase 2's front door

Reward levels as a dedicated council slot at the lowest PFC level:
slots [token, REWARD (5-level embedding: none/+-1/+-2), hippocampus,
bands...]. Input only — observed, gated on (RPE = level - predicted
value scales tok_u/beta and the band gates at that token), read by the
value heads — NEVER a generation target (A64: a press never pays
itself; a model rewarded for emitting <+2> hallucinates approval) and
never seen by the decoder except through the PFC bundle. The pulse's
job on a reward token: open the gates, write the pre-reward context
hard, pull the slow bands toward it — credit assignment, not token
production. First Phase-2 iteration after the three shape levers.

## 31. scan6 verdict (17:10 UTC): council 4 = a small, consistent CE gain

            6000            7500            12000
CE    4.441 vs 4.475   4.309 vs 4.347   4.127 vs 4.163   (-.035 each)
inctx  .417 vs .297     .380 vs .318     .501 vs .599
store  .101 vs .099     .129 vs .128     .240 vs .232
Recall's edge is inside the probe's noise (scan5b's .599 was the
outlier: .516 at 18000); the lesion pass was identical. Depth 4: a
small CE lever (+9% params, -7% speed). scan6 killed at 12100 (~$3.3).
scan7 = register {3: 4} on the council-2 base (sha f9877cd).
