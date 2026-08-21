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
