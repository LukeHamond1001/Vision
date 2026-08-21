# V10.1 RERUN — 500M, bf16, 5.1B tokens, episodic cast on the real-dialogue spine

Ratified 2026-08-21 (user). Supersedes the v10 flash stopped at step
43,500 (pod ngq30ri1hc7jc4; volume 895khkglke keeps the 42,000
checkpoint with band states). Why stopped: the binder never armed —
the v10 cast was 24 PERSISTENT facts per life from an 8x8 vocabulary,
each asked tens of thousands of times; 192 facts total is memorized by
any model of this size (train-lane recall 0.96 vs unseen-life in-ctx
0.21 = chance). The debug gate G1 was `false` for the same reason and
was deferred to the 500M growth chart. A69-R2 (synthetic dense weaver)
had armed the in-ctx binder at 84% at d=128 — the FACULTY is learnable;
the v10 diet did not teach it.

## 0. Spec (fixed)
- HybridLM d=1280, 20L, T=2048, 16k untied vocab, bands 3-6 with
  BAND6_CLOCKS {3:1,4:8,5:64,6:512}, matrix store, logit keying,
  norm_mix, aux_trunk 0.2, gate_init -2.0. From scratch.
- Precision: fp32 master weights + AdamW; bf16 autocast in the trunk
  forward/backward; band states, store, losses, economy readings fp32
  (swamping law at 1M-token horizons). Certified by Step 3 before money.
- Tokens 5.1B, 8 lives (lanes == lives), one epoch of the UC spine +
  late stages x2 (Muennighoff) — the existing tokenized sources on the
  prep volume 2o9gtwzkhd. Stage fracs .08/.27/.38/.27. Judge unchanged
  (same JUDGE_VERSION, thresholds, density targets).
- lr 4e-5, warmup 2000, cosine on the global step; lam from the bf16
  smoke's holds/step (A60f product 0.25); ledger_cap 200k; sleep ladder
  infancy 0 / childhood 32 / adol+tail 16, ARM C, block 2, A76 H=1e-3,
  press_pay (T, T//8). Demoted kill policy stands (WARN lines; auto-
  stops: non-finite loss, tail audit >1%, dead instruments).
- Battery: HB_EVERY 6000 / 2500-chunk walk / lesions every 2nd beat
  through childhood; adolescence restart at 9000 / 5000 / 2; collapse
  probe with sampled distinct3 + entropy; growth-chart binder WARN.

## 1. Episodic cast (the fix) — builder change, $0
- EpisodicCast beside the persistent LifeCast roster: facts drawn from
  LARGE name/object vocabularies (hundreds each; colors stay the
  probe's closed set so chance stays 20%), planted continually
  (k per day), each asked n times at gaps from the stage menu
  (incl. 131k/1M in adolescence/tail = band food), then RETIRED so the
  weights cannot hoard them; eval lives draw their own novel facts.
- Same plant/ask/correction grammar, same probe events, same binding-
  margin distractors, same press classes (pos2/pos1/none) — the
  economy, heartbeat and serve room read it unchanged. Cast-ask density
  unchanged overall (~1 per 350 tokens); ~80% episodic / 20% roster.
- Law: episodic_rate=0 reproduces today's build bit-exactly
  (fingerprint test); retired facts are never asked; every ask's fact
  was planted earlier in the same life.

## 2. G1 gate at debug — $0, blocks everything
- Rebuild life_gate bio/ctrl (+eval) shards with the episodic cast
  (~12M tokens, 4 lives), run at d=128 for 12k steps (A69-R2's
  duration). PASS = in-ctx closed-set accuracy >= 0.40 (2x chance)
  on the UNSEEN eval lives, with short/b3/b4 bins and lesions
  reported. FAIL at d=128 -> escalate to the 78M pod pair ($15-40,
  the ledger's fallback) before any 500M money. No armed binder, no
  launch.
- Ride-along A/B on the same shards (same cost): decoupled RoPE
  (rotary on stream q/k, memory keys unrotated) + QK-norm vs the
  certified learned-absolute default. Ships ONLY on a win: CE parity
  or better, binder intact, store laws green. A wash keeps absolute —
  no uncertified axis enters the paid run.

## 3. bf16 certification — ~1 day, ~$5
- Code: autocast(bfloat16) around the trunk; init_state / band updates
  / store writes forced fp32; loss and economy readings fp32; no
  GradScaler (bf16). Default off = bit-exact fingerprint.
- Laws: dtype audit of model._st and store under autocast; fp32-vs-
  bf16 CE trajectory parity at d=128 (500 steps, tolerance pre-set);
  fid channels track within noise.
- Paid smoke on the H100 (same pod_v10.sh smoke stage, PRECISION env):
  300 steps fp32 vs 300 steps bf16 at lanes 8: tok/s (target >= 1.4x,
  i.e. >= 23k), peak memory, CE overlap, fid:3-6, FlashAttention kernel
  engaged. Cost bar for the flash: MIN_TOKS 22000.

## 4. Build + ship — mule pod, ~3 h, ~$1
- RTX 2000 Ada mule on the prep volume (EU-RO-1), dockerEntrypoint,
  SHA-pinned script: verify sources present -> build 5.1B / 8 lives
  with the episodic cast -> eval shards (2 lives, novel facts) ->
  smoke shard -> manifest (JUDGE_VERSION, cast stats, retirement
  law) -> runpodctl send; ship code rides the H100 env.
- Flash volume: move the old flash dir aside (keep v10.pt@42000 +
  v10_states for the record), receive the new corpus.

## 5. Launch — H100 US-NE-1, ~2.3 days, ~$180
- pod_v10.sh flow: receive -> bf16 smoke (lam, holds/step, cost bar)
  -> flash. Heartbeats, publisher, history persistence, warm restarts,
  end-of-life state bank as deployed at 6139f12 + this plan's env.
- Watch protocol unchanged: manual kill-fix-relaunch on the WARN
  lines; binder milestone at childhood end (40%); lesions through
  adolescence; fid:5 recovery under uninterrupted accumulation.

## Timeline
Aug 21: Step 1 + gate shards + Step 2 running overnight (CPU).
Aug 22: Step 3 (code + laws + debug parity) in parallel with the mule
build once G1 passes; $5 smoke; launch by evening UTC if G1 and the
smoke pass. Aug 25 (±): flash lands; serve-room day-one protocol.
Budget: $64 spent + ~$1 + ~$5 + ~$180 = ~$250.

## What v10 did NOT include — inventory and rerun decision
| Item | v10 status / evidence | Rerun |
|---|---|---|
| A71 band capacity (slowheavy widths, bands 5/6 at 1.5-2x) | OUT: lost A/B twice (v5.0; v10 gate -0.07% CE = noise, b5+ .29 vs .355) | OUT until lesions show bands load-bearing AND capacity-limited |
| A73 splice replay (SWS) | OUT: b5+ crashed to .129 (steals long-span replay) | OUT |
| A74 novelty-weighted replay | OUT: -0.08% = noise | OUT |
| A75 tied embeddings + larger vocab | OUT: +18% CE, binder dead | OUT (16k untied) |
| A77 dreaming (leashed REM) | OUT: hurt every bin at d=128 ("dreams need a bigger brain") | OUT; served life |
| Band 7 (8M horizon) | machinery certified (A70), ~600 ticks at 5.1B too thin | OUT |
| Reading diet (TinyStories/Cosmopedia/FineWeb-Edu/peS2o long-doc band food) | never built — corpus is dialogue-only | OUT for time (v10.2: the long-document band food is the piece most worth restoring) |
| 10B tokens | corpus truth 3.7B one-epoch spine -> 5.1B with x2 late stages | 5.1B |
| Lanes 12-16 | pinned 8 (memory-certain; lanes == lives) | 8 (bf16 headroom noted) |
| bf16 | out (uncertified, A49) | IN, gated by Step 3 |
| RoPE (decoupled) / QK-norm | out (mem-token geometry uncertified) | gated A/B in Step 2; ship only on a win |
| Sliding-window attention + bands (forced division of labor) | new (2026-08-21) | debug ablation if the Step-2 A/B harness has room; else v10.2 |
| Drive-ledger pruning | built (ledger_cap) | IN |
| End-of-life state bank / serve seeding | built (603dce6) | IN |
| Sleep-pool serialization across restarts | not built; pool is per-segment by design now | OUT |
| lm-eval-harness mini-row (HellaSwag/ARC-e/PIQA) | not built | optional, low priority |
| Density-matched press control | not built (v10.1 instrument) | OUT for the demo |
| Episodic (novel-fact) cast | absent — the flaw | IN (Step 1) |
| Adolescent pruning / map-like band states / embodiment | user decision: out / v11 | OUT |
| 78M raised life fork | parked | parked |

## Attention — why no RoPE, what frontier uses, what the math says
- v10 used learned absolute positions (`pos`, 2.6M params) because
  injected band/store reads have NO position: RoPE rotates q/k by
  position, so a memory token would need a fake one (far = decayed
  away, near = a lie about recency, zero = geometric nonsense).
  Absolute embeddings give memory tokens learned slots. Scale was
  never the reason; RoPE was named for v10.1, not silently absent.
- Frontier (to Jan 2026): RoPE (+YaRN/NTK scaling) for the stream;
  GQA/MQA for KV economy; MLA with decoupled RoPE dims (DeepSeek);
  interleaved sliding-window + global layers (Gemma 2/3, GPT-OSS);
  QK-norm; attention sinks; hybrids with linear-attention/SSM layers
  every 3-7 blocks (Qwen3-Next, Nemotron-H, Kimi Linear, MiniMax);
  FlashAttention-2/3, fp8 attention on Hopper.
- For this architecture attention's job is local precision + faithful
  memory reads, so: (1) bf16 -> FlashAttention is the largest lever
  now (fp32 SDPA has no flash kernel; at T=2048 x 8 lanes x 20 layers
  attention is a big share of the 1 s step); (2) DECOUPLED RoPE —
  rotary on stream q/k only, memory keys unrotated and matched by
  content (a summary of the past has no position); (3) QK-norm for
  stability at 20L; (4) sliding-window + bands as the purest division
  of labor (ablation); (5) GQA/MLA irrelevant at our KV sizes.
  (1) ships (certified); (2)+(3) only on a debug win; (4) later.
