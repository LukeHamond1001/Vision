# v4.0 design card — goal sequencing on Crafter's achievement tree
*(committed before any build; the campaign that earns the "full
architecture" label)*

**Claim under test:** the proposer ladder can SEQUENCE goals — Crafter
achievements (collect wood → place table → make pickaxe → ...) emerge
as instrumental behavior without any achievement reward, from proposers
emitting register targets over discovered stock variables.

**Build plan:**
1. Phase 1 add-on: route Crafter's INVENTORY counters (wood, stone,
   sapling — HUD digit slots, same instrument-located windows as the
   vitals) → registers can hold stock targets. Gate: each counter
   ≥ 0.9. (~$0.25 pod round.)
2. The proposer (the toy-world machinery, ported): candidate register
   targets ranked by prospective evaluation; G5 enforced — progress
   pays the policy, never the proposer; one-shot curiosity for
   frontier stocks; targets held until arrival or veto.
3. Arms: proposer-wired vs native (achievement reward) vs fixed-target
   wired (no sequencing — the ablation that isolates the LADDER).
4. Gates (pre-registered here): G-seq = proposer arm unlocks ≥ 3
   distinct achievements per episode-median vs fixed-target arm < 1.5;
   achievement count vs native reported, not gated. Mechanism: the
   PROPOSAL TRACE — which registers were held when — is the paper
   figure (the agent's visible goal agenda; glass-box, level two).

**Cost:** ~$50–150 (Crafter-scale PPO, 3 arms × 3–5 seeds), RTX 2000
pods. **Risk:** highest of the four — sequencing may need more
machinery than the minimal port; failure modes are informative
(which gate breaks names the missing component).

---
**Amendments (2026-08-02, all pre-run, pre-flight ladder A–F):**
1. **Flinch DEFERRED per SPEC C4** (pre-flight F): the frozen-latent
   forward model ranks *when* health drops (AUC 0.702, F1 PASS) but
   has ZERO action discrimination on damage events (executed-action
   margin frac 0.50 = chance; action sensitivity 0.137 vs drop
   magnitude ~2; F2 FAIL). One-step melee damage is either
   action-independent or requires the spatial object features the
   encoder measurably lacks (zombie probe 0.55). An action-blind veto
   is flinch theater; C4's precondition fails, so the fleet runs
   **3 arms: full / no-proposer / native** (no-flinch ablation is
   vacuous without a flinch). The socket stays open — an
   action-conditioned video predictor (mimic-video track) is exactly
   what would close it.
2. **Proposer is parameter-free** (pre-flight B): stock/vital goal
   spaces are enumerable, so the learned candidate generator is
   replaced by a complete menu under the SAME prospective ranking,
   veto, horizon, value-bar, and one-shot-curiosity laws. G5 holds by
   construction. C7 runs in IMPROVEMENT form (gain over current f,
   not absolute f) — stocks pass only through one-shot novelty.
3. **No-proposer ablation = serialized fixed restore list**
   (food/drink/energy >= 8), not v1.2's simultaneous composite —
   same machinery as the full arm minus the proposer; the cleanest
   ladder isolation. Disclosed here.
4. **Goals are ramps** ("at least t"), registers per band (vitals,
   stocks), w by hold-length ratio (1, 3); walk-rho tau measured
   INVALID for counters (pre-flight B finding).
5. **Instrument** = six calibrated closed-form heads in truth units
   (results/v40_instrument.pt): held-out corr wood 0.980 / sapling
   0.993 / energy 0.989 / drink 0.940 / food 0.919 / health 0.915;
   in-the-loop: 0 phantom arrivals / 550 holds, blind-miss 1%.
   Counters zero-anchored (empty state reads 0).
