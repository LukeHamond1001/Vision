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
