# Outreach kit
*(everything ready to fire the day the video is uploaded. Nothing here
sends itself — every email goes from Luke, personally, one at a time.)*

## The play (order of operations, one morning)
1. Video uploaded (unlisted first; check it plays).
2. `iga-scale` flips PUBLIC. Final check: README renders, clips play,
   `pip`/clone + tiny smokes work on a fresh machine.
3. Post the X thread (clip files attached natively — act6 first frame
   as the hook image, act9 as the payoff clip). Pin it.
4. Show HN, same morning (HN + X cross-pollinate in the first hours).
5. mimic GitHub issue goes up (it reads better BEFORE any traffic
   arrives, not after — genuine, not promotional).
6. Emails: 3–4 per day over 3 days, personalized top lines, NOT a
   blast. Stagger so replies don't collide.
7. Reply fast for 72 hours. That's the whole campaign.

## Who (10 targets, why each one would care)

**Tier 0 — contribute-before-contact**
| Who | Why them | The hook |
|---|---|---|
| mimic-video maintainers (GitHub issue, not email) | their video predictor is exactly the action-conditioned world model our flinch/planning socket requires; Apache-2.0, open checkpoints | "a drive layer for your latents — adapter offer" |

**Tier 1 — goal generation / open-endedness / intrinsic motivation**
| Who | Why them | The hook |
|---|---|---|
| Pierre-Yves Oudeyer (Inria, Flowers) | autotelic agents & IMGEP are the learned-goal-generation literature; ours is the parameter-free counterpoint | "an autotelic agent with zero learned goal machinery — and a visible agenda" |
| Jeff Clune (UBC) | AI-GAs, open-endedness; cares about goal invention and honest evaluation | "sequencing emerges from two fixed appetites; pre-registered, misses reported" |
| Tim Rocktäschel (UCL) | open-endedness agenda; Crafter-adjacent evals | "the frontier-once rule as a one-line curiosity that can't be farmed" |
| Joel Lehman | novelty search lineage; wrote the book on deceptive objectives | "novelty that dies on arrival: one-shot curiosity with a non-farmability proof" |
| Minqi Jiang | open-endedness / UED; sharp on what emerges vs what's smuggled | "no crafting tree anywhere in the system — the trace proves it" |

**Tier 1 — specification gaming / safety**
| Who | Why them | The hook |
|---|---|---|
| Victoria Krakovna (DeepMind) | maintains THE specification-gaming examples list | "a new entry for the list — and the same gauge un-gamed beside it" (act9) |
| Rohin Shah (DeepMind) | value specification, honest-evaluation taste | "values as six frozen legible lines; the agent can't bend the judge" |
| David Krueger (Mila) | reward hacking, incentives | "the anti-wireheading construction, measured: every exploit nets zero" |

**Tier 1 — world models / benchmarks**
| Who | Why them | The hook |
|---|---|---|
| Danijar Hafner | AUTHOR OF CRAFTER (and Dreamer) — will instantly see what the native arm means and what the 3-vs-2 isolates | "we ran your benchmark with the achievement reward removed — here's what an agent wants when nobody tells it" |
| Sherry Yang | video world models as substrate | "the drive layer that would sit on a video model — the audit that decides if a latent is sense-worthy" |

## The email (3 paragraphs, ~120 words, one clip, no ask)

Rules: subject = specific claim, not "my project". First line names
THEIR work honestly. One result, one honest-verdict sentence
(mandatory — it is the credibility signature). Clip + repo. Soft
close. No job ask, no "pick your brain", no attachments except links.

### Template
> **Subj:** [specific claim tuned to them]
>
> [1 line: their work → why this lands on their desk.]
>
> [3 lines: the result. One number. The honest-verdict sentence:
> "The pre-registered gate missed on one seed of five (+0.8 vs +1.0,
> CI excludes zero) — reported as failed; the mechanism table is
> unambiguous."]
>
> 90-second clip: [link]. Repo with pre-registrations, all verdicts
> incl. the failures, and $5 reproduce commands: [link]. Happy to
> answer anything. — Luke

### Example — Hafner (the special case)
> **Subj:** Crafter with the achievement reward removed — what the agent wants instead
>
> You built Crafter, so you'll see in one frame what this is: we ran
> it with NO achievement reward and no task reward at all — the agent
> is driven only by a frozen, parameter-free drive layer (maintain
> vitals; seek the frontier of measured stocks, once each), with its
> goal agenda rendered live as text.
>
> Result, 5 seeds × 3M, paired worlds: drives+ladder reaches 3.0
> distinct achievements/episode vs 2.0 for drives-without-ladder;
> native (paid per achievement) gets 10.0, as it should — that row is
> in the table on purpose. Our pre-registered paired gate (≥ +1.0)
> missed on one seed (+0.8, CI95 [+0.4, +1.2]) and is reported as
> failed. The reward layer is provably unfarmable (telescoping claims;
> audited to exactness over 1,200 holds).
>
> 90-second clip of the live goal agenda: [act6]. Repo: [link].
> Thought you'd want to see your world through this lens. — Luke

### Example — Krakovna (spec-gaming)
> **Subj:** A specification-gaming example — with the un-gamed twin beside it
>
> Your specification-gaming list is the reason this experiment exists.
> We built a canonical mis-specified reward (pay per checkpoint entry)
> and trained on it: the agent finds the oscillation exploit, score
> 252, zero laps, every seed — a clean new entry for the list.
>
> The other half is the point: an agent reading the SAME progress
> gauge through a potential-based register (telescoping ⇒ exploits net
> exactly zero, proven then audited) races 6–7 laps. Same pixels, same
> learner. On our larger study the pre-registered gate missed on one
> of five seeds and is reported as failed — house rule.
>
> 45-second clip: [act9]. Repo with proofs/audits: [link]. — Luke

### Example — Oudeyer (autotelic)
> **Subj:** An autotelic agent with no learned goal generator — agenda visible as text
>
> Your IMGEP line asks how agents should invent their own goals; this
> is the deliberately opposite corner of that space: goal generation
> with ZERO learned machinery — an enumerable menu under two fixed
> appetites (maintain vitals; frontier-once on any measured stock),
> ranked prospectively, held in registers, paid by an unfarmable
> ledger.
>
> On Crafter (no task reward) it sequences: wood→sapling→wood₂… with
> maintenance interleaved, and the whole agenda renders as live text —
> the trace, not an interpretation. Effect vs the no-proposer ablation
> is CI-clean (+0.8 [0.4, 1.2]); our stricter pre-registered bar
> missed on one seed of five and is reported as failed.
>
> 90-second clip: [act6]. Repo: [link]. Curious whether this reads to
> you as a useful floor for the learned versions. — Luke

## mimic — the GitHub issue (goes up before any traffic)
> **Title:** Adapter offer: a frozen drive layer over mimic latents
> (registers + non-farmable progress + goal traces)
>
> We've built a parameter-free drive layer (goals held as measurable
> targets, potential-based progress that provably can't be
> reward-hacked, one-shot frontier curiosity) and validated it across
> three environments — write-up and demos: [repo]. Two of its
> components are gated on an action-conditioned world model, which is
> exactly what mimic provides.
>
> Concrete offer: we'd like to contribute a small adapter + notebook
> that (1) probes which quantities are linearly readable from your
> latents (our closed-form instrument audit), and (2) runs our drive
> layer over them on LIBERO episodes, producing the live goal-trace
> panel. Zero changes to your code — pure consumer. Would a PR along
> those lines be welcome, and is there a preferred entry point?

## X thread (9 posts; clips attached natively)
1. This agent was never told what to do. No reward function for the
   task. The right panel is what it WANTS — live, written by its own
   architecture. [act6 clip]
2. The drive layer is frozen before training and parameter-free:
   senses → wants → an unfarmable ledger → a proposer. The whole
   value system is three lines: keep vitals healthy; seek the
   frontier of anything measurable, once each.
3. "Unfarmable" is a theorem, not a vibe: progress pay telescopes, so
   every loop/oscillation/exploit nets exactly zero. Then we audited
   it: 1,200+ holds, exact to float precision, zero phantom payments.
4. Here's what that buys. Hand-written reward, canonical
   mis-specification: the agent finds the cheat — score 252, zero
   laps. Register reading the SAME gauge: races. [act9 clip]
5. On Crafter (5 seeds × 3M steps, paired worlds): drives+ladder 3.0
   achievements/ep, drives-only 2.0, native-with-answer-key 10.0.
   That last row is there on purpose — this isn't a leaderboard claim.
6. Our pre-registered gate (≥+1.0 paired) MISSED on one seed of five:
   +0.8, CI [+0.4, +1.2]. Reported as failed. The repo's history has
   every gate, every amendment, every miss. [act11 card]
7. Wants you can read are wants you can EDIT: we deleted one desire
   (one line) — sleeping fell by that desire's share, drink untouched.
   The edit dissected the behavior into its motives. [act10b card]
8. And the fleet's best finding was a reversal: what the drive can't
   measure, it learns to AVOID (772 tables vs 20). The missing senses
   are named, in numbers, by the failure. That's the next generation.
   [act12 card]
9. One drive layer, three worlds, zero law changes. Spec, proofs,
   pre-registrations, $5 reproduces: [repo]. Full 6-min video: [link].
   [act8 card]

## Show HN
**Title:** Show HN: An RL agent with no task reward — its goals render
as live text, and its reward provably can't be hacked
**Body:** ~6 lines: what it is, the act6 clip link, the honest verdict
sentence, three-worlds line, repo + $5 reproduce, "happy to answer
everything, including why the pre-registered gate failed."

## What success looks like (so we don't misread the week)
- 2–3 substantive researcher replies = campaign worked.
- One mimic maintainer response of any warmth = best single outcome.
- Stars/views are noise either way; conversations are signal.
- Every artifact is durable — this kit re-fires when the robot-substrate
  result lands on top of it.
