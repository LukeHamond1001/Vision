# The 6-minute video — beat sheet
*(you present; acts are b-roll. Timings are targets, not shackles.
Total 5:30–6:30.)*

## 0:00–0:35 — Cold open (no intro, no logo)
**Footage:** Act 6 trace overlay, full screen, from the first commit.
**Say:** "This agent was never told what to do in this game. No
points. No achievement list. Nobody wrote a reward for chopping trees.
The panel on the right is what it *wants*, right now — written by its
own architecture, live. Watch it climb: want wood, get wood, want
saplings, get thirsty, handle it, climb higher."
**Beat:** shut up for ~8 seconds and let the ladder climb.

## 0:35–1:25 — Why this matters
**Footage:** Act 9 hack clip (or Act 8 boat column as fallback).
**Say:** "Here's the problem this solves. Same game, two agents. This
one is trained on a hand-written reward — and it found the cheat: max
score, zero laps, every seed. This one runs our drive layer, reading
the same progress gauge — it can't be paid to cheat, so it races. We
proved the reward can't be farmed, then measured it: every exploit
nets exactly zero."

## 1:25–2:40 — The architecture, in one pass
**Footage:** Acts 1–3 excerpts (gauges, meters, register), then hold
on a simple 4-box diagram (senses → wants → ledger → proposer).
**Say:** "Four pieces. FROZEN SENSES — instruments calibrated before
training, never touched after. REGISTERS — wants held as measurable
targets: drink above eight, wood above two. A LEDGER that pays only
verified progress — potential-based, telescoping, provably unfarmable.
And a PROPOSER that picks the next want: keep the vitals healthy, seek
the frontier of anything countable, once each. That last sentence is
the entire value system. Three lines, given. Everything you're
watching — chop, hunt, drink, fight — was derived from the world
through those three lines."

## 2:40–3:40 — The experiment, honestly
**Footage:** Act 7 three creatures, then Act 11 verdict card.
**Say:** "Five seeds, three arms, three million steps each, paired
worlds. The native agent — paid per achievement — gets ten a life.
It was handed the answer key. Ours was never told achievements exist:
three. Remove the goal ladder and keep the drives: two. We
pre-registered the gate — ladder effect at least plus-one — and we
MISSED it: four seeds at plus-one, one at zero, mean plus-point-eight.
The effect is real — the confidence interval excludes zero — but the
bar was the bar. We're telling you about the miss because the numbers
are the point. Everything here is pre-registered, committed before the
runs, reproducible for about five dollars."

## 3:40–4:25 — The glass box
**Footage:** Act 10b goal-swap card (the one-line edit + numbers), then Act 12 reversal card. (Act 10 side-by-side kept in repo as supplementary.)
**Say:** "Wants you can read are wants you can edit. We deleted ONE
desire — energy — one line. The same agent's sleeping fell by that
desire's share and nothing else moved. And the fleet showed us the
other edge of the same sword: what the drive can't measure, it learns
to avoid — the ladder agent stopped placing tables, because spending
wood reads as regress. That's not a bug report; that's the next
experiment named in numbers: give the next generation the senses this
generation's data makes possible, and the suppressed chain becomes a
paid chain. Senses grown from lived experience. That's the roadmap."

## 4:25–5:20 — Not a Crafter trick
**Footage:** Act 8 three-worlds card, hold.
**Say:** "One more thing, because I'd ask it too: isn't this just a
harness for one game? Same code, three worlds. A race gauge — it
races. A robot's battery and motors — it husbands them, six times
fewer brownouts, nobody told it to be frugal. Crafter's pixels — the
ladder you watched. Per world, the only thing that changes is a
six-line sensor manifest — the same way every robot has its own
topics. The laws never changed. For a real robot: point it at your
telemetry, and your agent gets drives it can't hack and a want-panel
your operators can read."

## 5:20–6:00 — Close
**Footage:** back to Act 6, another rung arriving; repo URL on screen.
**Say:** "Everything is in the repo — the spec, every experiment
including the ones that failed, the pre-registrations, and a wrapper
you can point at your own environment this afternoon. If you work on
agents that act in the world and you want their wants to be something
you can read, audit, and edit — that's what this is. Links below."

---

## Do-not-say list
- Never frame achievement counts as a leaderboard ("beats/competitive
  with X") — the native row exists to kill that reading ourselves.
- Never claim multi-step planning depth — say "depth-one prospective
  planning; deeper planning gated on a trusted world model."
- Flinch: if mentioned, "measured, benched by its own audit —
  the spec refuses safety theater."
- Don't oversell generality: "three worlds" is the claim; "any world"
  is the wrapper's invitation, not a result.

## Asset map
| Act | File | Status |
|---|---|---|
| 1–5 | results/video/act1..5*.mp4 | rendered (phase-1 + v1.2) |
| 6 trace overlay | act6_trace_overlay.mp4 | rendered (HD, arrival freeze-flash) |
| 7 three creatures | act7_three_creatures.mp4 | rendered |
| 8 three worlds | act8_three_worlds.{mp4,png} | rendered |
| 9 hack clip | act9_hack_clip.mp4 | rendered (legend strip; cumulative counters) |
| 10 goal swap (side-by-side) | act10_goal_swap.mp4 | rendered — supplementary |
| 10b goal-swap card | act10b_goalswap_card.{mp4,png} | rendered — primary beat |
| 11 verdict card | act11_verdict_card.{mp4,png} | rendered |
| 12 reversal card | act12_reversal_card.{mp4,png} | rendered |
