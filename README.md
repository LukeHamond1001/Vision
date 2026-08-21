# Vision — a language being that wakes up having lived a life

A 500M-parameter language model whose pretraining is not a corpus but a
**lifetime**: eight staged lives of real dialogue, graded by a frozen judge
into the counterparty's graded button presses, with facts planted and
asked across gaps up to a million tokens, corrections, and nightly sleep
that replays only what the economy paid. The organism has organs a
transformer does not: **timescale bands** (a slow thread at 2k / 16k /
131k / 1M-token horizons) and a **contextual memory** (per-band
associative stores of what was actually said). The point of the program
is to show what each organ contributes by removing it from the same being
on the same day of the same life.

**The centerpiece** — [docs/CENTERPIECE.md](docs/CENTERPIECE.md): the V10.1
flash (bf16, 5.1B tokens, ~2.3 days on one H100) and its two demonstrations,
pre-registered before the first token:

| | switch | what the being becomes (pre-registered expectation) |
|---|---|---|
| **Demo 1 — bands removed** | `model.lesioned = {3,4,5,6}` | in the moment: fluent about what is in front of it, recall beyond its 2048-token chunk falls to chance, press anticipation gone |
| **Demo 2 — contextual memory removed, bands on** | `model.store_read_off = True` | juggles but fails: keeps the thread of the day and the counterparty, reaches for the wrong name |

Both switches already exist in the forward pass; with both off the model
is bit-exact to the certified one. Expectations are directions the organ
program predicts, measured as paired speech-gated contrasts with
pre-registered bars — a miss is published as a miss.

**Status (2026-08-21):** the v10 flash was stopped at step 43,500 because
its cast of 24 persistent facts per life was memorized and the binder
never armed (ledger: "THE STOP"). V10.1 fixes the diet (an episodic cast
of novel facts, retired after use) and certifies bf16; its go/no-go is a
78M mini-flash of the real run, in flight now. Live rows stream to the
`results-v10` branch; the ledger's last entry is always the current state.

## How it is built — the short version

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) is the long version.

- **Cortex**: 20 blocks, d=1280, 8 heads, T=2048, learned absolute
  positions, 16k untied vocab — 75% of 525M parameters.
- **Bands 3–6**: gated delta-write cells ticking every 1 / 8 / 64 / 512
  chunks, each projected to one memory token the cortex attends; a
  fidelity head per band (`fid:k`) and a press-prophet head per band.
- **Stores**: one logit-keyed associative matrix per band, written every
  chunk (key = the preceding context, value = the token's identity),
  read at mid-depth through a gate.
- **Drive** (`iga/lm_drive.py`): press tokens `<+1> <+2> <-1> <-2>` in the
  stream; holds, mints, voids and vetoes by exact accounting; `lam` set
  from the measured press density so `lam × holds ≈ 0.25`.
- **Sleep** (`iga/lm_sleep.py`): replay of paid spans, ARM C contrastive
  correction pairs, A76 homeostasis; sleepless infancy, then 1:16.
- **The life** (`iga/lm_data_life.py`): UltraChat → SmolTalk2 → Smol-Magpie
  in order, staged infancy / childhood / adolescence / tail, a frozen
  judge grading every exchange, the episodic cast, corrections 3–8%.
- **The battery** (`scripts/heartbeat_v10.py`), every 6000 steps on unseen
  lives: CE, recall by gap bin (in-ctx … b6), collapse, incumbent, tail
  audit, prophet AUC, and the **lesions** — each band, all bands, the
  stores — the two demos' growth curve, read in flight.

## Evidence and honesty

- [docs/LANGUAGE_LADDER.md](docs/LANGUAGE_LADDER.md) — the ledger: every
  run, every gate, every kill and amendment, written before the next
  launch. v5.0 (first band LM) → v9.4 (certified 78M core, the raised
  life) → v10 (the flash, stopped) → V10.1. Read the last entry first.
- [docs/V10_1_RERUN_PLAN.md](docs/V10_1_RERUN_PLAN.md) — the ratified run,
  gates, and the inventory of everything gated OUT with its evidence
  (A71 band widths, A73 splice replay, A74 novelty replay, A75 tied
  vocab, A77 dreaming — each lost its gate and stayed out).
- [docs/V10_FLASH.md](docs/V10_FLASH.md) — the design spec and the brain
  fidelity audit (what corresponds, what is named as divergent).
- [docs/DEMO_PROTOCOL.md](docs/DEMO_PROTOCOL.md) — speech-gated,
  within-run, n ≥ 20: the rules every headline obeys, registered before
  launch. [docs/SKEPTIC_REBUTTAL.md](docs/SKEPTIC_REBUTTAL.md) — the
  strongest baseline ("nightly SFT on pressed spans would do the same"),
  answered in advance.
- `results/evidence/` — gate results; `results/v10_flash/` — the stopped
  run's battery rows; [results/README.md](results/README.md) maps the rest.

**House rules:** gates are registered before runs and amended only with a
ledgered reason; failed gates are reported as failed; a candidate organ
enters a paid run only after it wins its own gate; measurement is local
and free, pods are for GPU training only; no single-run attribution —
every headline is a within-run contrast.

## Run it

```bash
python -m pytest tests -q          # 219 laws: sleep harvest, seam, bf16, rope, episodic cast, drive, ladder
```

```bash
GATE_EPI=1 python scripts/life_gate.py 12000 128 bio   # the $0 debug gates on prepared mini-shards: [steps] [d] [arms: bio,ctrl,rope,modern,bandlr,conveyor|all]
```

The pod payloads are `scripts/pod_v10.sh` (flash) and
`scripts/pod_v10_rebuild.sh` (corpus + mini-flash gate); the driver is
`scripts/v10_driver.py`. Sources and their order: `scripts/fetch_v10_corpus.sh`.
The serve room — one continuous life with the buttons —
is `scripts/serve_v94s.py` / `scripts/room_cli.py`.

## Layout

```
iga/lm_hybrid.py       the organism: trunk, bands, stores, lesion switches
iga/lm_transformer.py  blocks (GELU/SwiGLU, absolute/rotary — gated flags)
iga/lm_bands.py        band cells and clocks
iga/lm_drive.py        the press economy (holds, mints, voids, vetoes, horizons)
iga/lm_sleep.py        the sleeper (replay, ARM C pairs, homeostasis, seam law)
iga/lm_press.py        press-prophet heads
iga/lm_judge.py        the frozen judge
iga/lm_data_life.py    the life builder (stages, cast, corrections, rituals)
iga/lm_data_*.py       sources and tokenization
iga/lm_train.py        the training loop (segments, bf16, checkpoints, warm restarts)
iga/lm_serve.py        the serve room
iga/lm_eval.py, lm_gen.py, lm_calibrate.py, lm_conveyor.py, lm_ab.py, lm_dream.py
scripts/               live entry points (README inside); archive/ = every earlier rung
tests/                 the laws
docs/                  ledger, specs, protocols; drive-layer/ = the origin program
results/               evidence, battery rows, gate logs (README inside)
```

## Where this came from

The language program grew out of a drive-layer program for agents — a
parameter-free, provably non-farmable progress reward with readable goals,
measured on Crafter, a simulated robot and a reward-gaming track, four of
six pre-registered gates missed and reported. That program is finished and
audited; its README, spec, audit and paper are preserved verbatim under
[docs/drive-layer/](docs/drive-layer/README.md), and its modules (`iga/agent.py`,
`iga/goal_machine.py`, `iga/experiments_v*.py`, …) and `results/` artifacts
are untouched so its reproduce matrix still runs. The timescale bands of
the being are that program's registers, grown into a language organism.

Apache-2.0. Data, launch playbooks and keys are not in the repo.
