# scripts/ — live entry points

| script | role |
|---|---|
| `pod_v10.sh` | the flash pod payload (H100): receive corpus → bf16/fp32 smoke (lam, cost bar) → flash with heartbeats, publisher, warm restarts, end-of-life state bank. Booted by `dockerEntrypoint` from a SHA-pinned raw URL; env knobs documented in the header |
| `pod_v10_rebuild.sh` | the mule payload (RTX 2000 Ada): rebuild the 5.1B episodic corpus on the prep volume, run the 78M mini-flash gate on its GPU, ship everything to the flash volume with `runpodctl send` |
| `fetch_v10_corpus.sh` | the named sources of the corpus spine (UltraChat → SmolTalk2 → Smol-Magpie-Ultra), fetch tiers |
| `v10_driver.py` | the staged-life driver: segments, sleep ladder, battery cadence, PLAN/trace rows |
| `heartbeat_v10.py` | the in-flight battery: CE + recall by gap bin on unseen lives, collapse, cast/incumbent, tail audit, prophet, lesions (per band, all bands, store, base) |
| `life_gate.py` | the $0 debug gates (G1–G4, organ arms: rope / modern / bandlr / conveyor) on prepared mini-shards |
| `serve_v94s.py`, `live_room.py`, `room_cli.py`, `parent_session.py` | the serve room — one continuous life with presses and nightly sleep (built on the 78M raised life; the V10.1 being is served through the same room) |
| `demo_three_acts.py` | the A66 operant demo (baseline → parenting → evidence at three timescales); the template the centerpiece demos extend |

`archive/` holds every earlier pod payload (v5.2 → v9.4, the r1–r9
trainer rungs, prep/prune/export ops) and the 78M-era gate drivers
(A62–A69, A76). They are the scripts the ledger cites; paths in ledger
entries before 2026-08-21 are pre-archive (`scripts/x` → `scripts/archive/x`).
Archived Python scripts still resolve the repo root themselves; the
three that don't (`ab_sleep_debug.py`, `parenting_debug.py`,
`score_a63.py`) run with `PYTHONPATH=.` from the root.
