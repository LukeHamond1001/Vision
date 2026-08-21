# results/ — committed artifacts

Two programs share this directory. Paths are frozen: the drive-layer
reproduce matrix and the ledger cite them by name.

## The language being (current program)

| path | what |
|---|---|
| `evidence/` | the A-series gate results (A62–A76), v9 autopsies, v10 gate JSONs (`v10_gates*.json`, `v10_g1_*.json`, `v10_a77.json`), `a67_parenting/` (the raised 78M life's sessions), `shakedown/` (local heartbeat shakedown rows) |
| `v10_flash/hb_v10.jsonl` | the v10 flash's battery rows (steps 6000–42000; the run stopped at 43,500 — see the ledger, "THE STOP") |
| `gate_v10/` | local gate logs of the v10 (roster-cast) era: G1 at 8k/12k, the G1–G4 run, the A77 dream gate |
| `gate_epi/` | local gate logs of the episodic-cast era (bio / rope / modern arms at d=128) |
| `autopsy_v*.py`, `v5x/v6x/v7x/v8x_autopsy.*`, `v53_finebins.txt` | autopsy instruments and readouts of the 78M rungs (v5.3 → v8.0). Autopsies score against `mix_r1_eval` (branch `data-r1eval`), never `mix_v9` |
| `lm_constants*.json` | calibration constants of the language drive (`iga/lm_calibrate.py`) |

Battery rows of the live run stream to the `results-v10` branch
(`hb_v10.jsonl`, `HEARTBEAT.log`, `v10_driver.jsonl`); the mini-flash
gate publishes `mini_hb.jsonl` there. Raw pod ledgers of earlier rungs
are the `results-*` branches.

**Naming note.** `v10_policy_*.pt`, `v10_summary.json`, `v10_behavior.jsonl`
are the DRIVE-LAYER v1.0 Crafter artifacts (2026-07), not the language
program's V10 flash (2026-08). The language program's v10 files are the
ones under `v10_flash/`, `gate_v10/`, `gate_epi/` and `evidence/v10_*`.

## The drive-layer program (finished, audited)

`INTERPRETATION.md` (the round-by-round log), `battery.{json,md}`,
`e2a_*/e3b_*/e5*` (toy-world battery), `v03`–`v40` JSONs/`.pt` policies
(the campaign cards), `video/` (the demo reel). Reproduce and verify:
[docs/drive-layer/README.md](../docs/drive-layer/README.md) and
[docs/drive-layer/AUDIT.md](../docs/drive-layer/AUDIT.md).
