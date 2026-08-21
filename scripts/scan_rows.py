#!/usr/bin/env python3
"""scan_rows.py — read an iteration's heartbeat rows (scan_hb_<ITER>.jsonl
on results-v10, or a local file) against the plan's R1-R4 (docs/
ONE_TOKEN_PLAN.md). Usage: python scripts/scan_rows.py scan1 [file]"""
import json, subprocess, sys

iter_ = sys.argv[1] if len(sys.argv) > 1 else "scan1"
if len(sys.argv) > 2:
    text = open(sys.argv[2]).read()
else:
    text = subprocess.run(
        ["git", "show", f"origin/results-v10:scan_hb_{iter_}.jsonl"],
        capture_output=True, text=True).stdout
rows = [json.loads(l) for l in text.splitlines() if l.strip()]
if not rows:
    print("no rows yet"); sys.exit(0)
for r in rows:
    probes = {p["probe"]: p for p in r.get("rows", [])}
    ce = probes.get("ce_recall", {}).get("ce")
    rec = probes.get("ce_recall", {}).get("recall", {})
    rec_s = " ".join(f"{k}={v['acc']}" for k, v in rec.items() if v.get("acc") is not None)
    les = {k[7:]: v.get("ce_delta") for k, v in probes.items() if k.startswith("lesion_")}
    les_s = " ".join(f"{k}:{v:+.3f}" for k, v in les.items() if v is not None)
    col = probes.get("collapse", {})
    sh = probes.get("store_health", {})
    bd = probes.get("boundary", {})
    print(f"step {r.get('step')} tok {r.get('tokens'):,} verdict {r.get('verdict')}  "
          f"R1 ce={ce}  recall[{rec_s}]")
    if les_s:
        print(f"   R2 lesion dCE  {les_s}")
    print(f"   R4 collapse d3={col.get('distinct3')} sampled={col.get('distinct3_sampled')} H={col.get('entropy')} | "
          f"alpha={sh.get('alpha')} tok_u={sh.get('tok_u', {}).get('mean')} | "
          f"boundary deficit0_16={bd.get('deficit_0_16')} thread_off0_64={(bd.get('deltas_0_64') or {}).get('thread_off')}")
    if r.get("warnings"):
        print("   WARN", r["warnings"])
