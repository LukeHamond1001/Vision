"""v5.0 trainer — conveyor -> bands -> drive, one loop, checkpointed.

Loss = next-token CE
     + FID_W * mean(1 - band next-window fidelity)   (predictor training)
     + drive pays (in-graph, negative: arriving is rewarded)

v0 scope notes (honest): episode recall holds settle at scene end
rather than <eot_model> (the scope machinery exists; turn settlement
is a later wiring). Band-fidelity maintain-holds pay ledger-only —
their gradient pressure already lives in the fidelity loss, and
double-paying the same quantity would be minting.

Usage:  python -m iga.lm_train smoke
        python -m iga.lm_train run --d 256 --steps 20000 --chunk 512
"""

import argparse
import time
import torch

from .lm_conveyor import Vocab, Conveyor, splits
from .lm_bands import BandLM
from .lm_drive import Drive

FID_W = 0.1


def process_chunk(model, drive, conveyor, T, device, opt=None):
    x, y, events = conveyor.chunk(T)
    x, y = x.to(device), y.to(device)
    logits, model._st, ticks = model(x, model._st, None)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, model.vocab_size), y.reshape(-1))
    losses = [ce]
    fid_terms = []
    for k in range(1, len(ticks)):
        for _, fid in ticks[k]:
            if fid.requires_grad:
                fid_terms.append((1 - fid).mean())
            drive.tick_fid(k, fid)
    if fid_terms:
        losses.append(FID_W * torch.stack(fid_terms).mean())
    logp = torch.log_softmax(logits, dim=-1)
    for lane, evs in enumerate(events):
        for p, kind, d in sorted(evs, key=lambda e: e[0]):
            if kind == "probe" and p > 0:
                prob = logp[lane, p - 1, d["answer"]].exp()
                drive.probe(lane, prob, d["gap"])
            elif kind == "earned":
                drive.earned(lane, d["ok"])
    drive.step_t += T
    drive.sweep(losses)
    loss = torch.stack([l if l.dim() == 0 else l.mean() for l in losses]).sum()
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model._st = model.detach_state(model._st)
    # reading tensors retained across the boundary leave the graph here:
    # pay gradients flow through settlement-chunk readings only
    drive.detach_readings()
    return float(ce.detach()), float(loss.detach())


def train(d=64, lanes=4, T=256, steps=40, seed=0, device="cpu",
          ckpt=None, log_every=10, data=None):
    drive = Drive(n_lanes=lanes, seed=seed)
    if data:  # prepared real-data shard (A8): UltraChat conveyor
        from .lm_data_ultrachat import UltraConveyor, load_tokenizer
        import os
        conveyor = UltraConveyor(data, n_lanes=lanes)
        tok = load_tokenizer(os.path.join(data, "tokenizer.json"))
        vocab, vocab_size = tok, tok.get_vocab_size()
    else:
        vocab = Vocab()
        vocab_size = len(vocab)
        conveyor = Conveyor(vocab, n_lanes=lanes, seed=splits(seed)["train"],
                            bias_fn=drive.bin_weights)
    model = BandLM(vocab_size, d=d).to(device)
    model._st = model.init_state(lanes, device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    t0 = time.time()
    ce_first = None
    for step in range(1, steps + 1):
        ce, loss = process_chunk(model, drive, conveyor, T, device, opt)
        ce_first = ce_first or ce
        if step % log_every == 0 or step == 1:
            tok_s = lanes * T * step / (time.time() - t0)
            print(f"step {step:5d}  ce {ce:.3f}  loss {loss:.3f}  "
                  f"holds {len(drive.ledger):4d}  {tok_s:,.0f} tok/s")
        if ckpt and step % 500 == 0:
            torch.save({"model": model.state_dict(),
                        "opt": opt.state_dict(), "step": step}, ckpt)
    audit = drive.audit()
    print("audit:", audit)
    print("panel:\n" + drive.panel())
    return model, drive, vocab, ce_first, ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "run"])
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--ckpt", default="lm_ladder.pt")
    ap.add_argument("--data", default=None,
                    help="prepared shard dir (lm_data_ultrachat prepare)")
    a = ap.parse_args()
    if a.mode == "smoke":
        model, drive, vocab, ce0, ce1 = train(d=64, lanes=4, T=256, steps=40,
                                              device="cpu", data=a.data)
        assert ce1 < ce0, "smoke: CE did not fall"
        assert drive.audit()["telescoping_exact"], "smoke: ledger not exact"
        print(f"SMOKE PASS  ce {ce0:.3f} -> {ce1:.3f}")
    else:
        train(d=a.d, lanes=a.lanes, T=a.chunk, steps=a.steps, seed=a.seed,
              device=a.device, ckpt=a.ckpt, log_every=50, data=a.data)


if __name__ == "__main__":
    main()
