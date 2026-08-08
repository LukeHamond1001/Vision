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
                if d.get("distractors"):
                    # binding-margin channel (v5.1): pay only for
                    # beating the other colors in play — prior- and
                    # recency-tracking are worth exactly zero
                    pd = torch.stack([logp[lane, p - 1, i].exp()
                                      for i in d["distractors"]]).max()
                    reading = torch.clamp(prob - pd, min=0.0)
                else:
                    reading = prob
                drive.probe(lane, reading, d["gap"])
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
          ckpt=None, log_every=10, data=None, talk="dense", widths=None,
          compile_model=False, constants=None, arch="bands"):
    if "cuda" in str(device):
        torch.set_float32_matmul_precision("high")  # TF32 (A12)
    torch.manual_seed(seed)  # reproducible init (A14)
    drive = Drive(n_lanes=lanes, seed=seed, constants=constants,
                  imagination_absent=(arch == "transformer"))
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
    if arch == "transformer":
        from .lm_transformer import TransformerLM
        model = TransformerLM(vocab_size, d=d, max_T=T).to(device)
    elif arch == "hybrid":
        from .lm_hybrid import HybridLM
        model = HybridLM(vocab_size, d=d, max_T=T).to(device)
        drive.bin_band = {0: 3, 1: 3, 2: 4, 3: 5}  # carry bands (A19)
    else:
        model = BandLM(vocab_size, d=d, talk=talk,
                       widths=widths).to(device)
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile unavailable ({e}); running eager")
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
                  f"holds {len(drive.ledger):4d}  {tok_s:,.0f} tok/s",
                  flush=True)
        if ckpt and step % 500 == 0:
            torch.save({"model": model.state_dict(),
                        "opt": opt.state_dict(), "step": step,
                        "drive": {"ema": drive.ema,
                                  "records": drive.records,
                                  "minted": sorted(drive.minted),
                                  "holds_settled": len(drive.ledger),
                                  "vetoes": drive.vetoes}}, ckpt)
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
    ap.add_argument("--talk", default="tick")
    ap.add_argument("--constants", default=None,
                    help="calibrated constants json (lm_calibrate)")
    ap.add_argument("--arch", default="bands",
                    choices=["bands", "transformer", "hybrid"])
    a = ap.parse_args()
    if a.mode == "smoke":
        model, drive, vocab, ce0, ce1 = train(d=64, lanes=4, T=256, steps=40,
                                              device="cpu", data=a.data)
        assert ce1 < ce0, "smoke: CE did not fall"
        assert drive.audit()["telescoping_exact"], "smoke: ledger not exact"
        print(f"SMOKE PASS  ce {ce0:.3f} -> {ce1:.3f}")
    else:
        train(d=a.d, lanes=a.lanes, T=a.chunk, steps=a.steps, seed=a.seed,
              device=a.device, ckpt=a.ckpt, log_every=50, data=a.data,
              talk=a.talk, constants=a.constants, arch=a.arch)


if __name__ == "__main__":
    main()
