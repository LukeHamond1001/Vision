"""A77 — dream on a leash (true REM, gated; V10_FLASH 1c).

The math that makes it safe (verified 2026-08-19): model collapse
is a theorem for the REPLACE regime; the ACCUMULATE regime — real
data never leaves, self-data a bounded sliver — has bounded error
(Gerstgrasser 2024), and verifier-gated self-data can improve. Our
night sits in the safe regime on four axes:
  1. real fraction ~99%: dreams are <= a sliver of sleep blocks and
     the wake stream never stops;
  2. selection is EXTERNAL: the frozen public judge scores every
     candidate; below-floor dreams train nothing (and an optional
     fact-consistency closure — the pod driver injects one built
     from the shard manifest — vetoes contradictions, because
     generation samples the STRONGEST belief (A67-P6) and an
     unguarded dream would rehearse convictions);
  3. generation is SEEDED by a real press-paid memory span
     (conditioned, manifold-tethered — like real dreams);
  4. the collapse signature is monitored: every dream block logs
     the generations' distinct-3gram ratio; the caller (heartbeat/
     driver) kills dreaming on contraction.

One dream block = one tiny-lr CE step on [real seed + judged-best
continuation], trunk-alone, mastery floor applies. Everything
else — freeze surface, eval mode, optimizer hygiene — is the
certified sleep block's.
"""

import torch

from .lm_sleep import FREEZE_EXACT, FREEZE_PREFIXES, MIN_REPLAY


@torch.no_grad()
def _feed(model, st, ids, device):
    """Run ids through the model in <= max_T chunks (the one-token
    organism has no window; max_T is its batching unit), returning the
    last logits and the advanced, detached state."""
    step = model.max_T if getattr(model, "windowless", False) else len(ids)
    lg = None
    for i in range(0, len(ids), step):
        x = torch.tensor([ids[i:i + step]], dtype=torch.long, device=device)
        lg, st, _ = model(x, st, None)
        model.pop_write_cost()
        model.pop_recon()
        st = model.detach_state(st)
    return lg, st


def _generate(model, seed_ids, n, tau, max_new, device, gen):
    outs = []
    for _ in range(n):
        st = model.init_state(1, device)
        cont = []
        lg, st = _feed(model, st, list(seed_ids), device)
        x = None
        for _ in range(max_new):
            if x is not None:
                lg, st, _ = model(x, st, None)
                model.pop_write_cost()
                model.pop_recon()
                st = model.detach_state(st)
            probs = torch.softmax(lg[0, -1].float() / tau, -1)
            nxt = int(torch.multinomial(probs, 1, generator=gen))
            cont.append(nxt)
            x = torch.tensor([[nxt]], device=device)
        outs.append(cont)
    return outs


def _distinct3(ids):
    if len(ids) < 3:
        return 1.0
    g = [tuple(ids[i:i + 3]) for i in range(len(ids) - 2)]
    return len(set(g)) / len(g)


def dream_block(model, opt, sleeper, tok, judge_fn, step,
                fact_check=None, n=4, tau=0.8, max_new=48,
                ctx=96, min_q=None, min_step_loss=1e-4,
                gen_seed=None):
    """One leashed dream. Returns a stats row or None (no seed /
    all candidates rejected — rejection is a feature)."""
    import random as _random
    if not sleeper.spans or sleeper.buffers is None:
        return None
    rng = sleeper.rng
    # the seed draw follows the night's replay weights — pay, and
    # with the saliency channel on, the hippocampus's |RPE| stamp
    # (dreams start from what the day marked; weights == pay at
    # saliency 0, so the certified draw is unchanged)
    weights = (sleeper._span_weights() if hasattr(sleeper, "_span_weights")
               else [s["pay"] for s in sleeper.spans])
    if not any(w > 0 for w in weights):
        return None
    span = rng.choices(sleeper.spans, weights=weights)[0]
    # the whole dreamed sequence [seed + continuation] must fit the
    # model's window (pos table = max_T + bands) — unless the model
    # streams tokens (ScanLM.windowless), where max_T is only the
    # batching unit and the seed is fed in chunks
    if not getattr(model, "windowless", False):
        ctx = min(ctx, model.max_T - max_new - 1)
    if ctx < MIN_REPLAY:
        return None
    hi = min(span["t1"], sleeper.end)
    lo = max(span["t0"], sleeper.start, hi - ctx)
    seed_ids = sleeper.buffers[span["lane"]][lo - sleeper.start:
                                             hi - sleeper.start]
    if len(seed_ids) < MIN_REPLAY:
        return None
    device = next(model.parameters()).device
    gen = torch.Generator(device="cpu")
    gen.manual_seed(gen_seed if gen_seed is not None
                    else _random.Random(step).randrange(2 ** 31))
    was_training = model.training
    model.eval()
    model.store_read_off = True
    try:
        conts = _generate(model, seed_ids, n, tau, max_new,
                          device, gen)
    finally:
        model.store_read_off = False
    seed_txt = tok.decode(seed_ids)
    scored = []
    d3s = []
    for cont in conts:
        txt = tok.decode(cont)
        d3s.append(_distinct3(cont))
        q = judge_fn(seed_txt, txt)
        ok = fact_check(txt) if fact_check is not None else True
        scored.append((q if ok else -1.0, cont, txt))
    scored.sort(key=lambda x: -x[0])
    best_q, best, _ = scored[0]
    floor = min_q if min_q is not None else 0.5
    row = {"step": step, "arm": "DREAM", "seed": (lo, hi),
           "pay": span["pay"], "best_q": round(best_q, 4),
           "distinct3": round(sum(d3s) / len(d3s), 4),
           "stepped": False}
    if best_q < floor:
        sleeper.stats.append(row)
        return row                      # rejected: nothing trains
    ids = list(seed_ids) + list(best)
    saved = [(p, p.requires_grad) for nm, p in model.named_parameters()
             if nm.startswith(FREEZE_PREFIXES) or nm in FREEZE_EXACT]
    for p, _ in saved:
        p.requires_grad_(False)
    model.store_read_off = True
    try:
        xs, ys = ids[:-1], ids[1:]
        step_len = model.max_T if getattr(model, "windowless", False) else len(xs)
        with torch.enable_grad():
            # the seed is real replay; the dream part carries the
            # gradient story — train on the WHOLE sequence (the
            # accumulate regime: real anchor inside every step). A
            # windowless model takes it in wake-sized chunks with the
            # state detached between them (wake semantics), one
            # backward over the summed CE.
            st = model.init_state(1, device)
            tot, n_tok = None, 0
            for i in range(0, len(xs), step_len):
                x = torch.tensor([xs[i:i + step_len]], dtype=torch.long, device=device)
                y = torch.tensor([ys[i:i + step_len]], dtype=torch.long, device=device)
                lg, st, _ = model(x, st, None)
                model.pop_write_cost()
                model.pop_recon()
                if hasattr(model, "pop_value_loss"):
                    model.pop_value_loss()
                if hasattr(model, "pop_bg_loss"):
                    model.pop_bg_loss()
                st = model.detach_state(st)
                l_i = torch.nn.functional.cross_entropy(
                    lg.float().reshape(-1, model.vocab_size),
                    y.reshape(-1), reduction="sum")
                tot = l_i if tot is None else tot + l_i
                n_tok += y.numel()
            loss = tot / n_tok
            if hasattr(model, "value") and "3" in getattr(model, "value", {}) \
                    and isinstance(st, dict) and 3 in st.get("h", {}):
                with torch.no_grad():      # logged, never used to select (no self-grading)
                    row["v_end"] = round(float(model.value["3"](st["h"][3]).mean()), 4)
            if abs(float(loss.detach())) >= min_step_loss:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               1.0)
                opt.step()
                sleeper.steps_taken += 1
                sleeper._downscale(model)
                row["stepped"] = True
                row["loss"] = round(float(loss.detach()), 4)
    finally:
        model.store_read_off = False
        for p, rg in saved:
            p.requires_grad_(rg)
        if was_training:
            model.train()
        opt.zero_grad()
    sleeper.stats.append(row)
    return row
