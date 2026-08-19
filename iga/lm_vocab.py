"""A65 (Phase 3 opener) — vocab surgery: extend a trained HybridLM
checkpoint with the four press tokens, OUTPUT-PARITY-EXACT.

New embed rows: small random (own generator — deterministic, no
global RNG). New head/aux rows: weight 0, bias -20 — the new
logits sit ~e^-20 below everything, so old-token probabilities are
preserved to numerical precision (the parity law test). tok_u
gains zeros (press tokens start at the neutral write strength any
fresh token has). Everything else in the model is vocab-free.

The BPE tokenizer gains the press tokens as appended specials
(ids V..V+3) at serve time; the synthetic weaver already carries
them natively.
"""

import torch

PRESS_TOKENS = ["<+1>", "<+2>", "<-1>", "<-2>"]
NEG_BIAS = -20.0


def extend_model_state(sd, n_new=4, seed=7):
    """state_dict -> (extended copy, V_old, V_new)."""
    sd = {k: v.clone() for k, v in sd.items()}
    V, d = sd["embed.weight"].shape
    g = torch.Generator().manual_seed(seed)
    sd["embed.weight"] = torch.cat(
        [sd["embed.weight"],
         0.02 * torch.randn(n_new, d, generator=g)], 0)
    for w, b in (("head.weight", "head.bias"),
                 ("aux_head.weight", "aux_head.bias")):
        if w in sd:
            sd[w] = torch.cat(
                [sd[w], torch.zeros(n_new, sd[w].shape[1])], 0)
            sd[b] = torch.cat(
                [sd[b], torch.full((n_new,), NEG_BIAS)], 0)
    if "tok_u" in sd:
        sd["tok_u"] = torch.cat([sd["tok_u"], torch.zeros(n_new)], 0)
    return sd, V, V + n_new
