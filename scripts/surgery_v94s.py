"""A65 surgery application — v94s_final -> v94sp (press-extended
substrate) + tokenizer_press.json, with old-token parity verified
on REAL stream tokens before anything ships.

Usage: python3 scripts/surgery_v94s.py <ckpt> <tokenizer.json> \
         <mix_r1_eval tokens.bin> <outdir>
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from iga.lm_data_ultrachat import load_tokenizer   # noqa: E402
from iga.lm_hybrid import HybridLM                 # noqa: E402
from iga.lm_vocab import PRESS_TOKENS, extend_model_state  # noqa: E402


def build(vocab, d=512, T=2048):
    return HybridLM(vocab, d=d, max_T=T, store="matrix",
                    keyed="logit", norm_mix=True, aux_trunk=0.2,
                    use_xl=False, gate_init=-2.0)


def main():
    ckpt, tok_path, bin_path, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    tok = load_tokenizer(tok_path)
    v0 = tok.get_vocab_size()
    added = tok.add_special_tokens(PRESS_TOKENS)
    ids = [tok.token_to_id(t) for t in PRESS_TOKENS]
    assert added == len(PRESS_TOKENS) and ids == list(
        range(v0, v0 + 4)), f"tokenizer surgery off: {ids}"
    tok.save(os.path.join(out, "tokenizer_press.json"))
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd2, V0, V1 = extend_model_state(st["model"])
    assert V0 == v0, f"ckpt vocab {V0} != tokenizer {v0}"
    print(f"extended {V0} -> {V1}; press ids {ids}", flush=True)

    m0 = build(V0)
    m0.load_state_dict(st["model"])
    m1 = build(V1)
    m1.load_state_dict(sd2)
    m0.eval()
    m1.eval()
    toks = np.fromfile(bin_path, dtype=np.uint16, count=4096)
    x = torch.tensor(toks.astype(np.int64)).view(2, 2048)
    st0, st1 = m0.init_state(1, "cpu"), m1.init_state(1, "cpu")
    worst_old, worst_new = 0.0, -1e9
    with torch.no_grad():
        for c in range(2):
            xc = x[c:c + 1]
            l0, st0, _ = m0(xc, st0, None)
            m0.pop_write_cost()
            m0.pop_recon()
            st0 = m0.detach_state(st0)
            l1, st1, _ = m1(xc, st1, None)
            m1.pop_write_cost()
            m1.pop_recon()
            st1 = m1.detach_state(st1)
            worst_old = max(worst_old,
                            float((l0 - l1[..., :V0]).abs().max()))
            worst_new = max(worst_new, float(l1[..., V0:].max()))
    print(f"parity: old-token max |d| {worst_old:.2e} "
          f"(bar 1e-4)  new-token max logit {worst_new:.1f} "
          f"(bar -10)", flush=True)
    assert worst_old < 1e-4 and worst_new < -10.0, "PARITY FAILED"
    torch.save({"model": sd2, "step": st.get("step"),
                "surgery": {"from": os.path.basename(ckpt),
                            "seed": 7, "press_ids": ids}},
               os.path.join(out, "v94sp.pt"))
    print(f"SURGERY OK -> {out}/v94sp.pt (step {st.get('step')})",
          flush=True)


if __name__ == "__main__":
    main()
