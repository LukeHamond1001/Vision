"""Talk to a scan organism checkpoint (v14+): local CPU or pod GPU.

Loads a driver checkpoint (model + cfg + saved states), rebuilds the
ScanLM, and generates from a prompt. Presses are banned from the mouth
(A64); silence is allowed and printed as [pause]. Lesion switches show
the organs are load-bearing, live.

  python scripts/scan_infer.py CKPT TOKENIZER.json [--manifest M.json]
      [--prompt "text"] [--n 120] [--temp 0.9] [--greedy]
      [--lesion none|bands|store|both] [--wake] [--ban-silence]

--wake seeds lane 0 from the checkpoint's saved band/store states (the
organism mid-life); default is a newborn lane (empty stores, zero bands).
"""
import argparse, json, sys, pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from iga.lm_scan import ScanLM  # noqa: E402


def _lane0(x):
    if torch.is_tensor(x):
        return x[:1].clone() if x.dim() >= 1 and x.shape[0] > 1 else x.clone()
    if isinstance(x, dict):
        return {k: _lane0(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_lane0(v) for v in x]
    return x


def _to_dev(x, dev):
    if torch.is_tensor(x):
        return x.to(dev)
    if isinstance(x, dict):
        return {k: _to_dev(v, dev) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_to_dev(v, dev) for v in x]
        return tuple(t) if isinstance(x, tuple) else t
    return x


def load_scan(ckpt_path, tok, dev="cpu"):
    """Rebuild a ScanLM from a driver or synthetic checkpoint. Returns
    (model, state_dict_bundle)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(state["cfg"])
    if "vocab_size" in cfg:                      # synthetic/test format
        V = cfg.pop("vocab_size")
        m = ScanLM(V, **cfg)
    else:                                        # the driver's model_cfg
        V = state["model"]["embed.weight"].shape[0]
        kw = dict(d=cfg["d"], n_layers=cfg["n_layers"],
                  n_heads=cfg.get("n_heads", 8), max_T=cfg.get("T", 64),
                  mlp=cfg.get("mlp", "gelu"),
                  aux_trunk=cfg.get("aux_trunk", 0))
        if cfg.get("clocks"):
            kw["clocks"] = {int(k): int(v) for k, v in cfg["clocks"].items()}
        if cfg.get("gate_init") is not None:
            kw["gate_init"] = cfg["gate_init"]
        m = ScanLM(V, **kw, **(cfg.get("scan") or {}))
        m.autocast_bf16 = cfg.get("precision") == "bf16"
    m = m.to(dev)
    dead = ("nov_max", "veto_eye", "veto_w", "veto_b", "cstores",
            "ckey_proj", "cquery_proj", "calpha", "ctx_in", "cmu")
    sd = {k: v for k, v in state["model"].items()
          if not any(k.startswith(d) or k == d for d in dead)}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    grafts = [k for k in missing if k.startswith("goal_")]
    hard = [k for k in missing if not k.startswith("goal_")]
    assert not hard, f"missing non-graft keys: {hard[:4]}"
    if grafts:
        print(f"[load_scan] grafted organs at default init: {grafts}",
              file=__import__("sys").stderr)
    sil = tok.token_to_id("<pad>")
    press = {tok.token_to_id(t): lv for lv, t in
             enumerate(("<+1>", "<+2>", "<-1>", "<-2>"), 1)
             if tok.token_to_id(t) is not None}
    if press and hasattr(m, "set_reward_tokens"):
        m.set_reward_tokens(press)
    eh, em = tok.token_to_id("<eot_human>"), tok.token_to_id("<eot_model>")
    if eh is not None and em is not None and hasattr(m, "set_eot_ids"):
        m.set_eot_ids(eh, em)
    return m, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("tok")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--lesion", default="none",
                    choices=["none", "bands", "store", "both"])
    ap.add_argument("--wake", action="store_true")
    ap.add_argument("--ban-silence", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store-boost", type=float, default=1.0,
                    help="49q: amplify the hippocampus's logit vote at "
                         "serve (mechanism verified correct at 57%, "
                         "gain young — this is the volume knob)")
    ap.add_argument("--prompt-shard", default=None,
                    help="life shard dir: prompt = raw tokens from tokens.bin")
    ap.add_argument("--prompt-lane", type=int, default=0)
    ap.add_argument("--prompt-off", type=int, default=0)
    ap.add_argument("--prompt-len", type=int, default=2048)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tok)
    m, state = load_scan(a.ckpt, tok, dev)
    m = m.eval()
    print(f"[ckpt step {state.get('step')}] "
          f"{sum(p.numel() for p in m.parameters())/1e6:.1f}M params on {dev}",
          file=sys.stderr)
    sil_id = tok.token_to_id("<pad>")

    m.lesioned = {3, 4, 5, 6, 7, 8} if a.lesion in ("bands", "both") else set()
    m.store_boost = float(a.store_boost)
    if a.lesion in ("store", "both"):
        m.store_read_off = True

    if a.wake and state.get("st") is not None:
        st = _to_dev(_lane0(state["st"]), dev)
        print("[wake: seeded from the checkpoint's lane-0 states]", file=sys.stderr)
    else:
        st = m.init_state(1, dev)

    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    if a.prompt_shard:
        import numpy as np
        man = json.load(open(pathlib.Path(a.prompt_shard) / "manifest.json"))
        life_len = man["life_len"]
        mm = np.memmap(pathlib.Path(a.prompt_shard) / "tokens.bin",
                       dtype=np.uint16, mode="r")
        s0 = a.prompt_lane * life_len + a.prompt_off
        ids = [int(t) for t in mm[s0:s0 + a.prompt_len]]
        print(f"[prompt: shard lane {a.prompt_lane} off {a.prompt_off} "
              f"len {len(ids)}]", file=sys.stderr)
    else:
        import re as _re
        ids = []
        if a.prompt:
            for part in _re.split(r"(<eot_human>|<eot_model>|<pad>)", a.prompt):
                if not part:
                    continue
                sid = tok.token_to_id(part) if part.startswith("<") else None
                if sid is not None:
                    ids.append(sid)
                else:
                    ids.extend(tok.encode(part).ids)
    if not ids:
        hid = tok.token_to_id("<eot_human>")
        ids = [hid if hid is not None else 0]
    logits = None
    with torch.no_grad():
        for i in range(0, len(ids), 64):          # windowless: chunked feed
            x = torch.tensor([ids[i:i + 64]], device=dev)
            logits, st, _ = m(x, st)
        out = []
        for _ in range(a.n):
            lg = logits[0, -1].float()
            if hasattr(m, "ban_presses"):
                lg = m.ban_presses(lg)
            if a.ban_silence and sil_id is not None:
                lg[sil_id] = float("-inf")
            if a.greedy:
                nxt = int(lg.argmax())
            else:
                pr = torch.softmax(lg / max(a.temp, 1e-4), -1).cpu()
                nxt = int(torch.multinomial(pr, 1, generator=gen))
            out.append(nxt)
            x = torch.tensor([[nxt]], device=dev)
            logits, st, _ = m(x, st)
    if a.prompt_shard:
        tail = [t for t in ids[-160:]]
        print("[...prompt tail]", tok.decode(tail), "\n[generation:]",
              file=sys.stderr)
    text, pauses = [], 0
    for t in out:
        if t == sil_id:
            pauses += 1
            text.append("[pause]")
        else:
            text.append(tok.decode([t]))
    print("".join(text))
    print(f"\n[{len(out)} tokens, {pauses} pauses, lesion={a.lesion}]",
          file=sys.stderr)


if __name__ == "__main__":
    main()
