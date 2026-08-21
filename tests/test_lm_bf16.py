"""bf16 MIXED-PRECISION LAWS (v10.1, 2026-08-21). Autocast covers the
trunk blocks only; everything the memory apparatus touches stays
fp32: band states (h/acc/pend), the store matrices, logits, losses.
Weights are fp32 masters, so checkpoints are precision-agnostic."""
import torch
import pytest
from iga.lm_hybrid import HybridLM

KW = dict(store="matrix", keyed="logit", norm_mix=True, aux_trunk=0.2,
          use_xl=False, gate_init=-2.0, clocks={3: 1, 4: 8, 5: 64, 6: 512})


def _model(attn="abs", seed=0):
    torch.manual_seed(seed)
    return HybridLM(512, d=64, n_layers=2, n_heads=4, max_T=64, attn=attn,
                    qk_norm=(attn == "rope"), **KW)


@pytest.mark.parametrize("attn", ["abs", "rope"])
def test_states_store_logits_stay_fp32_under_autocast(attn):
    m = _model(attn)
    x = torch.randint(0, 512, (2, 64))
    st = m.init_state(2, "cpu")
    lg32, _, _ = m(x, st, None)
    m.autocast_bf16 = True
    st = m.init_state(2, "cpu")
    for _ in range(9):                      # several chunks: ticks fire
        lg16, st, _ = m(x, st, None)
        st = m.detach_state(st)
    assert lg16.dtype == torch.float32
    for grp in ("h", "acc", "M"):
        assert all(v.dtype == torch.float32 for v in st[grp].values())
    assert all(p.dtype == torch.float32 for p in m.parameters())
    # first-chunk deviation is bf16-sized, not a different model
    m.autocast_bf16 = True
    st = m.init_state(2, "cpu")
    lg16a, _, _ = m(x, st, None)
    rel = float((lg16a - lg32).norm() / lg32.norm())
    assert rel < 0.02


def test_training_trajectories_agree(tmp_path):
    """Same seed/data, fp32 vs bf16: CE after a short run within a few
    percent, and the bf16 checkpoint reloads into an fp32 model."""
    from iga.lm_train import train
    out = {}
    for prec in ("fp32", "bf16"):
        ck = str(tmp_path / f"{prec}.pt")
        model, drive, vocab, ce0, ce1 = train(
            d=32, lanes=2, T=64, steps=500, seed=0, device="cpu",
            arch="hybrid", store="matrix", keyed="logit", norm_mix=True,
            aux_trunk=0.2, use_xl=False, gate_init=-2.0, log_every=250,
            n_layers=1, precision=prec, ckpt=ck)
        out[prec] = (ce1, ck, model.autocast_bf16)
    assert out["fp32"][2] is False and out["bf16"][2] is True
    ce32, ce16 = out["fp32"][0], out["bf16"][0]
    assert abs(ce16 - ce32) < 0.05 * ce32 + 0.05
    blob = torch.load(out["bf16"][1], map_location="cpu", weights_only=False)
    assert blob["cfg"]["precision"] == "bf16"
    assert all(v.dtype == torch.float32 for v in blob["model"].values())
    for grp in ("h", "acc"):
        assert all(v.dtype == torch.float32 for v in blob["st"][grp].values())
    kw = dict(KW); kw.pop("clocks")          # train() default ladder
    m2 = HybridLM(blob["model"]["embed.weight"].shape[0], d=32, n_layers=1,
                  n_heads=8, max_T=64, **kw)
    m2.load_state_dict(blob["model"], strict=True)   # precision-agnostic
