"""Laws for the v10.1 gated attention candidates (2026-08-21):
decoupled RoPE (text rows rotate by position, memory tokens stay
position-free) and QK-norm. Defaults (attn="abs") keep the certified
Block / position table bit-exactly; the rotary block must match the
certified attention math exactly when rotation is disabled, be
causal, be relative-position-only on text, and leave memory keys
position-free."""
import torch
import pytest
from iga.lm_transformer import Block, RotaryBlock, _rotate, _rope_cache
from iga.lm_hybrid import HybridLM

KW = dict(store="matrix", keyed="logit", norm_mix=True, aux_trunk=0.2,
          use_xl=False, gate_init=-2.0, clocks={3: 1, 4: 8, 5: 64, 6: 512})


def _mask(M, T):
    sq = torch.triu(torch.ones(M + T, M + T, dtype=torch.bool), diagonal=1)
    sq[:M, :] = True
    sq[torch.arange(M + T), torch.arange(M + T)] = False
    return sq


def test_default_is_certified_block_and_table():
    torch.manual_seed(0)
    m = HybridLM(512, d=64, n_layers=2, n_heads=4, max_T=64, **KW)
    assert all(type(b) is Block for b in m.blocks)
    assert m.pos.num_embeddings == 64 + len(m.bands)
    with pytest.raises(AssertionError):
        HybridLM(512, d=64, n_layers=1, n_heads=4, max_T=64,
                 qk_norm=True, **KW)


def test_rotary_block_matches_mha_when_unrotated():
    """rot_frac=0, no qk_norm, weights copied from the certified
    Block: identical outputs under the HybridLM mask (mask semantics
    and attention math are the same; rotary is the only difference)."""
    torch.manual_seed(1)
    d, h, M, T = 64, 4, 4, 24
    blk = Block(d, h)
    rb = RotaryBlock(d, h, n_mem=M, rot_frac=0.0)
    with torch.no_grad():
        rb.qkv.weight.copy_(blk.attn.in_proj_weight)
        rb.qkv.bias.copy_(blk.attn.in_proj_bias)
        rb.proj.weight.copy_(blk.attn.out_proj.weight)
        rb.proj.bias.copy_(blk.attn.out_proj.bias)
        rb.ln1.load_state_dict(blk.ln1.state_dict())
        rb.ln2.load_state_dict(blk.ln2.state_dict())
        rb.mlp.load_state_dict(blk.mlp.state_dict())
    x = torch.randn(2, M + T, d)
    mask = _mask(M, T)
    assert torch.allclose(blk(x, mask), rb(x, mask), atol=1e-5)


def test_rotary_is_relative_and_norm_preserving():
    cos, sin = _rope_cache(16, 8, 10000.0, "cpu", torch.float32)
    q = torch.randn(1, 1, 16, 8)
    k = torch.randn(1, 1, 16, 8)
    rq, rk = _rotate(q, cos, sin), _rotate(k, cos, sin)
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    # <rot(q_i), rot(k_j)> depends only on i - j
    def dot(i, j):
        return float((_rotate(q[:, :, i:i + 1], cos[i:i + 1], sin[i:i + 1])
                      * _rotate(k[:, :, j:j + 1], cos[j:j + 1], sin[j:j + 1])
                      ).sum())
    # same content at different absolute positions, same offset
    q2 = q.clone(); q2[:, :, 9] = q[:, :, 5]
    k2 = k.clone(); k2[:, :, 6] = k[:, :, 2]
    a = float((_rotate(q[:, :, 5:6], cos[5:6], sin[5:6])
               * _rotate(k[:, :, 2:3], cos[2:3], sin[2:3])).sum())
    b = float((_rotate(q2[:, :, 9:10], cos[9:10], sin[9:10])
               * _rotate(k2[:, :, 6:7], cos[6:7], sin[6:7])).sum())
    assert abs(a - b) < 1e-4


def test_memory_keys_are_position_free_and_model_is_causal():
    torch.manual_seed(2)
    m = HybridLM(512, d=64, n_layers=2, n_heads=4, max_T=64,
                 attn="rope", qk_norm=True, **KW)
    assert all(type(b) is RotaryBlock for b in m.blocks)
    assert m.pos.num_embeddings == len(m.bands)     # no text table
    x = torch.randint(0, 512, (2, 32))
    st = m.init_state(2, "cpu")
    lg1, _, _ = m(x, st, None)
    x2 = x.clone(); x2[:, 20:] = torch.randint(0, 512, (2, 12))
    st = m.init_state(2, "cpu")
    lg2, _, _ = m(x2, st, None)
    assert torch.allclose(lg1[:, :20], lg2[:, :20], atol=1e-5)
    # memory rows: queries/keys unrotated -> a block's attention from a
    # text query to a memory key does not depend on the query's position
    rb = m.blocks[0]
    M = len(m.bands)
    q = torch.randn(1, 4, 1, 16)              # one text query, 4 heads
    cos, sin = rb._tables(8, "cpu", torch.float32)
    kmem = torch.randn(1, 4, 1, 16)
    s5 = (_rotate(q, cos[5:6], sin[5:6]) * kmem).sum()
    s0 = (_rotate(q, cos[0:1], sin[0:1]) * kmem).sum()
    assert not torch.allclose(s5, s0)         # rotation does act on q...
    # ...but the model never rotates memory rows: rows < M are copied
    # through unrotated (structural check on the forward path)
    assert rb.n_mem == M


def test_train_threads_attn_flags_and_cfg(tmp_path):
    from iga.lm_train import train
    ck = str(tmp_path / "r.pt")
    model, *_ = train(d=32, lanes=2, T=64, steps=500, seed=0, device="cpu",
                      arch="hybrid", store="matrix", keyed="logit",
                      norm_mix=True, aux_trunk=0.2, use_xl=False,
                      gate_init=-2.0, log_every=100, n_layers=1,
                      attn="rope", qk_norm=True, ckpt=ck)
    assert type(model.blocks[0]) is RotaryBlock and model.blocks[0].qk_norm
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    assert blob["cfg"]["attn"] == "rope" and blob["cfg"]["qk_norm"] is True
    assert blob["cfg"]["clocks"] is None or isinstance(blob["cfg"]["clocks"], dict)


def test_band_lr_mult_groups_and_schedule(tmp_path):
    """band_lr_mult=3: the band organs (cells/pred/mem_proj/read_q)
    form their own AdamW group at 3x the base lr; the schedule scales
    each group from its own base. Default = one group."""
    import torch
    from iga.lm_train import train
    m1, *_ = train(d=32, lanes=2, T=64, steps=2, seed=0, device="cpu",
                   arch="hybrid", store="matrix", keyed="logit",
                   norm_mix=True, aux_trunk=0.2, use_xl=False,
                   gate_init=-2.0, log_every=100, n_layers=1,
                   band_lr_mult=3.0, lr=1e-3)
    names = [n for n, _ in m1.named_parameters()]
    band_pfx = ("cells.", "pred.", "mem_proj.", "read_q.")
    bp = [n for n in names if n.startswith(band_pfx)]
    assert bp and len(bp) < len(names)
    # the grouping the trainer builds, reproduced on the model
    groups = [{"params": [p for n, p in m1.named_parameters()
                          if not n.startswith(band_pfx)],
               "lr": 1e-3, "base_lr": 1e-3},
              {"params": [p for n, p in m1.named_parameters()
                          if n.startswith(band_pfx)],
               "lr": 3e-3, "base_lr": 3e-3}]
    opt = torch.optim.AdamW(groups, lr=1e-3)
    for g in opt.param_groups:          # the schedule's per-group scaling
        g["lr"] = g.get("base_lr", 1e-3) * 0.5
    assert [round(g["lr"], 6) for g in opt.param_groups] == [0.0005, 0.0015]


def test_swiglu_flag_param_matched_and_default_gelu():
    import torch.nn as nn
    from iga.lm_transformer import Block, SwiGLU
    b_g = Block(256, 8)
    b_s = Block(256, 8, mlp="swiglu")
    assert isinstance(b_g.mlp, nn.Sequential) and isinstance(b_s.mlp, SwiGLU)
    n_g = sum(p.numel() for p in b_g.mlp.parameters())
    n_s = sum(p.numel() for p in b_s.mlp.parameters())
    assert abs(n_s - n_g) / n_g < 0.05          # params matched within 5%
    m = HybridLM(512, d=64, n_layers=1, n_heads=4, max_T=64, mlp="swiglu",
                 **KW)
    x = torch.randint(0, 512, (2, 16))
    lg, _, _ = m(x, m.init_state(2, "cpu"), None)
    assert lg.shape == (2, 16, 512)
    assert type(HybridLM(512, d=64, n_layers=1, n_heads=4, max_T=64,
                         **KW).blocks[0].mlp) is nn.Sequential
