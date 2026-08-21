"""v5.2 substrate — a standard decoder-only transformer, regular
attention intact, playing exactly the role PPO played in Crafter: an
unmodified, ordinary learner underneath the drive layer.

Same forward API as BandLM so the conveyor, drive layer, trainer and
eval plug in unchanged: forward(tokens, st, scene_starts) ->
(logits, st, ticks). Stateless across chunks — its memory IS its
attention window (the chunk length). ticks are empty (no band
predictors), so the imagination gate has no instrument here and, per
the F2 law, holds no authority (Drive imagination_absent=True).
"""

import torch
import torch.nn as nn

from .lm_bands import N_BANDS


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, mask, kv=None):
        """kv (A30): extra key/value context prepended to this
        block's own keys — the previous chunk's cached hiddens
        (Transformer-XL carry). mask must then be [Lq, len(kv)+Lq]."""
        h = self.ln1(x)
        if kv is not None:
            hk = torch.cat([kv, h], dim=1)
            a, _ = self.attn(h, hk, hk, attn_mask=mask,
                             need_weights=False)
        else:
            a, _ = self.attn(h, h, h, attn_mask=mask,
                             need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


def _rope_cache(T, rot, base, device, dtype):
    """cos/sin tables [T, rot//2] for rotary positions 0..T-1."""
    inv = 1.0 / (base ** (torch.arange(0, rot, 2, device=device,
                                       dtype=torch.float32) / rot))
    t = torch.arange(T, device=device, dtype=torch.float32)
    ang = torch.outer(t, inv)                         # [T, rot/2]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def _rotate(x, cos, sin):
    """Apply rotary to the first 2*cos.shape[-1] dims of x [B,H,L,hd];
    the remaining dims pass through (partial rotary, GPT-NeoX style)."""
    r2 = cos.shape[-1] * 2
    xr, xp = x[..., :r2], x[..., r2:]
    x1, x2 = xr[..., 0::2], xr[..., 1::2]
    c, s_ = cos[None, None], sin[None, None]
    o1 = x1 * c - x2 * s_
    o2 = x1 * s_ + x2 * c
    out = torch.stack([o1, o2], dim=-1).flatten(-2)
    return torch.cat([out, xp], dim=-1)


class RotaryBlock(nn.Module):
    """DECOUPLED ROPE block (v10.1 gated candidate, 2026-08-21).
    Same parameter count as Block (in/out projections + MLP), but
    attention is computed here so rotary position can be applied to
    the TEXT rows only: the first n_mem rows of the sequence are the
    band/store memory tokens, which have no position — their keys
    (and queries) are left unrotated and matched by content alone,
    text queries/keys rotate by chunk position (partial rotary on
    rot_frac of each head). qk_norm: per-head RMS normalization of q
    and k before the dot product (stability at depth). Boolean mask
    semantics follow nn.MultiheadAttention (True = blocked)."""

    def __init__(self, d, heads, n_mem=0, rot_frac=0.5, base=10000.0,
                 qk_norm=False):
        super().__init__()
        assert d % heads == 0
        self.d, self.h, self.hd = d, heads, d // heads
        self.n_mem = int(n_mem)
        rot = int(self.hd * rot_frac)
        self.rot = rot - rot % 2
        self.base = float(base)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.qn = nn.Parameter(torch.ones(self.hd))
            self.kn = nn.Parameter(torch.ones(self.hd))
        self._cache = None

    def _tables(self, L, device, dtype):
        c = self._cache
        if c is None or c[0] < L or c[1] != device or c[2] != dtype:
            cos, sin = _rope_cache(max(L, 8), self.rot, self.base,
                                   device, dtype)
            self._cache = (max(L, 8), device, dtype, cos, sin)
            c = self._cache
        return c[3][:L], c[4][:L]

    @staticmethod
    def _rms(x, g):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                               + 1e-6).to(x.dtype) * g

    def forward(self, x, mask, kv=None):
        if kv is not None:
            raise NotImplementedError("RotaryBlock: XL carry not "
                                      "supported (use_xl must be False)")
        B, L, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(self.d, dim=-1)
        q = q.view(B, L, self.h, self.hd).transpose(1, 2)  # [B,H,L,hd]
        k = k.view(B, L, self.h, self.hd).transpose(1, 2)
        v = v.view(B, L, self.h, self.hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self._rms(q, self.qn), self._rms(k, self.kn)
        M = self.n_mem
        Tt = L - M
        if Tt > 0 and self.rot > 0:
            cos, sin = self._tables(Tt, x.device, q.dtype)
            q = torch.cat([q[:, :, :M], _rotate(q[:, :, M:], cos, sin)],
                          dim=2)
            k = torch.cat([k[:, :, :M], _rotate(k[:, :, M:], cos, sin)],
                          dim=2)
        allow = ~mask if mask is not None else None
        a = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=allow)
        a = a.transpose(1, 2).reshape(B, L, self.d)
        x = x + self.proj(a)
        return x + self.mlp(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d=256, n_layers=6, n_heads=8,
                 max_T=512, talk=None, widths=None):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.max_T = max_T
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_T, d)
        self.blocks = nn.ModuleList(
            [Block(d, n_heads) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)
        self.lesioned = set()   # API compat; no bands to lesion

    def init_state(self, B, device):
        return {"step": 0}

    def detach_state(self, st):
        return st

    def forward(self, tokens, st, scene_starts=None):
        B, T = tokens.shape
        dev = tokens.device
        mask = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool),
                          diagonal=1)
        x = self.embed(tokens) + self.pos(
            torch.arange(min(T, self.max_T), device=dev))[None, :T]
        for b in self.blocks:
            x = b(x, mask)
        logits = self.head(self.lnf(x))
        st["step"] += T
        return logits, st, [[] for _ in range(N_BANDS)]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
