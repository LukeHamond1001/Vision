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
