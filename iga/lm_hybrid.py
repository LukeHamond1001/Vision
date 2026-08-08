"""v5.3 — the complete architecture as an LLM (A19).

Attention is the hands, the bands are the memory, imagination is the
lookahead, the ledger is the law.

Substrate: a standard decoder-only transformer over each 512-token
window (language, in-window lookup). Riding above it: slow band
latents at clocks 512 / 4096 / 32768 — persistent across the whole
run, updated from pooled transformer states at their ticks, each with
a predictor (the forward model: imagination's instrument, so the gate
is LIVE here). The slow latents are injected back as MEMORY TOKENS
the transformer attends over — query-conditioned readout of the
latent ladder, the multiplicative lookup the pure-band machine
lacked.

Same forward API as BandLM/TransformerLM: (tokens, st, _) ->
(logits, st, ticks). Lesion zeroes the memory tokens and band reads —
the control is meaningful again.
"""

import torch
import torch.nn as nn

from .lm_bands import N_BANDS
from .lm_transformer import Block

HYBRID_CLOCKS = {3: 1, 4: 8, 5: 64}   # band idx -> tick every N chunks
                                       # (chunks of 512 -> 512/4k/32k)


class HybridLM(nn.Module):
    def __init__(self, vocab_size, d=128, n_layers=6, n_heads=8,
                 max_T=512, talk=None, widths=None):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.max_T = max_T
        self.bands = sorted(HYBRID_CLOCKS)          # [3, 4, 5]
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_T + len(self.bands), d)
        self.blocks = nn.ModuleList(
            [Block(d, n_heads) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)
        self.cells = nn.ModuleDict(
            {str(k): nn.GRUCell(d, d) for k in self.bands})
        self.pred = nn.ModuleDict(
            {str(k): nn.Linear(d, d) for k in self.bands})
        self.mem_proj = nn.ModuleDict(
            {str(k): nn.Linear(d, d, bias=False) for k in self.bands})
        self.lesioned = set()

    def init_state(self, B, device):
        return {"h": {k: torch.zeros(B, self.d, device=device)
                      for k in self.bands},
                "acc": {k: torch.zeros(B, self.d, device=device)
                        for k in self.bands},
                "cnt": {k: 0 for k in self.bands},
                "pend": {k: None for k in self.bands},
                "chunk": 0}

    def detach_state(self, st):
        st["h"] = {k: v.detach() for k, v in st["h"].items()}
        st["acc"] = {k: v.detach() for k, v in st["acc"].items()}
        st["pend"] = {k: (p.detach() if p is not None else None)
                      for k, p in st["pend"].items()}
        return st

    def _mem_tokens(self, st, B):
        toks = []
        for k in self.bands:
            h = st["h"][k]
            if k in self.lesioned:
                h = torch.zeros_like(h)
            toks.append(self.mem_proj[str(k)](h))
        return torch.stack(toks, dim=1)          # [B, M, d]

    def forward(self, tokens, st, scene_starts=None):
        B, T = tokens.shape
        dev = tokens.device
        M = len(self.bands)
        mem = self._mem_tokens(st, B)
        x = self.embed(tokens) + self.pos(
            torch.arange(M, M + T, device=dev))[None]
        mem = mem + self.pos(torch.arange(M, device=dev))[None]
        x = torch.cat([mem, x], dim=1)           # [B, M+T, d]
        # causal over text; every text position may attend all memory;
        # memory rows attend only themselves
        mask = torch.triu(torch.ones(M + T, M + T, device=dev,
                                     dtype=torch.bool), diagonal=1)
        mask[:M, :] = True
        mask[torch.arange(M + T), torch.arange(M + T)] = False
        for b in self.blocks:
            x = b(x, mask)
        hidden = x[:, M:]                        # text positions
        logits = self.head(self.lnf(hidden))
        # band updates from the pooled window, on their clocks
        window = hidden.mean(dim=1)
        ticks = [[] for _ in range(N_BANDS)]
        st["chunk"] += 1
        for k in self.bands:
            st["acc"][k] = st["acc"][k] + window
            st["cnt"][k] += 1
            if st["chunk"] % HYBRID_CLOCKS[k] == 0:
                pooled = st["acc"][k] / max(st["cnt"][k], 1)
                if st["pend"][k] is not None:
                    fid = nn.functional.cosine_similarity(
                        st["pend"][k], pooled, dim=-1)
                    ticks[k].append((T - 1, fid))
                st["h"][k] = self.cells[str(k)](pooled, st["h"][k])
                st["pend"][k] = self.pred[str(k)](st["h"][k])
                st["acc"][k] = torch.zeros_like(st["acc"][k])
                st["cnt"][k] = 0
        return logits, st, ticks

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
