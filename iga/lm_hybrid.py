"""v5.4 — the complete architecture as an LLM (A19 + A24).

Attention is the hands, the bands are the memory, imagination is the
lookahead, the ledger is the law.

Substrate: a standard decoder-only transformer over each 512-token
window (language, in-window lookup). Riding above it: slow band
latents at clocks 512 / 4096 / 32768 — persistent across the whole
run, injected back as MEMORY TOKENS the transformer attends over.

v5.4 (A24, from the run-4 autopsy): the v5.3 bands were written with
the MEAN of all window hiddens each tick — an unforecastable soup;
fid:4/5 sat below floor all run, the gate vetoed every cross-window
frontier, and the real carry transient died unpaid. Now:
  selective read — each band attention-pools the window with its own
    learned query (it chooses what to look at, not everything);
  closed-by-default write — an exposed-gate slow cell, update gate
    biased shut at init, so the state drifts slowly and the band's
    forward model has something learnable to predict;
  write cost — a small penalty on open gates keeps writes sparse.

Same forward API: (tokens, st, _) -> (logits, st, ticks). Lesion
zeroes the memory tokens. The trainer reads model.pop_write_cost().
"""

import torch
import torch.nn as nn

from .lm_bands import N_BANDS
from .lm_transformer import Block

HYBRID_CLOCKS = {3: 1, 4: 8, 5: 64}   # band idx -> tick every N chunks
                                       # (chunks of 512 -> 512/4k/32k)


class SlowCell(nn.Module):
    """Gated delta-write with the update gate biased closed at init:
    h' = (1-z)*h + z*cand. At init z ~ sigmoid(gate_bias) so the
    state barely moves until training earns the right to write."""

    def __init__(self, d, gate_bias=-2.0):
        super().__init__()
        self.z = nn.Linear(2 * d, d)
        self.cand = nn.Linear(2 * d, d)
        nn.init.constant_(self.z.bias, gate_bias)

    def forward(self, x, h):
        hx = torch.cat([h, x], dim=-1)
        z = torch.sigmoid(self.z(hx))
        c = torch.tanh(self.cand(hx))
        return (1 - z) * h + z * c, z.mean()


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
            {str(k): SlowCell(d) for k in self.bands})
        self.read_q = nn.ParameterDict(
            {str(k): nn.Parameter(torch.randn(d) / d ** 0.5)
             for k in self.bands})
        self.pred = nn.ModuleDict(
            {str(k): nn.Linear(d, d) for k in self.bands})
        self.mem_proj = nn.ModuleDict(
            {str(k): nn.Linear(d, d, bias=False) for k in self.bands})
        self.lesioned = set()
        self._write_cost = None

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
        # band updates: each band SELECTS from the window with its own
        # query (A24) instead of receiving the window mean
        ticks = [[] for _ in range(N_BANDS)]
        st["chunk"] += 1
        wcost = []
        for k in self.bands:
            w = torch.softmax(
                hidden @ self.read_q[str(k)] / self.d ** 0.5, dim=1)
            read = torch.einsum("bt,btd->bd", w, hidden)
            st["acc"][k] = st["acc"][k] + read
            st["cnt"][k] += 1
            if st["chunk"] % HYBRID_CLOCKS[k] == 0:
                pooled = st["acc"][k] / max(st["cnt"][k], 1)
                if st["pend"][k] is not None:
                    fid = nn.functional.cosine_similarity(
                        st["pend"][k], pooled, dim=-1)
                    ticks[k].append((T - 1, fid))
                st["h"][k], z = self.cells[str(k)](pooled, st["h"][k])
                wcost.append(z)
                st["pend"][k] = self.pred[str(k)](st["h"][k])
                st["acc"][k] = torch.zeros_like(st["acc"][k])
                st["cnt"][k] = 0
        if wcost:
            self._write_cost = torch.stack(wcost).mean()
        return logits, st, ticks

    def pop_write_cost(self):
        c = self._write_cost
        self._write_cost = None
        return c

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
