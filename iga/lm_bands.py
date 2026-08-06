"""v5.0 band model — six clocked recurrent bands, no attention anywhere.

Band k ticks every CLOCKS[k] tokens (~8x ladder). Band 1 ticks per
token over the embedding; band k>=2 consumes the mean of band k-1's
states over its window, conditioned top-down by band k+1's current
state. The LM head reads the concatenation of all band states — the
only route to the long range is band state, which is the point.

Per-band predictors P_k (k>=2) predict the NEXT window-mean of band
k-1 at each tick: the v0.5-v0.9 recipe transposed. They are trained
(fidelity loss), serve as the drive layer's fidelity channels, and are
the forward model the imagination gate scores with.

Scene masking (conveyor law A1): at a scene start, fast bands 1-4 are
zeroed for that lane; bands 5-6 persist. Lesion: zeroed bands at eval.
"""

import torch
import torch.nn as nn

CLOCKS = [1, 8, 64, 512, 4096, 32768]
N_BANDS = len(CLOCKS)
FAST_BANDS = (0, 1, 2, 3)  # masked at scene starts; 5th/6th persist


class BandLM(nn.Module):
    def __init__(self, vocab_size, d=128, talk="dense"):
        """talk='dense': every band's held state is projected into band
        1's input at EVERY token — the full council speaks into each
        word. Write clocks are untouched (reads don't overwrite; the
        rare write remains the memory mechanism). talk='tick': the
        original chain — band 1 hears band 2 only. The debug tier A/Bs
        the two; the winner is frozen before registered runs (A9)."""
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.talk_mode = talk
        self.embed = nn.Embedding(vocab_size, d)
        self.cells = nn.ModuleList([nn.GRUCell(d, d) for _ in range(N_BANDS)])
        self.topdown = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for _ in range(N_BANDS - 1)])
        if talk == "dense":
            self.talk = nn.Linear(N_BANDS * d, d, bias=False)
        self.pred = nn.ModuleList(
            [nn.Linear(d, d) for _ in range(N_BANDS)])  # pred[k] used for k>=1
        self.head = nn.Sequential(
            nn.Linear(N_BANDS * d, 2 * d), nn.GELU(), nn.Linear(2 * d, vocab_size))
        self.lesioned = set()

    def init_state(self, B, device):
        return {
            "h": [torch.zeros(B, self.d, device=device) for _ in range(N_BANDS)],
            "acc": [torch.zeros(B, self.d, device=device) for _ in range(N_BANDS)],
            "cnt": [0 for _ in range(N_BANDS)],
            "pend": [None for _ in range(N_BANDS)],
            "step": 0,
        }

    def detach_state(self, st):
        st["h"] = [h.detach() for h in st["h"]]
        st["acc"] = [a.detach() for a in st["acc"]]
        # pend survives the chunk boundary detached: slow-band fidelity is
        # still MEASURED across chunks (channels, imagination, maintain all
        # see it); its gradient only flows when both ticks share a chunk
        st["pend"] = [p.detach() if p is not None else None
                      for p in st["pend"]]
        return st

    def _read(self, st):
        hs = []
        for k in range(N_BANDS):
            h = st["h"][k]
            hs.append(torch.zeros_like(h) if k in self.lesioned else h)
        return hs

    def forward(self, tokens, st, scene_starts=None):
        """tokens [B,T]; scene_starts: {local_pos: [lane indices]}.
        Returns logits [B,T,V], st, ticks: per-band list of
        (local_pos, fidelity[B]) fidelity entries (differentiable)."""
        B, T = tokens.shape
        logits = []
        ticks = [[] for _ in range(N_BANDS)]
        for t in range(T):
            if scene_starts and t in scene_starts:
                lanes = scene_starts[t]
                for k in FAST_BANDS:
                    st["h"][k] = st["h"][k].clone()
                    st["h"][k][lanes] = 0.0
                    st["acc"][k] = st["acc"][k].clone()
                    st["acc"][k][lanes] = 0.0
                    st["pend"][k] = None
            x = self.embed(tokens[:, t])
            hs = self._read(st)
            # band 1 every token — dense talk: the whole ladder speaks
            # into every word; tick mode: band 2 only
            if self.talk_mode == "dense":
                inp0 = x + self.talk(torch.cat(hs, dim=-1))
            else:
                inp0 = x + self.topdown[0](hs[1])
            st["h"][0] = self.cells[0](inp0, st["h"][0])
            st["acc"][0] = st["acc"][0] + st["h"][0]
            st["cnt"][0] += 1
            for k in range(1, N_BANDS):
                if (st["step"] + 1) % CLOCKS[k] == 0:
                    window = st["acc"][k - 1] / max(st["cnt"][k - 1], 1)
                    if st["pend"][k] is not None:
                        fid = nn.functional.cosine_similarity(
                            st["pend"][k], window, dim=-1)
                        ticks[k].append((t, fid))
                    hs_k = self._read(st)
                    td = self.topdown[k](hs_k[k + 1]) if k + 1 < N_BANDS else 0.0
                    st["h"][k] = self.cells[k](window + td, st["h"][k])
                    st["pend"][k] = self.pred[k](st["h"][k])
                    st["acc"][k - 1] = torch.zeros_like(st["acc"][k - 1])
                    st["cnt"][k - 1] = 0
                    if k < N_BANDS:
                        st["acc"][k] = st["acc"][k] + st["h"][k]
                        st["cnt"][k] += 1
            st["step"] += 1
            logits.append(self.head(torch.cat(self._read(st), dim=-1)))
        return torch.stack(logits, dim=1), st, ticks

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
