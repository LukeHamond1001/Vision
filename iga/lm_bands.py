"""v5.0 band model — six clocked recurrent bands, no attention anywhere.

Band k ticks every CLOCKS[k] tokens (~8x ladder). Band 1 ticks per
token over the embedding; band k>=2 consumes the mean of band k-1's
states over its window (projected to its own width), conditioned
top-down by band k+1's current state. The LM head reads the
concatenation of all band states — the only route to the long range
is band state, which is the point.

Per-band predictors P_k (k>=2) predict the NEXT window-mean of band
k-1 at each tick: trained (fidelity loss), the drive layer's fidelity
channels, and the forward model the imagination gate scores with.

Talk modes (A9): "dense" projects every band's held state into band
1's input at EVERY token — reads only; write clocks untouched.
"tick" is the original chain (band 1 hears band 2 only).

Width shaping (A10): widths may differ per band. "slowheavy" gives
the slow bands more room — they carry the most compressed meaning.
The debug tier A/Bs talk x widths at matched params; winners are
frozen before registered runs.
"""

import torch
import torch.nn as nn

CLOCKS = [1, 8, 64, 512, 4096, 32768]
N_BANDS = len(CLOCKS)
FAST_BANDS = (0, 1, 2, 3)


def shape_widths(d, shape="uniform"):
    if shape == "uniform":
        return [d] * N_BANDS
    if shape == "slowheavy":
        return [d, d, d, int(1.5 * d), 2 * d, 2 * d]
    raise ValueError(shape)


class BandLM(nn.Module):
    def __init__(self, vocab_size, d=128, talk="dense", widths=None):
        super().__init__()
        widths = widths or shape_widths(d)
        self.widths = widths
        self.d = widths[0]
        self.vocab_size = vocab_size
        self.talk_mode = talk
        self.embed = nn.Embedding(vocab_size, widths[0])
        self.cells = nn.ModuleList(
            [nn.GRUCell(widths[k], widths[k]) for k in range(N_BANDS)])
        # in_proj[k]: band k-1's window mean -> band k's input space
        self.in_proj = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(widths[k - 1], widths[k])
                               for k in range(1, N_BANDS)])
        self.topdown = nn.ModuleList(
            [nn.Linear(widths[k + 1], widths[k], bias=False)
             for k in range(N_BANDS - 1)])
        # pred[k] (k>=1): band k's state -> next window mean of band k-1
        self.pred = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(widths[k], widths[k - 1])
                               for k in range(1, N_BANDS)])
        if talk == "dense":
            self.talk = nn.Linear(sum(widths), widths[0], bias=False)
        self.head = nn.Sequential(
            nn.Linear(sum(widths), 2 * widths[0]), nn.GELU(),
            nn.Linear(2 * widths[0], vocab_size))
        self.lesioned = set()

    def init_state(self, B, device):
        return {
            "h": [torch.zeros(B, w, device=device) for w in self.widths],
            "acc": [torch.zeros(B, w, device=device) for w in self.widths],
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
        B, T = tokens.shape
        logits = []
        ticks = [[] for _ in range(N_BANDS)]
        for t in range(T):
            x = self.embed(tokens[:, t])
            hs = self._read(st)
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
                    inp = self.in_proj[k](window)
                    if k + 1 < N_BANDS:
                        inp = inp + self.topdown[k](hs_k[k + 1])
                    st["h"][k] = self.cells[k](inp, st["h"][k])
                    st["pend"][k] = self.pred[k](st["h"][k])
                    st["acc"][k - 1] = torch.zeros_like(st["acc"][k - 1])
                    st["cnt"][k - 1] = 0
                    st["acc"][k] = st["acc"][k] + st["h"][k]
                    st["cnt"][k] += 1
            st["step"] += 1
            logits.append(self.head(torch.cat(self._read(st), dim=-1)))
        return torch.stack(logits, dim=1), st, ticks

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
