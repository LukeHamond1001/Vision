"""Demo-video renderer for the v0.9 Crafter campaign (local tool, no pod).

Acts (storyboard fixed in the plan):
  1 gauges   — gameplay + live three-band latent, slow band vs daylight
  2 meters   — mid band vs ground-truth food/drink (the 0.99 routing, live)
  3 register — wiring-on agent: φ (register distance) + meters, eat/drink
               events visible as φ drops
  4 ablation — same world, wiring-on vs wiring-off side by side
  5 matrix   — routing-matrix card: "the encoder was never told what
               daylight is"

Usage:
    python -m iga.render_demo smoke     # acts 1-2 with a random policy
    python -m iga.render_demo full      # all acts from phase-1/2 artifacts
"""

from __future__ import annotations

import sys

import numpy as np
import torch

FPS = 20
GAME = 320          # upscaled game panel (64 -> 320, nearest)
PANEL_W = 420       # trace panel width
H = GAME


class FFmpegWriter:
    """Raw-RGB pipe into the system ffmpeg (no imageio backend needed)."""

    def __init__(self, path: str, fps: int = FPS):
        import subprocess
        self.path = path
        self.proc = None
        self.fps = fps
        self._subprocess = subprocess

    def append_data(self, frame: np.ndarray) -> None:
        if self.proc is None:
            h, w = frame.shape[:2]
            self.proc = self._subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                 "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.fps),
                 "-i", "-", "-c:v", "libx264", "-preset", "medium",
                 "-crf", "20", "-pix_fmt", "yuv420p", self.path],
                stdin=self._subprocess.PIPE)
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()


def _upscale(frame_u8: np.ndarray, size: int = GAME) -> np.ndarray:
    k = size // frame_u8.shape[0]
    return np.repeat(np.repeat(frame_u8, k, 0), k, 1)


def _trace_panel(series: list[tuple[str, np.ndarray, str]], t: int,
                 width: int = PANEL_W, height: int = H,
                 window: int = 300) -> np.ndarray:
    """Rolling traces via matplotlib, rendered to an RGB array.
    series: list of (label, values-up-to-now, color)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(width / 100, height / 100),
                             dpi=100, squeeze=False)
    lo = max(0, t - window)
    for ax, (label, vals, color) in zip(axes[:, 0], series):
        v = vals[lo:t + 1]
        ax.plot(np.arange(lo, t + 1), v, color=color, lw=1.6)
        ax.set_xlim(lo, max(t + 1, lo + 50))
        ax.set_title(label, fontsize=7, loc="left", pad=2)
        ax.tick_params(labelsize=5, length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    if buf.shape[0] != height or buf.shape[1] != width:
        from PIL import Image
        buf = np.array(Image.fromarray(buf).resize((width, height)))
    return buf


def render_act(rec: dict, path: str, panels: str, title: str = "") -> None:
    """panels: 'gauges' (z bands + daylight), 'meters' (mid vs truth),
    'register' (phi + meters)."""
    z = torch.stack(rec["z"]) if isinstance(rec["z"], list) else rec["z"]
    z = z.numpy()
    n = len(rec["food"])
    day = np.array(rec["daylight"]); food = np.array(rec["food"])
    drink = np.array(rec["drink"]); phi = np.array(rec["phi"][1:])
    wr = FFmpegWriter(path)
    for t in range(n):
        game = _upscale(rec["frames"][t + 1])
        if panels == "gauges":
            series = [("slow band (2 dims) — never told what daylight is",
                       z[1:t + 2, 6:8], "tab:blue"),
                      ("daylight (ground truth, eval-only)", day[:t + 1], "tab:orange"),
                      ("mid band (2 dims)", z[1:t + 2, 4:6], "tab:green"),
                      ("fast band (4 dims)", z[1:t + 2, 0:4], "tab:gray")]
        elif panels == "meters":
            series = [("mid dim 0", z[1:t + 2, 4], "tab:green"),
                      ("mid dim 1", z[1:t + 2, 5], "tab:olive"),
                      ("food (truth)", food[:t + 1], "tab:red"),
                      ("drink (truth)", drink[:t + 1], "tab:blue")]
        else:  # register
            series = [("register distance phi", phi[:t + 1], "tab:purple"),
                      ("food (truth)", food[:t + 1], "tab:red"),
                      ("drink (truth)", drink[:t + 1], "tab:blue"),
                      ("daylight", day[:t + 1], "tab:orange")]
        panel = _trace_panel(series, t)
        row = np.concatenate([game, panel], axis=1)
        wr.append_data(row)
    wr.close()
    print(f"[render] {path}: {n} steps, panels={panels}", flush=True)


def render_matrix_card(routing_json: str, path: str, hold_s: float = 5.0) -> None:
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(routing_json))
    bands = ["fast", "mid", "slow"]
    vars_ = ["daylight", "food", "drink", "energy", "health"]
    M = np.array([[d[f"{b}_{v}"] for v in vars_] for b in bands])
    fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=100)
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(5), vars_, fontsize=9)
    ax.set_yticks(range(3), bands, fontsize=9)
    for i in range(3):
        for j in range(5):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=9)
    ax.set_title("routing (max |corr|) — encoder trained with NO variable labels",
                 fontsize=10)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    if img.shape[0] % 2:
        img = img[:-1]
    if img.shape[1] % 2:
        img = img[:, :-1]
    wr = FFmpegWriter(path)
    for _ in range(int(hold_s * FPS)):
        wr.append_data(img)
    wr.close()
    print(f"[render] {path}: matrix card", flush=True)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    from .crafter_agent import CrafterPolicy, record_rollout
    from .crafter_support import EncoderCrafter
    from .experiments import RESULTS
    out = RESULTS / "video"
    out.mkdir(exist_ok=True, parents=True)
    enc = EncoderCrafter((4, 2, 2), seed=0)
    enc.load_state_dict(torch.load(RESULTS / "v09_encoder.pt", map_location="cpu"))
    enc.freeze()
    mid = slice(4, 6)
    if mode == "smoke":
        pol = CrafterPolicy(seed=0)   # untrained policy: mechanics test only
        rec = record_rollout(enc, pol, torch.zeros(2), mid, seed=3, max_steps=240)
        render_act(rec, str(out / "act1_gauges.mp4"), "gauges")
        render_act(rec, str(out / "act2_meters.mp4"), "meters")
        render_matrix_card(str(RESULTS / "v09_routing.json"),
                           str(out / "act5_matrix.mp4"))
        return
    # full: needs phase-2 artifacts (v10_policy_s*_on.pt + summary)
    g = torch.tensor(__import__("json").load(
        open(RESULTS / "v10_summary.json"))["register_target"])
    pol = CrafterPolicy(seed=1)
    pol.load_state_dict(torch.load(RESULTS / "v10_policy_s1_on.pt",
                                   map_location="cpu"))
    rec = record_rollout(enc, pol, g, mid, seed=3, max_steps=1000)
    render_act(rec, str(out / "act1_gauges.mp4"), "gauges")
    render_act(rec, str(out / "act2_meters.mp4"), "meters")
    render_act(rec, str(out / "act3_register.mp4"), "register")
    pol_off = CrafterPolicy(seed=1)
    pol_off.load_state_dict(torch.load(RESULTS / "v10_policy_s1_off.pt",
                                       map_location="cpu"))
    rec_off = record_rollout(enc, pol_off, g, mid, seed=3, max_steps=1000)
    render_act(rec_off, str(out / "act4_wiring_off.mp4"), "register")
    render_matrix_card(str(RESULTS / "v09_routing.json"),
                       str(out / "act5_matrix.mp4"))


if __name__ == "__main__":
    main()
