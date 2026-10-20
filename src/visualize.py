"""Visualiza os spikes da SNN processando um caractere -> GIF animado.

Uso:
    python -m src.visualize demo/hello.png            # 1o caractere da imagem
    python -m src.visualize demo/hello.png --char 3   # caractere especifico
    python -m src.visualize --synthetic K             # gera caractere sintetico
    python -m src.visualize --synthetic K --steps 32  # mais timesteps = mais fluido
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageOps

from .infer import OCREngine, binarize, segment, to_28
from .dataset import render_char
from .snn import CHARSET

BG = "#05070d"
SPIKE = "#00e5ff"
GREEN = "#00e676"
NEG = "#30363d"
GLOW = LinearSegmentedColormap.from_list("glow", [BG, "#003c4d", SPIKE, "white"])
PALETTE = ["#00e5ff", "#ff9100", "#ea80fc", "#76ff03", "#ff5252"]


def get_char_tensor(engine, image_path=None, char_idx=0, synthetic=None):
    if synthetic:
        return to_28(np.asarray(ImageOps.grayscale(render_char(synthetic)).copy()) > 128)
    img = ImageOps.grayscale(Image.open(image_path))
    crops = sorted(segment(binarize(img)))
    if not crops:
        raise SystemExit("nenhum caractere segmentado na imagem")
    char_idx = min(char_idx, len(crops) - 1)
    return to_28(crops[char_idx][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--char", type=int, default=0)
    ap.add_argument("--synthetic", type=str, default=None)
    ap.add_argument("--steps", type=int, default=24, help="timesteps da simulacao")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="demo/spikes.gif")
    args = ap.parse_args()

    engine = OCREngine()
    engine.model.num_steps = args.steps
    x = get_char_tensor(engine, args.image, args.char, args.synthetic).unsqueeze(0)

    with torch.no_grad():
        logits, rec = engine.model(x.to(engine.device), record=True)
    probs = torch.softmax(logits, 1)[0].cpu()
    pred = int(logits.argmax(1))
    total_spikes = int(sum(torch.stack(rec[k]).sum() for k in ("s1", "s2", "s3")))

    T = args.steps
    s1 = torch.stack(rec["s1"])[:, 0].cpu().numpy()   # T,16,28,28
    s3 = torch.stack(rec["s3"])[:, 0].cpu().numpy()   # T,256
    m4 = torch.stack(rec["m4"])[:, 0].cpu().numpy()   # T,36

    # top-5 classes por potencial final -> trajetorias
    top5 = np.argsort(m4[-1])[-5:][::-1]
    m4max = np.abs(m4).max() + 1e-9
    m4n = m4 / m4max

    fig = plt.figure(figsize=(14, 7), facecolor=BG)
    gs = gridspec.GridSpec(4, 8, figure=fig, hspace=0.15, wspace=0.3,
                           left=0.03, right=0.97, top=0.90, bottom=0.09)

    # --- input + predicao ---
    ax_in = fig.add_subplot(gs[0:3, 0:2])
    ax_in.imshow(x[0, 0], cmap="gray", interpolation="nearest")
    ax_in.axis("off")
    ax_pred = fig.add_subplot(gs[3, 0:2])
    ax_pred.axis("off")
    pred_txt = ax_pred.text(0.5, 0.5, "?", ha="center", va="center",
                            fontsize=44, color=GREEN, weight="bold",
                            family="monospace")

    # --- conv1: grid 4x4 com rastro (glow decay) ---
    conv_ims = []
    for i in range(16):
        ax = fig.add_subplot(gs[i // 4, 2 + i % 4])
        ax.axis("off")
        conv_ims.append(ax.imshow(np.zeros((28, 28)), cmap=GLOW,
                                  vmin=0, vmax=1, interpolation="nearest"))
    fig.text(0.40, 0.955, "conv1 · 16 canais LIF (spikes com rastro)",
             ha="center", color="#8b949e", fontsize=11)

    # --- raster fc1 ---
    ax_r = fig.add_subplot(gs[0:2, 6:8])
    ax_r.set_facecolor(BG)
    ymax = max(int(np.nonzero(s3)[1].max()) + 20, 60) if s3.any() else 256
    ax_r.set_xlim(-0.5, T - 0.5); ax_r.set_ylim(0, ymax)
    ax_r.set_xticks([])
    ax_r.set_yticks([])
    for s in ax_r.spines.values():
        s.set_color("#21262d")
    ax_r.set_title("fc1 · raster de spikes", color="#8b949e", fontsize=11)

    # --- trajetorias do readout (top-5 classes) ---
    ax_tr = fig.add_subplot(gs[2:4, 6:8])
    ax_tr.set_facecolor(BG)
    ax_tr.set_xlim(0, T - 1); ax_tr.set_ylim(-1.1, 1.1)
    ax_tr.axhline(0, color="#21262d", lw=0.8)
    ax_tr.set_xticks(range(0, T, 4))
    ax_tr.set_xticklabels(range(1, T + 1, 4), color="#484f58", fontsize=8)
    ax_tr.set_yticks([])
    for s in ax_tr.spines.values():
        s.set_color("#21262d")
    ax_tr.set_title("readout · potencial de membrana (top-5)", color="#8b949e",
                    fontsize=11, pad=10)
    ax_tr.set_xlabel("timestep", color="#484f58", fontsize=9)

    lines, labels = [], []
    for k, cls in enumerate(top5):
        color = GREEN if cls == pred else PALETTE[k % len(PALETTE)]
        (ln,) = ax_tr.plot([], [], color=color, lw=2)
        lines.append(ln)
        labels.append(ax_tr.text(T - 0.9, float(m4n[-1, cls]), CHARSET[cls],
                                 color=color, fontsize=11, weight="bold", va="center"))

    clock = fig.text(0.06, 0.955, "", color="#8b949e", fontsize=11, family="monospace")
    trail = np.zeros((16, 28, 28))

    def frame(t):
        nonlocal trail
        t = min(t, T - 1)
        done = t == T - 1
        clock.set_text(f"t = {t+1:02d}/{T}   spikes = {int(s1[:t+1].sum()+s3[:t+1].sum()):,}")

        trail = trail * 0.55 + s1[t]  # rastro com decay -> efeito glow
        for i, im in enumerate(conv_ims):
            im.set_data(np.clip(trail[i], 0, 1))

        ys, xs = np.nonzero(s3[: t + 1])
        ax_r.scatter(xs, ys, s=10, c=SPIKE, marker="|", lw=1.0, alpha=0.85)

        for ln, cls in zip(lines, top5):
            ln.set_data(np.arange(t + 1), m4n[: t + 1, cls])
        pred_txt.set_text(CHARSET[pred] if done else "?")
        pred_txt.set_alpha(1.0 if done else min(0.2 + 0.05 * t, 0.95))

    anim = FuncAnimation(fig, frame, frames=T + 8, interval=1000 // args.fps)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"GIF salvo em {args.out} | pred='{CHARSET[pred]}' "
          f"conf={float(probs[pred]):.2f} | {total_spikes:,} spikes")


if __name__ == "__main__":
    main()
