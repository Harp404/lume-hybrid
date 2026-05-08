"""
plot_results.py - Generate plots from ablation history JSONs.

Reads ablation/history/*.json and writes PNG figures to ablation/figures/
for use in README, papers, and writeups.

Usage (from repo root):
    python ablation/plot_results.py

Output:
    ablation/figures/val_loss_vs_tokens.png   - main figure
    ablation/figures/train_val_curves.png     - overfitting check
    ablation/figures/final_metrics_bars.png   - summary bars
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
HISTORY_DIR = HERE / "history"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

ARCHS = ["transformer", "mamba2", "qwen36", "lume_hybrid"]
LABELS = {
    "transformer": "Transformer (LLaMA3-style)",
    "mamba2":      "Mamba2 (pure)",
    "qwen36":      "Qwen3.6-style hybrid",
    "lume_hybrid": "Lume-Hybrid (ours)",
}
COLORS = {
    "transformer": "#888888",
    "mamba2":      "#5599cc",
    "qwen36":      "#cc8855",
    "lume_hybrid": "#cc3344",
}
LINEWIDTHS = {
    "transformer": 1.6,
    "mamba2":      1.6,
    "qwen36":      1.6,
    "lume_hybrid": 2.4,
}

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "figure.dpi":       110,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
})


def load_all():
    data = {}
    for a in ARCHS:
        p = HISTORY_DIR / f"{a}_history.json"
        if not p.exists():
            print(f"[warn] missing: {p}")
            continue
        with open(p) as f:
            data[a] = json.load(f)
    return data


def plot_val_loss_vs_tokens(data):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for a in ARCHS:
        if a not in data:
            continue
        h = data[a]["history"]
        tokens = np.array([p["tokens"] for p in h]) / 1e6
        val_loss = np.array([p["val_loss"] for p in h])
        ax.plot(tokens, val_loss, label=LABELS[a],
                color=COLORS[a], linewidth=LINEWIDTHS[a])
    ax.set_xlabel("Training tokens (millions)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss vs training tokens - controlled architecture ablation")
    ax.legend(loc="upper right", framealpha=0.95)
    for a in ARCHS:
        if a not in data:
            continue
        h = data[a]["history"]
        x = h[-1]["tokens"] / 1e6
        y = h[-1]["val_loss"]
        ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(4, 0),
                    textcoords="offset points", fontsize=8,
                    color=COLORS[a], va="center")
    out = FIG_DIR / "val_loss_vs_tokens.png"
    plt.savefig(out)
    plt.close()
    print(f"[ok] {out}")


def plot_train_val_curves(data):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, a in zip(axes, ARCHS):
        if a not in data:
            ax.set_visible(False)
            continue
        h = data[a]["history"]
        tokens = np.array([p["tokens"] for p in h]) / 1e6
        train = np.array([p["train_loss"] for p in h])
        val = np.array([p["val_loss"] for p in h])
        ax.plot(tokens, train, label="train", color=COLORS[a], linewidth=1.5, alpha=0.5)
        ax.plot(tokens, val,   label="val",   color=COLORS[a], linewidth=2.0)
        ax.set_title(LABELS[a])
        ax.legend(loc="upper right", fontsize=8)
    for ax in axes[-2:]:
        ax.set_xlabel("Training tokens (M)")
    for ax in axes[::2]:
        ax.set_ylabel("Loss")
    fig.suptitle("Train vs validation loss (overfitting diagnostic)", y=1.00)
    out = FIG_DIR / "train_val_curves.png"
    plt.savefig(out)
    plt.close()
    print(f"[ok] {out}")


def plot_final_bars(data):
    metrics = [
        ("final_val_loss",   "Val Loss",         "lower is better"),
        ("bpb",              "Bits per Byte",    "lower is better"),
        ("english_ppl",      "English PPL",      "lower is better"),
        ("repetition_rate",  "Repetition Rate",  "lower is better"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for ax, (key, title, sub) in zip(axes, metrics):
        vals = []
        names = []
        colors = []
        for a in ARCHS:
            if a not in data:
                continue
            vals.append(data[a][key])
            names.append(LABELS[a].split(" ")[0])
            colors.append(COLORS[a])
        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
        for i, a in enumerate([a for a in ARCHS if a in data]):
            if a == "lume_hybrid":
                bars[i].set_edgecolor("#aa0000")
                bars[i].set_linewidth(1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{title}\n({sub})", fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Final metrics - all four architectures, 100M tokens", y=1.02)
    out = FIG_DIR / "final_metrics_bars.png"
    plt.savefig(out)
    plt.close()
    print(f"[ok] {out}")


def main():
    data = load_all()
    if not data:
        print("[error] no history JSONs found in", HISTORY_DIR)
        return
    print(f"[info] loaded {len(data)} archs: {list(data.keys())}")
    plot_val_loss_vs_tokens(data)
    plot_train_val_curves(data)
    plot_final_bars(data)
    print(f"\n[done] figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
