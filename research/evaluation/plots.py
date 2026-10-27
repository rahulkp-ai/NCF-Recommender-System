"""
research/evaluation/plots.py

Publication-quality matplotlib figures for the thesis.
Includes evaluation metrics: Hit@K, NDCG@K, Precision@K, Recall@K.
All figures saved to evaluation/figures/ as both PNG (300 DPI) and PDF.
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import run_comparison directly to update comparison.json automatically
from research.evaluation.compare import run_comparison


matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "figure.dpi":        150,
})

LOG_DIR = Path(__file__).resolve().parent
FIG_DIR = LOG_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

CORAL  = "#D85A30"
TEAL   = "#1D9E75"
PURPLE = "#7A5195"
ORANGE = "#FFA600"


def _load():
    with open(LOG_DIR / "comparison.json") as f:
        return json.load(f)


# ── Figure 1: Loss convergence comparison ────────────────────────────────────

def plot_loss_curves(data: dict = None):
    if data is None:
        data = _load()

    epochs = data["epochs"]
    s_loss = data["scratch"]["loss"]
    p_loss = data["pytorch"]["loss"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, s_loss, color=CORAL, lw=2,
            label="Scratch NCF (NumPy)", zorder=3)
    ax.plot(epochs, p_loss, color=TEAL,  lw=2,
            label="PyTorch NCF (MPS)",  zorder=3)

    ax.axhline(0.693, color="gray", lw=0.8, ls="--", alpha=0.5,
               label="Random baseline (0.693)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Training Loss Convergence: Scratch vs PyTorch NCF")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0.28)

    # Annotate final loss delta
    delta = data["comparison"].get("final_loss_delta", 0)
    ax.annotate(
        f"Δ = {delta:.4f}",
        xy=(epochs[-1], (s_loss[-1] + p_loss[-1]) / 2),
        xytext=(epochs[-1] - 4, (s_loss[-1] + p_loss[-1]) / 2 + 0.015),
        fontsize=9, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig1_loss_curves.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {FIG_DIR}/fig1_loss_curves.png/pdf")
    return fig


# ── Figure 2: Evaluation Metrics Over Epochs (Hit, NDCG, Precision, Recall) ─

def plot_evaluation_curves(data: dict = None):
    if data is None:
        data = _load()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    metrics = [
        ("hit_at_10",       "Hit@10",       axes[0, 0]),
        ("ndcg_at_10",      "NDCG@10",      axes[0, 1]),
        ("precision_at_10", "Precision@10", axes[1, 0]),
        ("recall_at_10",    "Recall@10",    axes[1, 1]),
    ]

    for key, title, ax in metrics:
        s_pairs = data["scratch"].get(key, data["scratch"].get("hit_at_10", []))
        p_pairs = data["pytorch"].get(key, data["pytorch"].get("hit_at_10", []))

        s_ep, s_val = zip(*s_pairs) if s_pairs else ([], [])
        p_ep, p_val = zip(*p_pairs) if p_pairs else ([], [])

        if s_val:
            ax.plot(s_ep, s_val, color=CORAL, lw=2, marker="o", ms=4, label="Scratch NCF")
        if p_val:
            ax.plot(p_ep, p_val, color=TEAL,  lw=2, marker="s", ms=4, label="PyTorch NCF")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"Recommendation Quality: {title}")
        ax.legend(frameon=False)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig2_evaluation_curves.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {FIG_DIR}/fig2_evaluation_curves.png/pdf")
    return fig


# ── Figure 3: Evaluation Metrics Summary (Bar Chart Comparison) ────────────

def plot_eval_metrics_bar(data: dict = None):
    if data is None:
        data = _load()

    labels = ["Hit@10", "NDCG@10", "Precision@10", "Recall@10"]
    
    s_scores = [
        data["scratch"].get("best_hit", 0),
        data["scratch"].get("best_ndcg", 0),
        data["scratch"].get("best_precision", 0),
        data["scratch"].get("best_recall", 0),
    ]
    p_scores = [
        data["pytorch"].get("best_hit", 0),
        data["pytorch"].get("best_ndcg", 0),
        data["pytorch"].get("best_precision", 0),
        data["pytorch"].get("best_recall", 0),
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rects1 = ax.bar(x - width/2, s_scores, width, label="Scratch NCF", color=CORAL, zorder=3)
    rects2 = ax.bar(x + width/2, p_scores, width, label="PyTorch NCF", color=TEAL, zorder=3)

    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics Comparison: Hit@10, NDCG@10, Precision@10, Recall@10")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)

    for rect in rects1 + rects2:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f"{height:.4f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig3_metrics_bar_comparison.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {FIG_DIR}/fig3_metrics_bar_comparison.png/pdf")
    return fig


# ── Figure 4: Training time comparison ───────────────────────────────────────

def plot_time_comparison(data: dict = None):
    if data is None:
        data = _load()

    s_total = data["scratch"]["total_time"]
    p_total = data["pytorch"]["total_time"]
    speedup = data["comparison"]["total_speedup"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    bars = ax.bar(
        ["Scratch NCF\n(NumPy)", "PyTorch NCF\n(MPS)"],
        [s_total, p_total],
        color=[CORAL, TEAL], width=0.5, zorder=3
    )
    ax.set_ylabel("Total training time (seconds)")
    ax.set_title(f"Training Efficiency ({speedup}× speedup)")
    for bar, val in zip(bars, [s_total, p_total]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{val:.0f}s", ha="center", fontsize=11, fontweight="bold")

    ax2 = axes[1]
    epochs = data["epochs"]
    ax2.plot(epochs, data["scratch"]["time_per_epoch"],
             color=CORAL, lw=1.5, label="Scratch")
    ax2.plot(epochs, data["pytorch"]["time_per_epoch"],
             color=TEAL,  lw=1.5, label="PyTorch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Per-Epoch Training Time")
    ax2.legend(frameon=False)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig4_time_comparison.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {FIG_DIR}/fig4_time_comparison.png/pdf")
    return fig


# ── Figure 5: Four-panel summary (thesis page) ───────────────────────────────

def plot_thesis_summary(data: dict = None):
    """Single figure with all four panels — key thesis results page."""
    if data is None:
        data = _load()

    fig = plt.figure(figsize=(12, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    epochs = data["epochs"]
    s_loss = data["scratch"]["loss"]
    p_loss = data["pytorch"]["loss"]

    # Panel A: Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, s_loss, color=CORAL, lw=2, label="Scratch")
    ax1.plot(epochs, p_loss, color=TEAL,  lw=2, label="PyTorch")
    ax1.axhline(0.693, color="gray", lw=0.8, ls="--", alpha=0.4)
    ax1.set_title("(A) Loss convergence")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend(frameon=False, fontsize=9)

    # Panel B: Ranking Quality (Hit@10)
    ax2 = fig.add_subplot(gs[0, 1])
    for key, col, mk in [("scratch", CORAL, "o"), ("pytorch", TEAL, "s")]:
        pairs = data[key].get("hit_at_10", [])
        if pairs:
            ep, hit = zip(*pairs)
            ax2.plot(ep, hit, color=col, lw=2, marker=mk, ms=4,
                     label=f"{key.capitalize()} Hit@10")
    ax2.set_title("(B) Hit@10 Ranking Quality")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Hit@10 Score")
    ax2.legend(frameon=False, fontsize=9)

    # Panel C: Speedup per epoch
    ax3 = fig.add_subplot(gs[1, 0])
    speedup = [x for x in data["comparison"]["speedup_per_epoch"] if x]
    ax3.bar(range(1, len(speedup)+1), speedup,
            color=TEAL, alpha=0.8, zorder=3)
    avg = data["comparison"]["avg_speedup"]
    ax3.axhline(avg, color=CORAL, lw=1.5, ls="--",
                label=f"Mean {avg}×")
    ax3.set_title("(C) Per-epoch speedup")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Speedup (×)")
    ax3.legend(frameon=False, fontsize=9)

    # Panel D: Summary evaluation metrics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    c = data["comparison"]
    rows = [
        ["Metric",          "Scratch",                        "PyTorch",                        "Delta"],
        ["Final loss",      str(data["scratch"]["loss"][-1]), str(data["pytorch"]["loss"][-1]), str(c.get("final_loss_delta", "—"))],
        ["Best Hit@10",     str(data["scratch"].get("best_hit", "—")), str(data["pytorch"].get("best_hit", "—")), str(c.get("best_hit_delta", "—"))],
        ["Best NDCG@10",    str(data["scratch"].get("best_ndcg", "—")), str(data["pytorch"].get("best_ndcg", "—")), "—"],
        ["Precision@10",   str(data["scratch"].get("best_precision", "—")), str(data["pytorch"].get("best_precision", "—")), "—"],
        ["Recall@10",      str(data["scratch"].get("best_recall", "—")), str(data["pytorch"].get("best_recall", "—")), "—"],
        ["Total time (s)",  str(data["scratch"]["total_time"]), str(data["pytorch"]["total_time"]), f"{c.get('total_speedup', '—')}×"],
    ]
    tbl = ax4.table(cellText=rows[1:], colLabels=rows[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax4.set_title("(D) Summary evaluation comparison", pad=14)

    fig.suptitle(
        "NCF Benchmark: Scratch (NumPy) vs PyTorch Implementation",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"fig5_thesis_summary.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {FIG_DIR}/fig5_thesis_summary.png/pdf")
    return fig


def generate_all_figures():
    data = _load()
    plot_loss_curves(data)
    plot_evaluation_curves(data)
    plot_eval_metrics_bar(data)
    plot_time_comparison(data)
    plot_thesis_summary(data)
    print(f"\nAll figures saved in {FIG_DIR}/")


if __name__ == "__main__":
    # Ensure comparison.json is up-to-date before rendering plots
    run_comparison()
    generate_all_figures()