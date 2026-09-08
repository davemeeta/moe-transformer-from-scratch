"""Plotting for one Explainer Agent run -- kept to matplotlib (already a
project dependency) since these are one-off static comparison charts, not
the interactive routing heatmap routing_analysis.py already produces."""

from __future__ import annotations

from pathlib import Path


def plot_attribution_bars(
    tokens: list[str], values: list[float], title: str, path: str | Path
) -> None:
    import matplotlib.pyplot as plt

    display_tokens = [tok.replace("\n", "\\n") or "·" for tok in tokens]
    colors = ["tab:blue" if v >= 0 else "tab:red" for v in values]

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(tokens)), 4))
    ax.bar(range(len(tokens)), values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(display_tokens, rotation=45, ha="right")
    ax.set_ylabel("attribution")
    ax.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
