"""Routing analysis: expert utilization, collapse score, per-token routing.

All of this reads the same per-layer MoEOutput objects the model already
produces during a forward pass (via MoEGPT(..., return_router_outputs=True))
-- nothing here changes what the model computes, it only inspects it.
"""

from __future__ import annotations

import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from moe_transformer.checkpoint import load_checkpoint
from moe_transformer.data import Tokenizer, TokenDataset
from moe_transformer.model.moe import MoEOutput
from moe_transformer.models import MoEGPT
from moe_transformer.train import build_model_config, get_batch, resolve_device


def accumulate_layer_counts(
    batches_of_router_outputs: list[list[MoEOutput]],
) -> tuple[torch.Tensor, int, int]:
    """Sums expert_assignment_counts across multiple forward passes (batches).

    Returns (counts, total_tokens, top_k):
      - counts: (n_layer, num_experts) long tensor, summed across all batches.
      - total_tokens: total token count N summed across batches.
      - top_k: read off the first batch's routing (assumed constant across batches).
    """
    n_layer = len(batches_of_router_outputs[0])
    num_experts = batches_of_router_outputs[0][0].expert_assignment_counts.numel()
    total_counts = torch.zeros(n_layer, num_experts, dtype=torch.long)
    total_tokens = 0
    top_k = batches_of_router_outputs[0][0].topk_idx.shape[1]

    for router_outputs in batches_of_router_outputs:
        for layer_idx, moe_out in enumerate(router_outputs):
            total_counts[layer_idx] += moe_out.expert_assignment_counts
        total_tokens += router_outputs[0].topk_idx.shape[0]

    return total_counts, total_tokens, top_k


def utilization_from_counts(counts: torch.Tensor, total_tokens: int, top_k: int) -> torch.Tensor:
    """counts: (n_layer, num_experts) -> f_i per layer, each row summing to 1.

    Normalizes by total_tokens * top_k, not total_tokens: with top_k > 1 each
    token contributes top_k assignments spread across experts, so per-expert
    utilization sums to top_k across experts if normalized against N alone.
    """
    return counts.float() / (total_tokens * top_k)


def collapse_score(f_i: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy-based collapse score for one or more layers' utilization
    distributions f_i (..., num_experts), each row assumed to sum to ~1.

    0.0 = perfectly balanced (uniform over experts), 1.0 = fully collapsed
    onto a single expert. Defined as 1 - H(f) / H_max, where H is Shannon
    entropy and H_max = log(num_experts) is entropy at perfect balance.
    """
    num_experts = f_i.shape[-1]
    safe_log = f_i.clamp_min(eps).log()
    entropy = -(f_i * safe_log).sum(dim=-1)
    max_entropy = math.log(num_experts)
    return 1.0 - entropy / max_entropy


def per_token_routing_table(router_outputs: list[MoEOutput], token_strs: list[str]) -> dict:
    """For a single-sequence (batch=1) forward pass, builds a (n_layer,
    seq_len) top-1 expert-index table plus full top-k detail per token/layer
    for richer inspection (e.g. heatmap hover text).
    """
    n_layer = len(router_outputs)
    seq_len = len(token_strs)
    top1_experts = torch.zeros(n_layer, seq_len, dtype=torch.long)
    detail: list[list[list[tuple[int, float]]]] = []

    for layer_idx, moe_out in enumerate(router_outputs):
        topk_idx = moe_out.topk_idx  # (seq_len, k) -- batch=1, flattened
        topk_weights = moe_out.topk_weights
        assert topk_idx.shape[0] == seq_len, "router_outputs must come from a batch=1 forward pass"
        top1_experts[layer_idx] = topk_idx[:, 0]  # torch.topk sorts descending
        layer_detail = [
            list(zip(topk_idx[t].tolist(), topk_weights[t].tolist())) for t in range(seq_len)
        ]
        detail.append(layer_detail)

    return {"top1_experts": top1_experts, "detail": detail, "tokens": token_strs}


def plot_expert_utilization(f_i: torch.Tensor, scores: torch.Tensor, path: str | Path) -> None:
    """f_i: (n_layer, num_experts) utilization, scores: (n_layer,) collapse
    scores. Saves a grouped bar chart, one subplot per layer, as a PNG."""
    import matplotlib.pyplot as plt

    n_layer, num_experts = f_i.shape
    fig, axes = plt.subplots(1, n_layer, figsize=(3.5 * n_layer, 4), sharey=True, squeeze=False)
    axes = axes[0]
    uniform = 1.0 / num_experts

    for layer_idx, ax in enumerate(axes):
        ax.bar(range(num_experts), f_i[layer_idx].tolist(), color="steelblue")
        ax.axhline(uniform, color="gray", linestyle="--", linewidth=1, label="uniform")
        ax.set_title(f"layer {layer_idx}\ncollapse={scores[layer_idx].item():.3f}")
        ax.set_xlabel("expert")
        ax.set_xticks(range(num_experts))

    axes[0].set_ylabel("fraction of routing slots")
    axes[0].legend(fontsize=8)
    fig.suptitle("Expert utilization per layer (0 = balanced, 1 = collapsed)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_routing_heatmap(routing_table: dict, path: str | Path) -> None:
    """Interactive per-token routing heatmap (top-1 expert per token/layer,
    hover shows the full top-k breakdown). Saved as a standalone HTML file."""
    import plotly.graph_objects as go

    top1 = routing_table["top1_experts"]
    tokens = routing_table["tokens"]
    detail = routing_table["detail"]
    n_layer, seq_len = top1.shape

    display_tokens = [tok.replace("\n", "\\n") for tok in tokens]
    hover_text = [
        [
            "<br>".join(f"expert {e}: {w:.2f}" for e, w in detail[layer][t])
            for t in range(seq_len)
        ]
        for layer in range(n_layer)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=top1.tolist(),
            x=[f"{i}:{tok}" for i, tok in enumerate(display_tokens)],
            y=[f"layer {i}" for i in range(n_layer)],
            text=hover_text,
            hovertemplate="token=%{x}<br>%{y}<br>%{text}<extra></extra>",
            colorscale="Viridis",
            colorbar=dict(title="expert"),
        )
    )
    fig.update_layout(
        title="Per-token expert routing (top-1 shown; hover for full top-k)",
        xaxis_title="token",
        yaxis_title="layer",
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.model.kind != "moe":
        raise ValueError("routing_analysis requires model.kind=moe (dense models have no router)")

    model_config = build_model_config(cfg.model)
    device = resolve_device(cfg.training.device, cfg.model.kind)
    print(f"device: {device}")

    model = MoEGPT(model_config).to(device)
    if cfg.analysis.checkpoint:
        load_checkpoint(cfg.analysis.checkpoint, model, device=device)
        print(f"loaded checkpoint from {cfg.analysis.checkpoint}")
    else:
        print("no checkpoint given -- analyzing a freshly-initialized (untrained) model")
    model.eval()

    val_ds = TokenDataset(cfg.data.val_bin, block_size=model_config.block_size)

    batches_of_router_outputs = []
    with torch.no_grad():
        for _ in range(cfg.analysis.num_batches):
            x, _ = get_batch(val_ds, cfg.analysis.batch_size, device)
            _, _, aux = model(x, return_router_outputs=True)
            batches_of_router_outputs.append(aux["router_outputs"])

    counts, total_tokens, top_k = accumulate_layer_counts(batches_of_router_outputs)
    f_i = utilization_from_counts(counts, total_tokens, top_k)
    scores = collapse_score(f_i)

    print(f"\nexpert utilization over {total_tokens:,} tokens ({cfg.analysis.num_batches} batches):")
    for layer_idx in range(f_i.shape[0]):
        pct = "  ".join(f"e{e}={p:.1%}" for e, p in enumerate(f_i[layer_idx].tolist()))
        print(f"  layer {layer_idx}: {pct}   collapse_score={scores[layer_idx].item():.3f}")

    out_dir = Path(cfg.analysis.out_dir)
    utilization_path = out_dir / "expert_utilization.png"
    plot_expert_utilization(f_i, scores, utilization_path)
    print(f"\nwrote {utilization_path}")

    tokenizer = Tokenizer()
    token_ids = tokenizer.encode(cfg.analysis.sample_text)
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]
    idx = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        _, _, aux = model(idx, return_router_outputs=True)
    routing_table = per_token_routing_table(aux["router_outputs"], token_strs)

    heatmap_path = out_dir / "routing_heatmap.html"
    plot_routing_heatmap(routing_table, heatmap_path)
    print(f"wrote {heatmap_path}")


if __name__ == "__main__":
    main()
