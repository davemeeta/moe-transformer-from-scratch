"""Batch evaluation: runs the full Orchestrator (Explainer -> Faithfulness
-> Red-Team -> Judge -> Report) over several real prompts sampled from this
project's own training corpus (data/raw/tinyshakespeare.txt), then
aggregates per-explainer faithfulness/red-team/judge stats into the
results table this project's README uses.

For a MoE checkpoint, also reports the router's own expert-collapse score
(reusing routing_analysis.py's utilities) alongside the aggregate
divergence rate -- shown side by side as context, not as a statistical
correlation: a handful of prompts is nowhere near enough to fit one
honestly.

Usage:
    python -m moe_transformer.xai.run_batch_eval xai.checkpoint=checkpoints/moe_compare_v2/step_000600
    python -m moe_transformer.xai.run_batch_eval xai.checkpoint=... xai.batch_num_prompts=10
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from moe_transformer.data import Tokenizer, TokenDataset
from moe_transformer.routing_analysis import (
    accumulate_layer_counts,
    collapse_score,
    utilization_from_counts,
)
from moe_transformer.train import get_batch, resolve_device
from moe_transformer.xai.orchestrator import Orchestrator
from moe_transformer.xai.run_explainer import build_model, resolve_model_kind_and_config

CORPUS_PATH = "data/raw/tinyshakespeare.txt"


def sample_prompts(corpus_path: str, num_prompts: int, min_len: int = 20, max_len: int = 100) -> list[str]:
    lines = Path(corpus_path).read_text().splitlines()
    candidates = [line.strip() for line in lines if min_len <= len(line.strip()) <= max_len]
    # dedupe while preserving order, so a repeated line (e.g. "Speak, speak.")
    # doesn't eat multiple slots in a short sample
    seen: set[str] = set()
    unique = []
    for line in candidates:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return unique[:num_prompts]


def aggregate(all_results: list[dict]) -> dict:
    explainers = ("attention", "shap", "lime")
    agg: dict = {}
    for name in explainers:
        faithfulness = [r["faithfulness"][name] for r in all_results]
        redteam = [r["redteam"][name] for r in all_results]
        judge = [r["judge"]["verdicts"][name] for r in all_results]
        n = len(all_results)
        agg[name] = {
            "mean_deletion_auc": sum(f["deletion_auc"] for f in faithfulness) / n,
            "mean_insertion_auc": sum(f["insertion_auc"] for f in faithfulness) / n,
            "mean_comprehensiveness": sum(f["comprehensiveness"] for f in faithfulness) / n,
            "mean_sufficiency": sum(f["sufficiency"] for f in faithfulness) / n,
            "mean_edit_fraction": sum(rt.edit_fraction for rt in redteam) / n,
            "flip_rate": sum(1 for rt in redteam if rt.flipped) / n,
            "mean_plausibility": sum(v["plausibility_score"] for v in judge) / n,
            "divergence_rate": sum(1 for v in judge if v["divergent"]) / n,
        }
    return agg


def render_aggregate_markdown(prompts: list[str], agg: dict, collapse: dict | None) -> str:
    lines = ["# RouteLens batch evaluation", ""]
    lines.append(f"**Prompts evaluated:** {len(prompts)} (sampled from `{CORPUS_PATH}`)")
    lines.append("")
    for p in prompts:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append(
        "| explainer | mean del_auc | mean ins_auc | mean comp | mean suff | "
        "mean edit_frac | flip rate | mean plausibility | divergence rate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, s in agg.items():
        lines.append(
            f"| {name} | {s['mean_deletion_auc']:.3g} | {s['mean_insertion_auc']:.3g} | "
            f"{s['mean_comprehensiveness']:.3g} | {s['mean_sufficiency']:.3g} | "
            f"{s['mean_edit_fraction']:.3g} | {s['flip_rate']:.2f} | "
            f"{s['mean_plausibility']:.2f} | {s['divergence_rate']:.2f} |"
        )
    lines.append("")
    if collapse is not None:
        lines.append("## Router context (not a statistical correlation -- too few prompts to fit one honestly)")
        lines.append("")
        lines.append(f"Expert-collapse score per layer (0 = balanced, 1 = collapsed onto one expert): {collapse['scores']}")
        lines.append("")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    kind, model_config = resolve_model_kind_and_config(cfg)
    device = resolve_device(cfg.training.device, kind)
    print(f"device: {device}  kind: {kind}")

    model = build_model(kind, model_config, device, cfg.xai.checkpoint)
    tokenizer = Tokenizer()

    prompts = sample_prompts(CORPUS_PATH, cfg.xai.batch_num_prompts)
    print(f"sampled {len(prompts)} prompts from {CORPUS_PATH}")

    orchestrator = Orchestrator(
        model,
        tokenizer,
        device,
        model_config.block_size,
        shap_max_evals=cfg.xai.shap_max_evals,
        lime_num_samples=cfg.xai.lime_num_samples,
        comprehensiveness_k=cfg.xai.comprehensiveness_k,
        redteam_protected_k=cfg.xai.redteam_protected_k,
        redteam_top_n_neighbors=cfg.xai.redteam_top_n_neighbors,
        redteam_max_edits=cfg.xai.redteam_max_edits,
        redteam_flip_threshold=cfg.xai.redteam_flip_threshold,
        deep_max_edits=cfg.xai.redteam_deep_max_edits,
        deep_top_n_neighbors=cfg.xai.redteam_deep_top_n_neighbors,
        judge_model=cfg.xai.judge_model,
    )

    all_results = []
    for i, prompt in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] {prompt!r}")
        all_results.append(orchestrator.run(prompt))

    agg = aggregate(all_results)
    print("\naggregate results:")
    for name, s in agg.items():
        print(f"  {name}: {s}")

    collapse = None
    if kind == "moe":
        val_ds = TokenDataset(cfg.data.val_bin, block_size=model_config.block_size)
        with torch.no_grad():
            x, _ = get_batch(val_ds, cfg.analysis.batch_size, device)
            _, _, aux = model(x, return_router_outputs=True)
        counts, total_tokens, top_k = accumulate_layer_counts([aux["router_outputs"]])
        f_i = utilization_from_counts(counts, total_tokens, top_k)
        scores = collapse_score(f_i)
        collapse = {"scores": [round(s, 3) for s in scores.tolist()]}
        print(f"\nrouter collapse score per layer: {collapse['scores']}")

    out_dir = Path(cfg.xai.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_md = render_aggregate_markdown(prompts, agg, collapse)
    report_path = out_dir / "batch_report.md"
    report_path.write_text(report_md)
    print(f"\nwrote {report_path}")

    summary_path = out_dir / "batch_summary.json"
    summary_path.write_text(json.dumps({"prompts": prompts, "aggregate": agg, "collapse": collapse}, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
