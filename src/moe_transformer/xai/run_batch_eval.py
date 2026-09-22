"""Batch evaluation: runs the full Orchestrator (Explainer -> Faithfulness
-> Red-Team -> Judge -> Report) over many real prompts sampled from this
project's own training corpus (data/raw/tinyshakespeare.txt), then
aggregates per-explainer faithfulness/red-team/judge stats with bootstrap
95% confidence intervals -- so a claim like "LIME never survives the
red-team attack" comes with an honest measure of how much it could move on
a different sample of prompts.

Everything stochastic is seeded from cfg.seed: which prompts get sampled,
LIME's perturbation sampling, the judge LLM (temperature 0 + Ollama seed),
and the bootstrap itself. Per-prompt results stream to
batch_per_prompt.jsonl as they finish, so a long run that dies at prompt 40
keeps the first 39.

For a MoE checkpoint, also reports the router's expert-collapse score
(reusing routing_analysis.py's utilities) as context.

Usage:
    python -m moe_transformer.xai.run_batch_eval xai.checkpoint=checkpoints/moe_compare_v2/step_000600
    python -m moe_transformer.xai.run_batch_eval xai.checkpoint=... xai.batch_num_prompts=20 xai.out_dir=outputs/xai/batch_small
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from moe_transformer.data import Tokenizer, TokenDataset
from moe_transformer.routing_analysis import (
    accumulate_layer_counts,
    collapse_score,
    utilization_from_counts,
)
from moe_transformer.train import get_batch, resolve_device
from moe_transformer.xai.agents.base import AgentState
from moe_transformer.xai.orchestrator import Orchestrator
from moe_transformer.xai.run_explainer import build_model, resolve_model_kind_and_config

CORPUS_PATH = "data/raw/tinyshakespeare.txt"
EXPLAINERS = ("attention", "shap", "lime")
CONTINUOUS_METRICS = (
    "deletion_auc",
    "insertion_auc",
    "comprehensiveness",
    "sufficiency",
    "edit_fraction",
)
BOOTSTRAP_RESAMPLES = 2000


def sample_prompts(
    corpus_path: str | Path, num_prompts: int, seed: int, min_len: int = 20, max_len: int = 100
) -> list[str]:
    lines = Path(corpus_path).read_text().splitlines()
    # dedupe (order-preserving, so the seeded sample is stable) -- a repeated
    # line like "Speak, speak." shouldn't be able to fill several slots
    unique = list(dict.fromkeys(
        line.strip() for line in lines if min_len <= len(line.strip()) <= max_len
    ))
    if num_prompts >= len(unique):
        return unique
    return random.Random(seed).sample(unique, num_prompts)


def summarize_state(prompt: str, state: AgentState) -> dict:
    """Compact, JSON-serializable per-prompt record -- everything aggregate()
    needs, without the raw attribution vectors."""
    explainers = {}
    for name in EXPLAINERS:
        f = state["faithfulness"][name]
        rt = state["redteam"][name]
        j = state["judge"]["verdicts"][name]
        explainers[name] = {
            "deletion_auc": f["deletion_auc"],
            "insertion_auc": f["insertion_auc"],
            "comprehensiveness": f["comprehensiveness"],
            "sufficiency": f["sufficiency"],
            "router_tau": f.get("router_attribution_agreement"),
            "edit_fraction": rt.edit_fraction,
            "flipped": rt.flipped,
            "plausibility": j["plausibility_score"],
            "judge_failed": j["judge_failed"],
            "divergent": j["divergent"],
        }
    return {
        "prompt": prompt,
        "target_token": state["target_token_str"],
        "deep_retry": state["redteam_deep_retry"],
        "explainers": explainers,
    }


def bootstrap_ci(
    values: list[float], rng: np.random.Generator, n_resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    resampled_means = rng.choice(arr, size=(n_resamples, len(arr)), replace=True).mean(axis=1)
    low, high = np.quantile(resampled_means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def summarize(values: list[float], rng: np.random.Generator) -> dict:
    arr = np.asarray(values, dtype=float)
    low, high = bootstrap_ci(arr, rng)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ci_low": low,
        "ci_high": high,
        "n": int(len(arr)),
    }


def aggregate(records: list[dict], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    agg: dict = {}
    for name in EXPLAINERS:
        rows = [r["explainers"][name] for r in records]
        judged = [row for row in rows if not row["judge_failed"]]
        entry = {"n_prompts": len(rows), "n_judge_failures": len(rows) - len(judged)}
        for metric in CONTINUOUS_METRICS:
            entry[metric] = summarize([row[metric] for row in rows], rng)
        entry["flip_rate"] = summarize([float(row["flipped"]) for row in rows], rng)
        if judged:
            entry["plausibility"] = summarize([row["plausibility"] for row in judged], rng)
            entry["divergence_rate"] = summarize([float(row["divergent"]) for row in judged], rng)
        taus = [row["router_tau"] for row in rows if row["router_tau"] is not None]
        if taus:
            entry["router_tau"] = summarize(taus, rng)
        agg[name] = entry
    return agg


def _fmt(summary: dict | None, digits: int = 2) -> str:
    if summary is None:
        return "--"
    return f"{summary['mean']:.{digits}f} [{summary['ci_low']:.{digits}f}, {summary['ci_high']:.{digits}f}]"


def render_aggregate_markdown(
    checkpoint: str | None, seed: int, n_prompts: int, failures: list[dict], agg: dict,
    collapse: dict | None,
) -> str:
    lines = ["# RouteLens batch evaluation", ""]
    lines.append(
        f"**Checkpoint:** `{checkpoint or 'untrained'}` &middot; **Prompts:** {n_prompts} sampled from "
        f"`{CORPUS_PATH}` &middot; **Seed:** {seed} &middot; values are mean [95% bootstrap CI over prompts]"
    )
    lines.append("")
    if failures:
        lines.append(f"_{len(failures)} prompt(s) failed and were skipped (see batch_summary.json)._")
        lines.append("")

    lines.append("## Red-team and judge")
    lines.append("")
    lines.append("| explainer | flip rate | mean edit fraction | plausibility (1-5) | divergence rate |")
    lines.append("|---|---|---|---|---|")
    for name, s in agg.items():
        lines.append(
            f"| {name} | {_fmt(s['flip_rate'])} | {_fmt(s['edit_fraction'])} | "
            f"{_fmt(s.get('plausibility'))} | {_fmt(s.get('divergence_rate'))} |"
        )
    lines.append("")

    lines.append("## Faithfulness")
    lines.append("")
    lines.append("| explainer | del_auc | ins_auc | comprehensiveness | sufficiency | router τ |")
    lines.append("|---|---|---|---|---|---|")
    for name, s in agg.items():
        lines.append(
            f"| {name} | {_fmt(s['deletion_auc'], 3)} | {_fmt(s['insertion_auc'], 3)} | "
            f"{_fmt(s['comprehensiveness'], 3)} | {_fmt(s['sufficiency'], 3)} | "
            f"{_fmt(s.get('router_tau'), 3)} |"
        )
    lines.append("")

    if collapse is not None:
        lines.append(
            f"Router expert-collapse score per layer (0 = balanced, 1 = collapsed): {collapse['scores']}"
        )
        lines.append("")
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    kind, model_config = resolve_model_kind_and_config(cfg)
    device = resolve_device(cfg.training.device, kind)
    print(f"device: {device}  kind: {kind}")

    model = build_model(kind, model_config, device, cfg.xai.checkpoint)
    tokenizer = Tokenizer()

    prompts = sample_prompts(CORPUS_PATH, cfg.xai.batch_num_prompts, cfg.seed)
    print(f"sampled {len(prompts)} prompts from {CORPUS_PATH} (seed {cfg.seed})")

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
        seed=cfg.seed,
    )

    out_dir = Path(cfg.xai.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    failures: list[dict] = []
    started = time.time()
    with open(out_dir / "batch_per_prompt.jsonl", "w") as per_prompt_file:
        for i, prompt in enumerate(prompts):
            try:
                state = orchestrator.run(prompt)
            except Exception as e:  # one bad prompt shouldn't cost a long run
                failures.append({"prompt": prompt, "error": f"{type(e).__name__}: {e}"})
                print(f"[{i + 1}/{len(prompts)}] FAILED {prompt!r}: {type(e).__name__}: {e}")
                continue
            record = summarize_state(prompt, state)
            records.append(record)
            per_prompt_file.write(json.dumps(record) + "\n")
            per_prompt_file.flush()
            print(f"[{i + 1}/{len(prompts)}] ({time.time() - started:.0f}s) {prompt!r}")

    if not records:
        raise RuntimeError("every prompt failed -- nothing to aggregate")

    agg = aggregate(records, cfg.seed)

    collapse = None
    if kind == "moe":
        val_ds = TokenDataset(cfg.data.val_bin, block_size=model_config.block_size)
        with torch.no_grad():
            x, _ = get_batch(val_ds, cfg.analysis.batch_size, device)
            _, _, aux = model(x, return_router_outputs=True)
        counts, total_tokens, top_k = accumulate_layer_counts([aux["router_outputs"]])
        scores = collapse_score(utilization_from_counts(counts, total_tokens, top_k))
        collapse = {"scores": [round(s, 3) for s in scores.tolist()]}

    report_md = render_aggregate_markdown(
        cfg.xai.checkpoint, cfg.seed, len(records), failures, agg, collapse
    )
    (out_dir / "batch_report.md").write_text(report_md)
    (out_dir / "batch_summary.json").write_text(json.dumps({
        "checkpoint": cfg.xai.checkpoint,
        "seed": cfg.seed,
        "kind": kind,
        "n_prompts": len(records),
        "failures": failures,
        "aggregate": agg,
        "collapse": collapse,
    }, indent=2))
    print(f"\n{report_md}")
    print(f"wrote {out_dir}/batch_report.md, batch_summary.json, batch_per_prompt.jsonl")


if __name__ == "__main__":
    main()
