"""Explainer Agent CLI: runs SHAP, LIME, attention rollout, and (for MoE
checkpoints) router trace on one prompt, saves attribution bar charts and a
JSON summary. Works on dense or MoE checkpoints -- router trace is simply
None for dense.

Usage:
    python -m moe_transformer.xai.run_explainer xai.checkpoint=checkpoints/moe/step_000030
    python -m moe_transformer.xai.run_explainer xai.sample_text="some other prompt"
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from moe_transformer.checkpoint import load_checkpoint
from moe_transformer.config import ModelConfig
from moe_transformer.data import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.train import build_model_config, resolve_device
from moe_transformer.xai.orchestrator import Orchestrator
from moe_transformer.xai.visualize import plot_attribution_bars


def resolve_model_kind_and_config(cfg: DictConfig) -> tuple[str, ModelConfig]:
    """Reads the model kind + config this run will use -- from the
    checkpoint's own saved config.yaml if one is given (same pattern as
    compare.py/routing_analysis.py), else from cfg.model -- without touching
    a device yet. Kind must be known *before* resolve_device runs: MoE's
    dynamic-shape router dispatch is pathologically slow on MPS (see
    model/moe.py's docstring), so picking a device based on cfg.model.kind's
    default ("dense") while actually loading a MoE checkpoint would silently
    route MoE onto the wrong device.
    """
    if cfg.xai.checkpoint:
        saved_cfg = OmegaConf.load(Path(cfg.xai.checkpoint) / "config.yaml")
        return saved_cfg.model.kind, build_model_config(saved_cfg.model)
    return cfg.model.kind, build_model_config(cfg.model)


def build_model(
    kind: str, model_config: ModelConfig, device: str, checkpoint: str | None
) -> DenseGPT | MoEGPT:
    model = DenseGPT(model_config) if kind == "dense" else MoEGPT(model_config)
    if checkpoint:
        load_checkpoint(checkpoint, model, device=device)
        print(f"loaded checkpoint from {checkpoint} (kind={kind})")
    else:
        print("no checkpoint given -- explaining a freshly-initialized (untrained) model")
    model.to(device)
    model.eval()
    return model


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    kind, model_config = resolve_model_kind_and_config(cfg)
    device = resolve_device(cfg.training.device, kind)
    print(f"device: {device}  kind: {kind}")

    model = build_model(kind, model_config, device, cfg.xai.checkpoint)
    block_size = model_config.block_size
    tokenizer = Tokenizer()

    orchestrator = Orchestrator(
        model,
        tokenizer,
        device,
        block_size,
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
    result = orchestrator.run(cfg.xai.sample_text, cfg.xai.target_token_id)

    print(f"\nprompt: {cfg.xai.sample_text!r}")
    print(f"explaining P(next_token={result['target_token_str']!r})")

    out_dir = Path(cfg.xai.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_attribution_bars(
        result["tokens"],
        result["attention_importance"],
        f"Attention rollout -> P({result['target_token_str']!r})",
        out_dir / "attention_importance.png",
    )
    plot_attribution_bars(
        result["shap_words"],
        result["shap_values"],
        f"SHAP -> P({result['target_token_str']!r})",
        out_dir / "shap_attribution.png",
    )
    plot_attribution_bars(
        result["lime_words"],
        result["lime_values"],
        f"LIME -> P({result['target_token_str']!r})",
        out_dir / "lime_attribution.png",
    )
    print(f"wrote attribution charts to {out_dir}/")

    router_trace = result["router_trace"]
    if router_trace is not None:
        top1_last_layer = router_trace["top1_experts"][-1].tolist()
        print("\nlast-layer top-1 expert per token:")
        for tok, expert in zip(result["tokens"], top1_last_layer):
            print(f"  {tok!r:>12} -> expert {expert}")
    else:
        print("\nno router trace (dense model has no router)")

    print("\nfaithfulness (higher deletion/comprehensiveness = better; "
          "lower insertion/sufficiency = better; positive router agreement = attention "
          "tracks routing):")
    header = f"  {'explainer':<10} {'del_auc':>10} {'ins_auc':>10} {'comp':>10} {'suff':>10} {'router_tau':>10}"
    print(header)
    for name, scores in result["faithfulness"].items():
        router_tau = scores.get("router_attribution_agreement")
        router_str = f"{router_tau:>10.3f}" if router_tau is not None else f"{'--':>10}"
        print(
            f"  {name:<10} {scores['deletion_auc']:>10.3g} {scores['insertion_auc']:>10.3g} "
            f"{scores['comprehensiveness']:>10.3g} {scores['sufficiency']:>10.3g} {router_str}"
        )

    print("\nred-team necessity attack (fewer edits/lower edit_fraction to flip = "
          "less faithful; protected tokens are each explainer's own top-k):")
    header = f"  {'explainer':<10} {'protected':>9} {'edits':>6} {'edit_frac':>10} {'flipped':>8} {'final_prob':>12}"
    print(header)
    for name, rt in result["redteam"].items():
        print(
            f"  {name:<10} {len(rt.protected_positions):>9} {rt.num_edits:>6} "
            f"{rt.edit_fraction:>10.3g} {str(rt.flipped):>8} {rt.final_prob:>12.3g}"
        )
    if result["redteam_deep_retry"]:
        print(f"  (deeper red-team retry ran for: {', '.join(result['redteam_deep_retry'])})")

    print("\njudge (independent plausibility read vs. the faithfulness/red-team evidence "
          "above -- 'divergent' means the two signals disagree):")
    header = f"  {'explainer':<10} {'plausibility':>12} {'faithfulness':>14} {'divergent':>10}"
    print(header)
    for name, verdict in result["judge"]["verdicts"].items():
        print(
            f"  {name:<10} {verdict['plausibility_score']:>12} "
            f"{verdict['faithfulness_confidence']:>14} {str(verdict['divergent']):>10}"
        )
        print(f"    -> {verdict['plausibility_reasoning']}")

    summary = {
        "text": cfg.xai.sample_text,
        "tokens": result["tokens"],
        "target_token_id": result["target_token_id"],
        "target_token_str": result["target_token_str"],
        "attention_importance": result["attention_importance"],
        "shap_words": result["shap_words"],
        "shap_values": result["shap_values"],
        "lime_words": result["lime_words"],
        "lime_values": result["lime_values"],
        "router_top1_last_layer": (
            router_trace["top1_experts"][-1].tolist() if router_trace is not None else None
        ),
        "faithfulness": result["faithfulness"],
        "redteam": {name: rt.to_dict() for name, rt in result["redteam"].items()},
        "redteam_deep_retry": result["redteam_deep_retry"],
        "judge": result["judge"],
    }
    summary_path = out_dir / "explanation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")

    report_path = out_dir / "report.md"
    report_path.write_text(result["report_markdown"])
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
