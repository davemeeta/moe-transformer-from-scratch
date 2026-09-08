"""RouteLens dashboard: type a prompt, see attention/SHAP/LIME/router path/
faithfulness/red-team/judge side by side, live -- so the project is
demoable without narrating a JSON file.

Standalone Gradio app (not a Hydra script, unlike every other CLI in this
project): a checkpoint path is something a user changes interactively here,
which doesn't fit Hydra's command-line-override model, so model loading is
reimplemented directly from the same small pieces run_explainer.py uses
(build_model_config, resolve_device, load_checkpoint) rather than forcing
an interactive UI through a config framework built for one-shot runs.

Usage:
    python -m moe_transformer.xai.dashboard
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import gradio as gr
from omegaconf import OmegaConf

from moe_transformer.checkpoint import load_checkpoint
from moe_transformer.data import Tokenizer
from moe_transformer.models import DenseGPT, MoEGPT
from moe_transformer.train import build_model_config, resolve_device
from moe_transformer.xai.orchestrator import Orchestrator
from moe_transformer.xai.visualize import plot_attribution_bars

TMP_DIR = Path(tempfile.gettempdir()) / "routelens_dashboard"
TMP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CHECKPOINT = "checkpoints/moe_compare_v2/step_000600"
DEFAULT_PROMPT = "First Citizen:\nBefore we proceed any further, hear me speak."

_model_cache: dict[str, tuple] = {}


def load_model(checkpoint: str):
    if checkpoint in _model_cache:
        return _model_cache[checkpoint]

    saved_cfg = OmegaConf.load(Path(checkpoint) / "config.yaml")
    kind = saved_cfg.model.kind
    model_config = build_model_config(saved_cfg.model)
    device = resolve_device("auto", kind)

    model = DenseGPT(model_config) if kind == "dense" else MoEGPT(model_config)
    load_checkpoint(checkpoint, model, device=device)
    model.to(device)
    model.eval()

    tokenizer = Tokenizer()
    result = (model, tokenizer, device, model_config.block_size, kind)
    _model_cache[checkpoint] = result
    return result


def run_pipeline(checkpoint: str, prompt: str):
    if not checkpoint or not Path(checkpoint).exists():
        raise gr.Error(f"checkpoint not found: {checkpoint!r}")
    if not prompt.strip():
        raise gr.Error("enter a prompt")

    model, tokenizer, device, block_size, kind = load_model(checkpoint)
    orchestrator = Orchestrator(model, tokenizer, device, block_size)
    state = orchestrator.run(prompt)

    target = state["target_token_str"]
    headline = f"Predicting P(next_token={target!r})"

    attn_path = TMP_DIR / "attention.png"
    shap_path = TMP_DIR / "shap.png"
    lime_path = TMP_DIR / "lime.png"
    plot_attribution_bars(
        state["tokens"], state["attention_importance"],
        f"Attention rollout -> P({target!r})", attn_path,
    )
    plot_attribution_bars(
        state["shap_words"], state["shap_values"], f"SHAP -> P({target!r})", shap_path,
    )
    plot_attribution_bars(
        state["lime_words"], state["lime_values"], f"LIME -> P({target!r})", lime_path,
    )

    router_trace = state["router_trace"]
    if router_trace is not None:
        top1 = router_trace["top1_experts"][-1].tolist()
        router_rows = [[tok, expert] for tok, expert in zip(state["tokens"], top1)]
    else:
        router_rows = [["(dense model -- no router)", ""]]

    faithfulness_rows = []
    redteam_rows = []
    judge_rows = []
    for name in ("attention", "shap", "lime"):
        f = state["faithfulness"][name]
        router_tau = f.get("router_attribution_agreement")
        faithfulness_rows.append([
            name, round(f["deletion_auc"], 4), round(f["insertion_auc"], 4),
            round(f["comprehensiveness"], 4), round(f["sufficiency"], 4),
            round(router_tau, 4) if router_tau is not None else "--",
        ])

        rt = state["redteam"][name]
        redteam_rows.append([
            name, len(rt.protected_positions), rt.num_edits,
            round(rt.edit_fraction, 4), rt.flipped, round(rt.final_prob, 5),
        ])

        j = state["judge"]["verdicts"][name]
        judge_rows.append([
            name, j["plausibility_score"], j["faithfulness_confidence"], j["divergent"],
            j["plausibility_reasoning"],
        ])

    return (
        headline,
        attn_path, shap_path, lime_path,
        router_rows,
        faithfulness_rows, redteam_rows, judge_rows,
        state["report_markdown"],
    )


with gr.Blocks(title="RouteLens") as demo:
    gr.Markdown("# RouteLens\nA multi-agent auditor for a mixture-of-experts transformer: "
                "attribution, faithfulness, red-teaming, and an independent judge -- live.")

    with gr.Row():
        checkpoint_box = gr.Textbox(label="checkpoint", value=DEFAULT_CHECKPOINT)
        prompt_box = gr.Textbox(label="prompt", value=DEFAULT_PROMPT, lines=2)
    run_button = gr.Button("Run", variant="primary")

    headline_md = gr.Markdown()

    with gr.Row():
        attn_image = gr.Image(label="Attention rollout", type="filepath")
        shap_image = gr.Image(label="SHAP", type="filepath")
        lime_image = gr.Image(label="LIME", type="filepath")

    router_table = gr.Dataframe(
        headers=["token", "last-layer expert"], label="Router trace (top-1 expert per token)"
    )

    with gr.Row():
        faithfulness_table = gr.Dataframe(
            headers=["explainer", "del_auc", "ins_auc", "comp", "suff", "router_tau"],
            label="Faithfulness",
        )
        redteam_table = gr.Dataframe(
            headers=["explainer", "protected", "edits", "edit_frac", "flipped", "final_prob"],
            label="Red-team necessity attack",
        )

    judge_table = gr.Dataframe(
        headers=["explainer", "plausibility", "faithfulness", "divergent", "reasoning"],
        label="Judge",
    )

    report_md = gr.Markdown(label="Full report")

    run_button.click(
        run_pipeline,
        inputs=[checkpoint_box, prompt_box],
        outputs=[
            headline_md, attn_image, shap_image, lime_image,
            router_table, faithfulness_table, redteam_table, judge_table,
            report_md,
        ],
    )


if __name__ == "__main__":
    demo.launch()
