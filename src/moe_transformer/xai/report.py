"""Renders one prediction's full RouteLens audit -- attributions,
faithfulness scores, red-team result, judge verdict -- into a single
markdown report. Pure string formatting: reads orchestrator state, writes
nothing to disk itself (the CLI decides where reports live, same
separation every other agent already keeps).
"""

from __future__ import annotations

from moe_transformer.xai.agents.base import AgentState


def _top_k_display(words: list[str], values: list[float], k: int = 5) -> str:
    order = sorted(range(len(values)), key=lambda i: -abs(values[i]))[:k]
    return ", ".join(f"`{words[i]!r}` ({values[i]:+.3f})" for i in order)


def render_markdown_report(state: AgentState) -> str:
    lines: list[str] = []
    lines.append("# RouteLens explanation report")
    lines.append("")
    lines.append(f"**Prompt:** `{state['text']!r}`")
    lines.append(f"**Predicting:** `P(next_token={state['target_token_str']!r})`")
    lines.append("")

    router_trace = state.get("router_trace")
    if router_trace is not None:
        lines.append("## Router trace (last layer, top-1 expert per token)")
        lines.append("")
        lines.append("| token | expert |")
        lines.append("|---|---|")
        top1 = router_trace["top1_experts"][-1].tolist()
        for tok, expert in zip(state["tokens"], top1):
            lines.append(f"| `{tok!r}` | {expert} |")
        lines.append("")
    else:
        lines.append("_No router trace -- dense model has no router._")
        lines.append("")

    lines.append("## Top attributions per explainer")
    lines.append("")
    lines.append(f"- **attention:** {_top_k_display(state['tokens'], state['attention_importance'])}")
    lines.append(f"- **shap:** {_top_k_display(state['shap_words'], state['shap_values'])}")
    lines.append(f"- **lime:** {_top_k_display(state['lime_words'], state['lime_values'])}")
    lines.append("")

    lines.append("## Faithfulness + red-team + judge")
    lines.append("")
    lines.append(
        "| explainer | del_auc | ins_auc | comp | suff | router_τ | edits | flipped | "
        "plausibility | faithfulness | divergent |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name in ("attention", "shap", "lime"):
        f = state["faithfulness"][name]
        rt = state["redteam"][name]
        j = state["judge"]["verdicts"][name]
        router_tau = f.get("router_attribution_agreement")
        router_str = f"{router_tau:.3f}" if router_tau is not None else "--"
        lines.append(
            f"| {name} | {f['deletion_auc']:.3g} | {f['insertion_auc']:.3g} | "
            f"{f['comprehensiveness']:.3g} | {f['sufficiency']:.3g} | {router_str} | "
            f"{rt.num_edits} | {rt.flipped} | {j['plausibility_score']} | "
            f"{j['faithfulness_confidence']} | {j['divergent']} |"
        )
    lines.append("")

    divergent_names = [
        name for name, v in state["judge"]["verdicts"].items() if v["divergent"]
    ]
    if divergent_names:
        lines.append("## Findings")
        lines.append("")
        for name in divergent_names:
            v = state["judge"]["verdicts"][name]
            lines.append(
                f"- **{name}** sounds plausible (score {v['plausibility_score']}/5) but is "
                f"**{v['faithfulness_confidence']}** -- the explanation doesn't survive the "
                f"red-team attack. Judge's reasoning: {v['plausibility_reasoning']}"
            )
        lines.append("")

    if state.get("redteam_deep_retry"):
        lines.append(
            f"_Deeper red-team retry ran for: {', '.join(state['redteam_deep_retry'])}._"
        )
        lines.append("")

    return "\n".join(lines)
