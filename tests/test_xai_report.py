from moe_transformer.xai.agents.report_agent import ReportAgent
from moe_transformer.xai.report import render_markdown_report
from moe_transformer.xai.redteam.search import RedTeamResult


def make_redteam_result(flipped: bool) -> RedTeamResult:
    return RedTeamResult(
        protected_positions=[0],
        editable_positions=[1],
        original_prob=0.5,
        final_prob=0.1 if flipped else 0.5,
        flipped=flipped,
        num_edits=1 if flipped else 0,
        edit_fraction=1.0 if flipped else 0.0,
        substitutions=[],
    )


def base_state():
    return {
        "text": "Hello world",
        "target_token_str": "!",
        "tokens": ["Hello", " world"],
        "attention_importance": [0.9, 0.1],
        "shap_words": ["Hello", " world"],
        "shap_values": [0.8, 0.2],
        "lime_words": ["Hello", " world"],
        "lime_values": [0.7, 0.3],
        "router_trace": None,
        "faithfulness": {
            "attention": {
                "deletion_auc": 0.1, "insertion_auc": 0.9,
                "comprehensiveness": 0.5, "sufficiency": 0.1,
                "original_prob": 0.5, "router_attribution_agreement": 0.3,
            },
            "shap": {
                "deletion_auc": 0.1, "insertion_auc": 0.9,
                "comprehensiveness": -0.2, "sufficiency": 0.4,
                "original_prob": 0.5,
            },
            "lime": {
                "deletion_auc": 0.1, "insertion_auc": 0.9,
                "comprehensiveness": 0.5, "sufficiency": 0.9,
                "original_prob": 0.5,
            },
        },
        "redteam": {
            "attention": make_redteam_result(flipped=False),
            "shap": make_redteam_result(flipped=True),
            "lime": make_redteam_result(flipped=True),
        },
        "judge": {
            "plausibility": {},
            "verdicts": {
                "attention": {
                    "plausibility_score": 4, "plausibility_reasoning": "seems fine",
                    "faithfulness_confidence": "faithful", "divergent": False,
                    "needs_deeper_redteam": True,
                },
                "shap": {
                    "plausibility_score": 5, "plausibility_reasoning": "looks obviously right",
                    "faithfulness_confidence": "unfaithful", "divergent": True,
                    "needs_deeper_redteam": False,
                },
                "lime": {
                    "plausibility_score": 4, "plausibility_reasoning": "reasonable",
                    "faithfulness_confidence": "unfaithful", "divergent": True,
                    "needs_deeper_redteam": False,
                },
            },
        },
        "redteam_deep_retry": ["attention"],
    }


def test_render_markdown_report_includes_prompt_and_target():
    report = render_markdown_report(base_state())
    assert "Hello world" in report
    assert "!" in report


def test_render_markdown_report_flags_divergent_explainers_in_findings():
    report = render_markdown_report(base_state())
    assert "## Findings" in report
    assert "**shap**" in report
    assert "**lime**" in report
    # attention is not divergent, so it shouldn't appear in the findings bullets
    findings_section = report.split("## Findings")[1]
    assert "**attention**" not in findings_section


def test_render_markdown_report_notes_deep_retry():
    report = render_markdown_report(base_state())
    assert "Deeper red-team retry ran for: attention" in report


def test_render_markdown_report_handles_no_router_trace():
    state = base_state()
    report = render_markdown_report(state)
    assert "No router trace" in report


def test_report_agent_writes_markdown_into_state():
    state = base_state()
    result = ReportAgent().run(state)
    assert "report_markdown" in result
    assert "# RouteLens explanation report" in result["report_markdown"]
