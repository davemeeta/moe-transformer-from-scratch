"""RouteLens: multi-agent explainability auditor for this project's models.

Phase 1 provides the Explainer Agent -- SHAP, LIME, attention-rollout, and
MoE router-trace attributions for a single next-token prediction, all
produced from the same trained checkpoint via moe_transformer.models. Later
phases (faithfulness scoring, adversarial red-teaming, an LLM judge, report
synthesis) read the ExplainerAgent's output rather than recomputing it.
"""

from __future__ import annotations
