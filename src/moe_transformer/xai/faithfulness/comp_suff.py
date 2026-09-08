"""Comprehensiveness/sufficiency (DeYoung et al., ERASER, 2020): single-
number faithfulness scores at one fixed top-k, cheaper to read at a glance
than a full deletion/insertion curve.

Comprehensiveness: how much P(target_token) drops when the top-k tokens are
*removed* -- high is faithful (those tokens were doing real work).
Sufficiency: how much P(target_token) drops when *only* the top-k tokens are
*kept* -- low is faithful (those tokens alone were enough).
"""

from __future__ import annotations

from moe_transformer.xai.faithfulness.utils import EvalFn, sorted_by_importance


def _top_k_positions(values: list[float], k_fraction: float) -> set[int]:
    n = len(values)
    k = max(1, round(k_fraction * n))
    order = sorted_by_importance(values)
    return set(order[:k])


def comprehensiveness(
    values: list[float], k_fraction: float, original_prob: float, eval_fn: EvalFn
) -> float:
    top_k = _top_k_positions(values, k_fraction)
    all_positions = set(range(len(values)))
    return original_prob - eval_fn(all_positions - top_k)


def sufficiency(
    values: list[float], k_fraction: float, original_prob: float, eval_fn: EvalFn
) -> float:
    top_k = _top_k_positions(values, k_fraction)
    return original_prob - eval_fn(top_k)
