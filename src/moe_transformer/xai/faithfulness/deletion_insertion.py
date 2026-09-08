"""Deletion/Insertion faithfulness curves (Petsiuk et al., RISE, 2018).

An attribution is faithful if the tokens it calls most important are the
ones the model actually relies on -- not just tokens that look plausible to
a human. Deletion removes tokens in descending |importance| order and
tracks how fast P(target_token) collapses (a faithful explainer front-loads
the collapse -> low area under the curve). Insertion does the reverse,
building the sequence back up from nothing (a faithful explainer front-loads
the recovery -> high area under the curve). Neither metric needs to know
what "important" is supposed to mean -- only whether the ranking matches
what actually moves the model's output.
"""

from __future__ import annotations

from moe_transformer.xai.faithfulness.utils import EvalFn, sorted_by_importance


def deletion_curve(values: list[float], eval_fn: EvalFn) -> tuple[list[float], list[float]]:
    """Returns (fractions_removed, probabilities), starting from the full
    sequence (fraction=0) and removing one token at a time in descending
    |importance| order."""
    order = sorted_by_importance(values)
    n = len(values)
    all_positions = set(range(n))
    fractions = [0.0]
    probs = [eval_fn(all_positions)]
    removed: set[int] = set()
    for step, idx in enumerate(order, start=1):
        removed.add(idx)
        fractions.append(step / n)
        probs.append(eval_fn(all_positions - removed))
    return fractions, probs


def insertion_curve(values: list[float], eval_fn: EvalFn) -> tuple[list[float], list[float]]:
    """Returns (fractions_inserted, probabilities), starting from an empty
    sequence (fraction=0) and adding one token at a time in descending
    |importance| order."""
    order = sorted_by_importance(values)
    n = len(values)
    fractions = [0.0]
    probs = [eval_fn(set())]
    kept: set[int] = set()
    for step, idx in enumerate(order, start=1):
        kept.add(idx)
        fractions.append(step / n)
        probs.append(eval_fn(kept))
    return fractions, probs
