# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Fork addition: a KL constraint for the search and an early abort of refusal counting.
#
# `kl_constraint` makes Optuna treat a trial whose KL divergence is above the limit as
# infeasible, so the sampler searches the useful zone only; such a trial also skips most
# of its refusal evaluation. `early_abort` stops the refusal evaluation after the first
# chunk of prompts when the refusal rate is clearly worse than the median of the finished
# feasible trials. Both are off by default; with both off nothing here is on the code path.

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, TrialState

if TYPE_CHECKING:
    from .config import Settings

# One-sided 95% confidence.
WILSON_Z = 1.645

# The abort stays inactive until this many finished feasible trials exist.
MIN_FEASIBLE_TRIALS = 5

# A trial without a recorded `kl_excess` (an interrupted trial, a journal of an older run)
# counts as infeasible by this much.
MISSING_KL_EXCESS = 1e9


@dataclass
class TrialLimits:
    """
    Limits of one trial, set by the optimization loop before the scorers run and read
    by the scorers through `Context.limits`. The last three fields are outputs.
    """

    # KL divergence above this makes the trial infeasible. None: no constraint.
    kl_constraint: float | None = None
    # Refusal rate above which a trial is aborted after the first chunk. None: inactive.
    abort_threshold: float | None = None
    # Set by the evaluator once the KL scorer reported a value above the limit.
    kl_exceeded: bool = False
    # Outputs: the refusal scorer evaluated only the first chunk, and why.
    partial: bool = False
    early_abort: bool = False


def wilson_lower_bound(successes: int, total: int, z: float = WILSON_Z) -> float:
    """One-sided lower bound of the Wilson score interval of a proportion."""
    if total <= 0:
        return 0.0
    p = successes / total
    z2 = z * z
    centre = p + z2 / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    return max(0.0, (centre - margin) / (1 + z2 / total))


def first_chunk_size(total: int) -> int:
    """Size of the first chunk of prompts: a third, at least one."""
    return max(1, math.ceil(total / 3))


def _kl_constraint_violation(trial: Any) -> list[float]:
    excess = trial.user_attrs.get("kl_excess")
    return [MISSING_KL_EXCESS if excess is None else float(excess)]


def build_sampler(settings: "Settings") -> TPESampler:
    """The TPE sampler of the study; constrained by the KL limit when one is set."""
    kwargs: dict[str, Any] = {}
    if settings.kl_constraint is not None:
        kwargs["constraints_func"] = _kl_constraint_violation
    return TPESampler(
        n_startup_trials=settings.n_startup_trials,
        n_ei_candidates=128,
        multivariate=True,
        seed=settings.seed,
        **kwargs,
    )


def abort_threshold(
    trials: Sequence[FrozenTrial],
    value_index: int,
    kl_constraint: float | None,
) -> float | None:
    """
    Median refusal value of the finished feasible trials, or None while there are fewer
    than MIN_FEASIBLE_TRIALS of them. Trials cut short (`partial`, `early_abort`) carry
    a rate of a part of the prompts and are left out.
    """
    values: list[float] = []
    for trial in trials:
        if trial.state != TrialState.COMPLETE or trial.values is None:
            continue
        if trial.user_attrs.get("partial") or trial.user_attrs.get("early_abort"):
            continue
        if kl_constraint is not None:
            excess = trial.user_attrs.get("kl_excess")
            if excess is None or excess > 0:
                continue
        values.append(trial.values[value_index])

    if len(values) < MIN_FEASIBLE_TRIALS:
        return None
    return statistics.median(values)


def make_trial_limits(
    settings: "Settings",
    trials: Sequence[FrozenTrial],
    value_index: int | None,
) -> TrialLimits | None:
    """Limits for the next trial, or None when both settings are off."""
    if settings.kl_constraint is None and not settings.early_abort:
        return None
    threshold = (
        abort_threshold(trials, value_index, settings.kl_constraint)
        if settings.early_abort and value_index is not None
        else None
    )
    return TrialLimits(
        kl_constraint=settings.kl_constraint,
        abort_threshold=threshold,
    )


def record_trial_limits(
    trial: Any, limits: TrialLimits | None, kl_value: float | None
) -> None:
    """Writes the user attributes of the limits; nothing when no limits were active."""
    if limits is None:
        return
    if limits.kl_constraint is not None and kl_value is not None:
        trial.set_user_attr("kl_excess", kl_value - limits.kl_constraint)
    if limits.partial:
        trial.set_user_attr("partial", True)
    if limits.early_abort:
        trial.set_user_attr("early_abort", True)
