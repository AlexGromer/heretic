# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Fork addition: a KL constraint for the search and an early abort of refusal counting.
#
# `kl_constraint` makes Optuna treat a trial whose KL divergence is above the limit as
# infeasible, so the sampler searches the useful zone only; such a trial also skips most
# of its refusal evaluation. `early_abort` stops the refusal evaluation after the first
# chunk of prompts when the refusal rate is clearly worse than the 0.75 quantile of the
# refusal values of the finished feasible trials. Both are off by default; with both off
# nothing here is on the code path.

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, TrialState

if TYPE_CHECKING:
    from .config import Settings

# One-sided 95% confidence.
WILSON_Z = 1.645

# The abort stays inactive until max(MIN_FEASIBLE_TRIALS, n_startup_trials) finished
# feasible trials exist (the startup trials of the sampler are random, so the threshold
# of fewer trials says little about the zone TPE will search).
MIN_FEASIBLE_TRIALS = 5

# The abort threshold is this quantile of the refusal values of the finished feasible trials.
ABORT_QUANTILE = 0.75

# A trial without a recorded `kl_excess` whose KL objective value is not known either
# (an interrupted trial) counts as infeasible by this much.
MISSING_KL_EXCESS = 1e9


@dataclass
class TrialLimits:
    """
    Limits of one trial, set by the optimization loop before the scorers run and read
    by the scorers through `Context.limits`. The last three fields are outputs.
    """

    # KL divergence above this makes the trial infeasible. None: no constraint.
    kl_constraint: float | None = None
    # Refusal rate above which a trial is aborted after the first chunk (when the Wilson
    # lower bound of the first chunk is above it). None: inactive.
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


def kl_excess_of(
    trial: Any, kl_constraint: float, kl_index: int | None
) -> float | None:
    """
    KL minus the limit of a trial: the recorded `kl_excess`, else computed from the KL
    objective value (`kl_index` is its position among the study values). A journal of an
    older run, or one written without the limits, has no attribute but does have the value.
    None when neither is known (an interrupted trial, no KL objective).
    """
    excess = trial.user_attrs.get("kl_excess")
    if excess is not None:
        return float(excess)
    values = getattr(trial, "values", None)
    if kl_index is None or values is None or not 0 <= kl_index < len(values):
        return None
    return float(values[kl_index]) - kl_constraint


def backfill_constraints(
    trials: Sequence[FrozenTrial], kl_constraint: float, kl_index: int | None
) -> int:
    """
    Gives the finished trials that have no constraint value their `KL - limit`, computed
    from the stored KL objective value; returns how many were changed. Optuna sets the
    constraint of a trial once, when it finishes, and never again, and a trial without one
    counts as feasible for TPE. So on `continue` / extended trials of a journal written
    without the limit (or with another one), the old trials with a KL above the limit would
    be treated as feasible. The change lives in memory only: the journal is not touched and
    the next start derives the same values. A trial whose KL is unknown is left as it is.
    """
    changed = 0
    for trial in trials:
        if trial.state not in (TrialState.COMPLETE, TrialState.PRUNED):
            continue
        if trial.constraints:
            continue
        excess = kl_excess_of(trial, kl_constraint, kl_index)
        if excess is None:
            continue
        trial.system_attrs["constraints"] = [excess]
        changed += 1
    return changed


class _KLConstrainedSampler(TPESampler):
    """TPE with the KL limit applied also to the trials of an older journal."""

    def __init__(self, kl_constraint: float, kl_index: int | None, **kwargs: Any):
        self._kl_constraint = kl_constraint
        self._kl_index = kl_index
        super().__init__(**kwargs)

    def before_trial(self, study: Any, trial: FrozenTrial) -> None:
        super().before_trial(study, trial)
        # The trial objects TPE reads in `sample_relative` (cached, not copied).
        backfill_constraints(
            study._get_trials(
                deepcopy=False,
                states=(TrialState.COMPLETE, TrialState.PRUNED),
                use_cache=True,
            ),
            self._kl_constraint,
            self._kl_index,
        )


def build_sampler(settings: "Settings", kl_index: int | None = None) -> TPESampler:
    """
    The TPE sampler of the study; constrained by the KL limit when one is set. `kl_index`
    is the position of the KL divergence among the objective values (None: no KL objective).
    """
    kwargs: dict[str, Any] = dict(
        n_startup_trials=settings.n_startup_trials,
        n_ei_candidates=128,
        multivariate=True,
        seed=settings.seed,
    )
    if settings.kl_constraint is None:
        return TPESampler(**kwargs)

    limit = settings.kl_constraint

    def constraints_func(trial: Any) -> list[float]:
        # Called by Optuna when a trial finishes; `kl_excess` was recorded by the loop.
        excess = kl_excess_of(trial, limit, kl_index)
        return [MISSING_KL_EXCESS if excess is None else excess]

    return _KLConstrainedSampler(
        limit, kl_index, constraints_func=constraints_func, **kwargs
    )


def quantile(values: Sequence[float], q: float) -> float:
    """Quantile with linear interpolation between the order statistics."""
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def abort_threshold(
    trials: Sequence[FrozenTrial],
    value_index: int,
    kl_constraint: float | None,
    kl_index: int | None = None,
    min_trials: int = MIN_FEASIBLE_TRIALS,
) -> float | None:
    """
    The 0.75 quantile of the refusal values of the finished feasible trials, or None while
    there are fewer than `min_trials` of them. Trials that the abort cut short are counted
    with their partial value: that value is the rate of the first chunk and is biased upward
    by the stopping rule (the trial stopped because the chunk looked bad), so it is not an
    unbiased estimate of the full rate, but leaving such trials out would bias the threshold
    downward and make the abort ever more aggressive. Trials of the KL-over-limit partial
    evaluation are infeasible and not counted.
    """
    values: list[float] = []
    for trial in trials:
        if trial.state != TrialState.COMPLETE or trial.values is None:
            continue
        if trial.user_attrs.get("partial"):
            continue
        if kl_constraint is not None:
            excess = kl_excess_of(trial, kl_constraint, kl_index)
            if excess is None or excess > 0:
                continue
        values.append(trial.values[value_index])

    if len(values) < min_trials:
        return None
    return quantile(values, ABORT_QUANTILE)


def make_trial_limits(
    settings: "Settings",
    trials: Sequence[FrozenTrial],
    value_index: int | None,
    kl_index: int | None = None,
) -> TrialLimits | None:
    """Limits for the next trial, or None when both settings are off."""
    if settings.kl_constraint is None and not settings.early_abort:
        return None
    threshold = (
        abort_threshold(
            trials,
            value_index,
            settings.kl_constraint,
            kl_index,
            max(MIN_FEASIBLE_TRIALS, settings.n_startup_trials),
        )
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
