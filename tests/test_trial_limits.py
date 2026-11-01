# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests of the KL constraint and the early abort of refusal counting (fork additions).
# Pure Python: no model, no GPU.

import json
import math
from typing import Any

import optuna
from optuna.distributions import FloatDistribution
from optuna.samplers import TPESampler
from optuna.trial import TrialState, create_trial

from heretic.config import ScorerConfig
from heretic.config import Settings as HereticSettings
from heretic.evaluator import Evaluator, ScorerEntry
from heretic.plugin import Context
from heretic.scorer import Score
from heretic.scorers.keyword_rate import KeywordRate, Settings
from heretic.scorers.kl_divergence import KLDivergence
from heretic.trial_limits import (
    MIN_FEASIBLE_TRIALS,
    TrialLimits,
    abort_threshold,
    build_sampler,
    first_chunk_size,
    make_trial_limits,
    record_trial_limits,
    wilson_lower_bound,
)
from heretic.utils import Prompt

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_settings(**kwargs: Any) -> HereticSettings:
    return HereticSettings.model_construct(n_startup_trials=2, seed=1, **kwargs)  # ty:ignore[invalid-argument-type]


# ---------------------------------------------------------------- settings


def test_defaults_unchanged() -> None:
    settings = HereticSettings.model_construct()
    assert HereticSettings.model_fields["kl_constraint"].default is None
    assert HereticSettings.model_fields["early_abort"].default is False
    assert settings.kl_constraint is None
    assert settings.early_abort is False


def test_new_settings_do_not_change_the_stored_settings() -> None:
    # Both are excluded from the settings stored in the journal, so a run with the
    # defaults writes the same study settings as before and `continue` takes the values
    # from the invocation.
    assert HereticSettings.model_fields["kl_constraint"].exclude is True
    assert HereticSettings.model_fields["early_abort"].exclude is True
    stored = json.loads(HereticSettings.model_construct().model_dump_json())
    assert "kl_constraint" not in stored
    assert "early_abort" not in stored


def test_defaults_build_the_old_sampler_and_no_limits() -> None:
    settings = make_settings(kl_constraint=None, early_abort=False)
    sampler = build_sampler(settings)
    assert isinstance(sampler, TPESampler)
    assert sampler._constraints_func is None
    assert sampler._n_ei_candidates == 128
    assert sampler._multivariate is True
    assert make_trial_limits(settings, [], 0) is None


# ------------------------------------------------------------- constraints


def run_study(kl_excesses: list[float | None]) -> optuna.Study:
    settings = make_settings(kl_constraint=0.5, early_abort=False)
    study = optuna.create_study(
        sampler=build_sampler(settings), directions=["minimize", "minimize"]
    )
    queue = list(kl_excesses)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", 0.0, 1.0)
        excess = queue.pop(0)
        if excess is not None:
            trial.set_user_attr("kl_excess", excess)
        return x, 1.0 - x

    study.optimize(objective, n_trials=len(kl_excesses))
    return study


def test_constraints_func_wired_and_receives_kl_excess() -> None:
    settings = make_settings(kl_constraint=0.5)
    sampler = build_sampler(settings)
    assert sampler._constraints_func is not None

    study = run_study([0.3, -0.1, 0.0])
    stored = [list(t.system_attrs["constraints"]) for t in study.trials]
    assert stored == [[0.3], [-0.1], [0.0]]


def test_constraints_func_survives_a_trial_without_kl_excess() -> None:
    # Trials of an older journal, or interrupted before the KL score, have no kl_excess.
    # They count as infeasible instead of crashing the sampler.
    study = run_study([None, -0.2])
    first, second = (list(t.system_attrs["constraints"]) for t in study.trials)
    assert first[0] > 0
    assert second == [-0.2]


# -------------------------------------------------------------- statistics


def test_wilson_lower_bound() -> None:
    # Reference value computed from the closed form.
    k, n, z = 7, 10, 1.645
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    expected = (centre - margin) / (1 + z * z / n)
    assert abs(wilson_lower_bound(k, n) - expected) < 1e-12
    assert 0.0 <= wilson_lower_bound(0, 10) <= 0.01
    assert wilson_lower_bound(10, 10) < 1.0
    assert wilson_lower_bound(5, 0) == 0.0


def test_first_chunk_size() -> None:
    assert first_chunk_size(30) == 10
    assert first_chunk_size(100) == 34
    assert first_chunk_size(2) == 1
    assert first_chunk_size(1) == 1


def frozen(
    refusals: float,
    kl: float,
    *,
    state: TrialState = TrialState.COMPLETE,
    **attrs: Any,
) -> optuna.trial.FrozenTrial:
    return create_trial(
        state=state,
        values=[refusals, kl] if state == TrialState.COMPLETE else None,
        params={"x": 0.5},
        distributions={"x": FloatDistribution(0.0, 1.0)},
        user_attrs=attrs,
    )


def test_threshold_inactive_while_fewer_than_five_feasible() -> None:
    assert MIN_FEASIBLE_TRIALS == 5
    trials = [frozen(0.2 + 0.1 * i, 0.1, kl_excess=-0.4) for i in range(4)]
    assert abort_threshold(trials, 0, 0.5) is None
    assert abort_threshold([], 0, 0.5) is None
    trials.append(frozen(0.6, 0.1, kl_excess=-0.4))
    assert abort_threshold(trials, 0, 0.5) is not None


def test_threshold_is_the_median_of_completed_feasible_trials() -> None:
    trials = [
        frozen(0.9, 0.1, kl_excess=-0.4),
        frozen(0.5, 0.1, kl_excess=-0.4),
        frozen(0.3, 0.1, kl_excess=-0.4),
        frozen(0.7, 0.1, kl_excess=0.0),  # on the limit is feasible
        frozen(0.1, 0.1, kl_excess=-0.4),
        # Not counted: infeasible, partial, aborted, failed.
        frozen(0.0, 9.0, kl_excess=8.5),
        frozen(0.0, 0.1, kl_excess=-0.4, partial=True),
        frozen(0.0, 0.1, kl_excess=-0.4, early_abort=True),
        frozen(0.0, 0.1, kl_excess=-0.4, state=TrialState.FAIL),
        frozen(0.0, 0.1, kl_excess=-0.4, state=TrialState.PRUNED),
    ]
    assert abort_threshold(trials, 0, 0.5) == 0.5


def test_threshold_without_kl_constraint_counts_every_full_trial() -> None:
    trials = [frozen(r, 0.1) for r in (0.1, 0.2, 0.3, 0.4, 0.5)]
    assert abort_threshold(trials, 0, None) == 0.3


def test_make_trial_limits() -> None:
    trials = [frozen(0.1 * i, 0.1, kl_excess=-0.4) for i in range(1, 8)]
    only_kl = make_trial_limits(make_settings(kl_constraint=0.5), trials, 0)
    assert only_kl is not None
    assert only_kl.kl_constraint == 0.5
    assert only_kl.abort_threshold is None

    both = make_trial_limits(
        make_settings(kl_constraint=0.5, early_abort=True), trials, 0
    )
    assert both is not None
    assert both.abort_threshold is not None
    assert abs(both.abort_threshold - 0.4) < 1e-12

    only_abort = make_trial_limits(
        make_settings(kl_constraint=None, early_abort=True), [], 0
    )
    assert only_abort is not None
    assert only_abort.kl_constraint is None
    assert only_abort.abort_threshold is None  # startup


# ---------------------------------------------------------------- recording


class FakeTrial:
    def __init__(self) -> None:
        self.user_attrs: dict[str, Any] = {}

    def set_user_attr(self, key: str, value: Any) -> None:
        self.user_attrs[key] = value


def test_record_without_limits_writes_nothing() -> None:
    trial = FakeTrial()
    record_trial_limits(trial, None, 0.4)  # ty:ignore[invalid-argument-type]
    assert trial.user_attrs == {}


def test_record_kl_excess_and_flags() -> None:
    trial = FakeTrial()
    limits = TrialLimits(kl_constraint=0.5)
    record_trial_limits(trial, limits, 0.75)  # ty:ignore[invalid-argument-type]
    assert trial.user_attrs == {"kl_excess": 0.25}

    trial = FakeTrial()
    limits = TrialLimits(kl_constraint=0.5, partial=True)
    record_trial_limits(trial, limits, 0.75)  # ty:ignore[invalid-argument-type]
    assert trial.user_attrs == {"kl_excess": 0.25, "partial": True}

    trial = FakeTrial()
    limits = TrialLimits(kl_constraint=None, early_abort=True)
    record_trial_limits(trial, limits, 0.1)  # ty:ignore[invalid-argument-type]
    assert trial.user_attrs == {"early_abort": True}


# ------------------------------------------------------------------ scorer


class ChunkContext:
    """Answers refuse for the prompts listed in `refused`; records the chunks asked for."""

    def __init__(self, refused: set[str], limits: TrialLimits | None = None) -> None:
        self.refused = refused
        self.limits = limits
        self.chunks: list[int] = []

    def get_responses(self, prompts: list[Prompt]) -> list[str]:
        self.chunks.append(len(prompts))
        return [
            "I cannot help with that." if p.user in self.refused else "Sure, here."
            for p in prompts
        ]


def make_refusal_scorer(n: int = 30) -> KeywordRate:
    scorer = KeywordRate(
        heretic_settings=HereticSettings.model_construct(), settings=Settings()
    )
    scorer.prompts = [Prompt(system="s", user=f"p{i}") for i in range(n)]
    return scorer


def test_scorer_without_limits_is_one_call_as_before() -> None:
    scorer = make_refusal_scorer()
    ctx = ChunkContext({f"p{i}" for i in range(6)})
    score = scorer.get_score(ctx)  # ty:ignore[invalid-argument-type]
    assert ctx.chunks == [30]
    assert score.value == 6 / 30
    assert score.md_display == "6/30"

    # Limits without threshold and without excess behave the same way.
    ctx = ChunkContext({f"p{i}" for i in range(6)}, TrialLimits(kl_constraint=0.5))
    score = scorer.get_score(ctx)  # ty:ignore[invalid-argument-type]
    assert ctx.chunks == [30]
    assert score.value == 6 / 30
    assert not ctx.limits.partial and not ctx.limits.early_abort  # ty:ignore[unresolved-attribute]


def test_early_abort_fires_above_the_threshold() -> None:
    scorer = make_refusal_scorer()
    # The first third (p0..p9) is refused completely: Wilson bound ~0.78 > 0.5.
    ctx = ChunkContext({f"p{i}" for i in range(10)}, TrialLimits(abort_threshold=0.5))
    score = scorer.get_score(ctx)  # ty:ignore[invalid-argument-type]
    assert wilson_lower_bound(10, 10) > 0.5
    assert ctx.chunks == [10]
    assert ctx.limits.early_abort is True  # ty:ignore[unresolved-attribute]
    assert ctx.limits.partial is False  # ty:ignore[unresolved-attribute]
    # The partial rate is the unbiased rate of the evaluated prompts.
    assert score.value == 1.0
    assert score.md_display == "10/10"


def test_early_abort_does_not_fire_below_the_threshold() -> None:
    scorer = make_refusal_scorer()
    # 2 of the first 10 refused: Wilson bound far below 0.5; the rest follows.
    ctx = ChunkContext({"p0", "p1", "p25"}, TrialLimits(abort_threshold=0.5))
    score = scorer.get_score(ctx)  # ty:ignore[invalid-argument-type]
    assert ctx.chunks == [10, 20]
    assert ctx.limits.early_abort is False  # ty:ignore[unresolved-attribute]
    assert score.value == 3 / 30
    assert score.md_display == "3/30"

    # High observed rate whose lower bound is still below the threshold: no abort.
    ctx = ChunkContext({f"p{i}" for i in range(6)}, TrialLimits(abort_threshold=0.5))
    assert wilson_lower_bound(6, 10) < 0.5
    scorer.get_score(ctx)  # ty:ignore[invalid-argument-type]
    assert ctx.chunks == [10, 20]
    assert ctx.limits.early_abort is False  # ty:ignore[unresolved-attribute]


def test_inactive_threshold_never_aborts() -> None:
    scorer = make_refusal_scorer()
    ctx = ChunkContext({f"p{i}" for i in range(30)}, TrialLimits(abort_threshold=None))
    score = scorer.get_score(ctx)  # ty:ignore[invalid-argument-type]
    assert ctx.chunks == [30]
    assert score.value == 1.0
    assert ctx.limits.early_abort is False  # ty:ignore[unresolved-attribute]


def test_kl_excess_evaluates_only_the_first_third_and_flags_partial() -> None:
    scorer = make_refusal_scorer()
    ctx = ChunkContext(
        {"p0", "p1", "p2"},
        TrialLimits(kl_constraint=0.5, abort_threshold=0.99, kl_exceeded=True),
    )
    score = scorer.get_score(ctx)  # ty:ignore[invalid-argument-type]
    assert ctx.chunks == [10]
    assert ctx.limits.partial is True  # ty:ignore[unresolved-attribute]
    assert ctx.limits.early_abort is False  # ty:ignore[unresolved-attribute]
    assert score.value == 3 / 10
    assert score.md_display == "3/10"


# --------------------------------------------------------------- evaluator


CALLS: list[str] = []


class FakeKL(KLDivergence):
    value = 0.0

    def get_score(self, ctx: Any) -> Score:
        CALLS.append("kl")
        return Score(value=self.value, rich_display="kl", md_display="kl")


class FakeRefusals(KeywordRate):
    seen_limits: Any = "unset"

    def get_score(self, ctx: Any) -> Score:
        CALLS.append("refusals")
        self.seen_limits = ctx.limits
        return Score(value=0.25, rich_display="r", md_display="r")


class FakeOther(KeywordRate):
    seen_limits: Any = "unset"

    def get_score(self, ctx: Any) -> Score:
        CALLS.append("other")
        self.seen_limits = ctx.limits
        return Score(value=0.5, rich_display="o", md_display="o")


def make_evaluator(kl_value: float) -> tuple[Evaluator, FakeRefusals, FakeOther]:
    refusals = FakeRefusals.__new__(FakeRefusals)
    other = FakeOther.__new__(FakeOther)
    kl = FakeKL.__new__(FakeKL)
    kl.value = kl_value
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.settings = HereticSettings.model_construct()
    evaluator.model = None  # ty:ignore[invalid-assignment]
    plugin = "heretic.scorers.keyword_rate.KeywordRate"
    evaluator._scorer_entries = [
        ScorerEntry(
            scorer=refusals,
            name="Refusals",
            config=ScorerConfig(plugin=plugin, optimization="minimize"),
        ),
        ScorerEntry(
            scorer=other,
            name="Over-refusals",
            config=ScorerConfig(plugin=plugin, optimization="minimize"),
        ),
        ScorerEntry(
            scorer=kl,
            name="KL divergence",
            config=ScorerConfig(
                plugin="heretic.scorers.kl_divergence.KLDivergence",
                optimization="minimize",
            ),
        ),
    ]
    return evaluator, refusals, other


def test_get_scores_without_limits_keeps_the_order() -> None:
    CALLS.clear()
    evaluator, refusals, other = make_evaluator(0.1)
    scores = evaluator.get_scores()
    assert CALLS == ["refusals", "other", "kl"]
    assert [name for name, _ in scores] == [
        "Refusals",
        "Over-refusals",
        "KL divergence",
    ]
    assert refusals.seen_limits is None
    assert other.seen_limits is None


def test_kl_scorer_runs_before_the_refusal_scorer_under_limits() -> None:
    CALLS.clear()
    evaluator, refusals, other = make_evaluator(0.1)
    limits = TrialLimits(kl_constraint=0.5)
    scores = evaluator.get_scores(limits)
    assert CALLS == ["kl", "refusals", "other"]
    # The result keeps the configured order, so it stays aligned with the baselines.
    assert [name for name, _ in scores] == [
        "Refusals",
        "Over-refusals",
        "KL divergence",
    ]
    assert limits.kl_exceeded is False
    # Only the refusal objective (first KeywordRate objective) gets the limits.
    assert refusals.seen_limits is limits
    assert other.seen_limits is None
    assert evaluator.get_kl_value(scores) == 0.1


def test_kl_above_the_limit_is_flagged_for_the_refusal_scorer() -> None:
    CALLS.clear()
    evaluator, _, _ = make_evaluator(0.9)
    limits = TrialLimits(kl_constraint=0.5)
    seen: list[bool] = []
    original = FakeRefusals.get_score

    def spy(self: FakeRefusals, ctx: Any) -> Score:
        seen.append(ctx.limits.kl_exceeded)
        return original(self, ctx)

    FakeRefusals.get_score = spy  # ty:ignore[invalid-assignment]
    try:
        evaluator.get_scores(limits)
    finally:
        FakeRefusals.get_score = original  # ty:ignore[invalid-assignment]
    assert seen == [True]
    assert limits.kl_exceeded is True


def test_get_kl_value_none_without_a_kl_scorer() -> None:
    evaluator, _, _ = make_evaluator(0.1)
    evaluator._scorer_entries = evaluator._scorer_entries[:2]
    CALLS.clear()
    scores = evaluator.get_scores()
    assert evaluator.get_kl_value(scores) is None
    assert evaluator.refusal_objective_index() == 0


# ------------------------------------------------------------------- smoke


def test_study_smoke_with_constraint_and_abort() -> None:
    # The loop of `main.objective` on a real in-memory study, with scripted scorers.
    settings = make_settings(kl_constraint=0.5, early_abort=True)
    study = optuna.create_study(
        sampler=build_sampler(settings), directions=["minimize", "minimize"]
    )
    scorer = make_refusal_scorer()
    evaluator, _, _ = make_evaluator(0.0)
    evaluator._scorer_entries[0].scorer = scorer
    evaluator._scorer_entries = [
        evaluator._scorer_entries[0],
        evaluator._scorer_entries[2],
    ]

    script = iter(
        # (KL, refused prompts among p0..p29)
        [(0.1, 3), (0.2, 6), (0.3, 9), (0.1, 12), (0.4, 15)]  # startup, feasible
        + [(0.9, 0), (0.2, 30), (0.2, 2)]
    )

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        kl, refused = next(script)
        trial.suggest_float("x", 0.0, 1.0)
        evaluator._scorer_entries[1].scorer.value = kl  # ty:ignore[unresolved-attribute]
        limits = make_trial_limits(
            settings,
            trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)),
            evaluator.refusal_objective_index(),
        )
        assert limits is not None
        ctx = ChunkContext({f"p{i}" for i in range(refused)})

        # Route the scorer through the scripted context.
        original = evaluator.get_scores

        import heretic.evaluator as evaluator_module

        class Ctx(evaluator_module.Context):
            def get_responses(self, prompts: list[Prompt]) -> list[str]:
                return ctx.get_responses(prompts)

        evaluator_module.Context = Ctx  # ty:ignore[invalid-assignment]
        try:
            scores = original(limits)
        finally:
            evaluator_module.Context = Context
        record_trial_limits(trial, limits, evaluator.get_kl_value(scores))
        return evaluator.get_objective_values(scores)

    study.optimize(objective, n_trials=8)
    attrs = [t.user_attrs for t in study.trials]
    assert [a["kl_excess"] <= 0 for a in attrs] == [True] * 5 + [False, True, True]
    assert attrs[5].get("partial") is True and "early_abort" not in attrs[5]
    # Median of the five feasible startup trials is 9/30; 30/30 refused is cut short,
    # 2/30 is not.
    assert attrs[6].get("early_abort") is True and "partial" not in attrs[6]
    assert "early_abort" not in attrs[7] and "partial" not in attrs[7]
    assert study.trials[6].values[0] == 1.0
    assert abs(study.trials[7].values[0] - 2 / 30) < 1e-12
