# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Fork addition for upstream issue #490: trial scores come from the live LoRA, which
# applies the base matrix and the adapter separately and rounds the intermediate results
# to the model dtype, while the exported model runs the merged matrix W + BA. The two
# do not round alike, so an exported model can score differently than its trial.
# This re-scores the best trials on their merged models after the optimization.
# The approach follows upstream PR #495 (own implementation).

from typing import Any

from optuna.study import Study
from optuna.trial import FrozenTrial

from .config import QuantizationMethod, Settings
from .evaluator import Evaluator
from .model import Model
from .modifier import Context, Modifier
from .system import empty_cache
from .utils import format_exception, print

# Study user attribute holding the merged-model score records, by trial number (as a string,
# because user attributes are stored as JSON).
MERGED_SCORES_ATTR = "merged_scores"


def get_stored_merged_scores(study: Study) -> dict[str, list[dict[str, Any]]]:
    return dict(study.user_attrs.get(MERGED_SCORES_ATTR, {}))


def score_values(records: list[dict[str, Any]]) -> dict[str, float]:
    return {record["name"]: record["score"]["value"] for record in records}


def rank_by_merged_scores(
    sorted_trials: list[FrozenTrial],
    merged_scores: dict[str, list[dict[str, Any]]],
    objective_names: list[str],
) -> list[FrozenTrial]:
    """
    Sorts the trials by their merged-model scores, in the same order as the live
    scores are sorted (non-decreasing objective values), if every trial has merged
    scores. Otherwise the live-score order is kept, since a partial ranking would
    compare scores of different operators.
    """
    if not sorted_trials or not all(
        str(trial.number) in merged_scores for trial in sorted_trials
    ):
        return sorted_trials

    values = {
        trial.number: score_values(merged_scores[str(trial.number)])
        for trial in sorted_trials
    }
    return sorted(
        sorted_trials,
        key=lambda trial: tuple(values[trial.number][name] for name in objective_names),
    )


def rescore_on_merged_models(
    study: Study,
    sorted_trials: list[FrozenTrial],
    settings: Settings,
    model: Model,
    evaluator: Evaluator,
    modifier: Modifier[Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    Evaluates the first `settings.n_merged_score_trials` trials of `sorted_trials` on
    their merged models, prints the live and merged scores, and stores the merged
    records in the study, so that they are not recomputed when the results are shown
    again. Returns all stored merged records by trial number.

    At most one merged model exists at a time. Merging replaces the live model, so the
    model is reloaded before the next trial is restored (Model.needs_reload).
    A trial whose merged evaluation fails keeps only its live scores.
    """
    merged_scores = get_stored_merged_scores(study)

    pending = [
        trial
        for trial in sorted_trials[: settings.n_merged_score_trials]
        if str(trial.number) not in merged_scores
    ]
    if not pending:
        return merged_scores

    print()
    print(
        f"Re-scoring [bold]{len(pending)}[/] trial(s) on merged models, because the exported "
        "model can score differently than the live LoRA the trials were selected by (#490)..."
    )
    if settings.quantization == QuantizationMethod.BNB_4BIT:
        print(
            "[yellow]Merging a 4-bit model reloads the base model in full precision on the CPU "
            "for every trial, which needs a lot of RAM and evaluates slowly.[/]"
        )

    for trial in pending:
        print()
        print(f"* [Trial {trial.user_attrs['index']:>3}] Merging and evaluating...")

        try:
            ctx = Context(settings=settings, model=model)
            modifier.reset_model(ctx)
            modifier.modify_model(
                ctx,
                modifier.parameters_class.from_dict(trial.user_attrs["parameters"]),
            )

            merged_model = model.get_merged_model()
            live_model = model.model
            model.model = merged_model
            try:
                records = evaluator.get_paired_score_records(evaluator.get_scores())
            finally:
                # Without quantization, merge_and_unload has consumed the live model
                # and get_merged_model has set needs_reload, so the next reset
                # reloads it. With 4-bit quantization, the merged model is a separate
                # copy and the live model is still intact.
                model.model = live_model
                del merged_model
                empty_cache()
        except Exception as error:
            print(
                f"  [yellow]Merged evaluation failed ({format_exception(error)}); "
                "only the live scores are available for this trial.[/]"
            )
            continue

        merged_scores[str(trial.number)] = records
        study.set_user_attr(MERGED_SCORES_ATTR, merged_scores)

        live = score_values(trial.user_attrs["scores"])
        for record in records:
            name = record["name"]
            merged = record["score"]["value"]
            if name in live:
                print(
                    f"  * [bold]{name}:[/] live {live[name]:.4g} -> merged {merged:.4g} "
                    f"({merged - live[name]:+.4g})"
                )

    return merged_scores
