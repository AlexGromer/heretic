# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Fork addition: re-rank the best trials by the refusal-eval judge after the optimization.
#
# The optimization picks trials by keyword refusals, which cannot see non-answers without
# refusal words (PARTIAL: overview instead of specifics, half the steps, educational pivot).
# Measured with the judge, a trial that the keywords rated well kept 42% refusals on
# Qwen3-14B. Judging every trial is too slow, so the first n_judge_ranking_trials trials
# of the Pareto front within judge_ranking_max_kl are judged, and the menu puts them first,
# sorted by their share of non-answers (REFUSAL + PARTIAL by default), then by KL.

import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from optuna.study import Study
from optuna.trial import FrozenTrial

from .config import Settings
from .judge import JudgeError, JudgeResult
from .model import Model
from .modifier import Context, Modifier
from .scorers.judge_rate import JudgeRate, judge_responses
from .scorers.judge_rate import Settings as JudgeSettings
from .utils import format_exception, load_prompts, print

# Study user attribute with the judge results, by trial number (as a string) and "baseline".
JUDGE_SCORES_ATTR = "judge_scores"
BASELINE_KEY = "baseline"
KL_NAME = "KL divergence"


def get_judge_settings(settings: Settings) -> JudgeSettings:
    """The [scorer.JudgeRate] settings (defaults if the section is missing)."""
    validated = JudgeRate.validate_settings(
        JudgeRate.get_settings_raw(settings.model_extra, "scorer", None)
    )
    assert isinstance(validated, JudgeSettings)
    return validated


def score_value(trial: FrozenTrial, name: str) -> float | None:
    return next(
        (
            record["score"]["value"]
            for record in trial.user_attrs.get("scores", [])
            if record["name"] == name
        ),
        None,
    )


def select_candidates(
    sorted_trials: list[FrozenTrial], count: int, max_kl: float
) -> list[FrozenTrial]:
    """The first `count` trials whose KL divergence (if scored) is at most `max_kl`."""
    candidates = []
    for trial in sorted_trials:
        kl = score_value(trial, KL_NAME)
        if kl is None or kl <= max_kl:
            candidates.append(trial)
        if len(candidates) == count:
            break
    return candidates


def result_record(
    result: JudgeResult, prompt_version: str, truncated: list[bool]
) -> dict[str, Any]:
    """
    The stored result. `truncated` marks answers that hit the length limit: a PARTIAL that
    "stops halfway" because of the limit is an artifact of the measurement, not evasion.
    """
    return {
        "rate": result.rate,
        "refusals": result.refusals,
        "total": len(result.labels),
        "counts": dict(result.counts),
        "truncated": sum(truncated),
        "truncated_counts": dict(
            Counter(label for label, cut in zip(result.labels, truncated) if cut)
        ),
        "prompt_version": prompt_version,
    }


def format_judge_record(record: dict[str, Any], labels: list[str]) -> str:
    counts = record["counts"]
    details = ", ".join(f"{label} {counts.get(label, 0)}" for label in labels)
    cut = record.get("truncated_counts", {})
    cut_details = ", ".join(
        f"{label} {cut[label]}" for label in labels if cut.get(label)
    )
    truncated = f"; cut at the limit: {cut_details}" if cut_details else ""
    return f"{record['refusals']}/{record['total']} ({details}{truncated})"


def rank_by_judge(
    sorted_trials: list[FrozenTrial],
    judge_scores: dict[str, dict[str, Any]],
) -> list[FrozenTrial]:
    """Judged trials first, by non-answer rate and then KL; the others keep their order."""
    judged = [t for t in sorted_trials if str(t.number) in judge_scores]
    others = [t for t in sorted_trials if str(t.number) not in judge_scores]
    judged.sort(
        key=lambda t: (
            judge_scores[str(t.number)]["rate"],
            score_value(t, KL_NAME) or 0.0,
        )
    )
    return judged + others


@contextmanager
def generation_settings(settings: Settings, response_length: int, batch_size: int):
    previous = (settings.max_response_length, settings.batch_size)
    settings.max_response_length = response_length
    if batch_size > 0:
        settings.batch_size = batch_size
    try:
        yield
    finally:
        settings.max_response_length, settings.batch_size = previous


def append_responses(
    path: str,
    evaluation: str,
    prompts: list[Any],
    responses: list[str],
    result: JudgeResult,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as file:
        for prompt, response, label in zip(prompts, responses, result.labels):
            line = {
                "evaluation": evaluation,
                "score_name": "Judge ranking",
                "system": prompt.system,
                "prompt": prompt.user,
                "response": response,
                "judge_label": label,
            }
            file.write(json.dumps(line, ensure_ascii=False) + "\n")


def judge_trials(
    study: Study,
    sorted_trials: list[FrozenTrial],
    settings: Settings,
    model: Model,
    modifier: Modifier[Any],
) -> dict[str, dict[str, Any]]:
    """
    Judges the original model (once) and the candidate trials that have no stored result
    for the current rubric, stores the results in the study, and returns the stored results
    of the current rubric (results of another rubric are not comparable).
    """
    if settings.n_judge_ranking_trials == 0:
        return {}

    stored: dict[str, dict[str, Any]] = dict(
        study.user_attrs.get(JUDGE_SCORES_ATTR, {})
    )

    judge_settings = get_judge_settings(settings)
    client = judge_settings.make_client()
    version = client.prompt_version

    def is_current(key: str) -> bool:
        return stored.get(key, {}).get("prompt_version") == version

    candidates = select_candidates(
        sorted_trials, settings.n_judge_ranking_trials, settings.judge_ranking_max_kl
    )
    pending = [t for t in candidates if not is_current(str(t.number))]
    if not pending and is_current(BASELINE_KEY):
        return {k: v for k, v in stored.items() if is_current(k)}

    print()
    print(
        f"Judging the original model and [bold]{len(pending)}[/] trial(s) with "
        f"[bold]{judge_settings.model}[/] (prompt {version}), "
        f"{settings.judge_ranking_response_length} tokens per answer..."
    )
    prompts = load_prompts(settings, judge_settings.prompts)
    ctx = Context(settings=settings, model=model)

    jobs: list[tuple[str, FrozenTrial | None]] = []
    if not is_current(BASELINE_KEY):
        jobs.append((BASELINE_KEY, None))
    jobs += [(str(trial.number), trial) for trial in pending]

    for key, trial in jobs:
        title = (
            "Original model" if trial is None else f"Trial {trial.user_attrs['index']}"
        )
        try:
            modifier.reset_model(ctx)
            if trial is not None:
                modifier.modify_model(
                    ctx,
                    modifier.parameters_class.from_dict(trial.user_attrs["parameters"]),
                )
            with generation_settings(
                settings,
                settings.judge_ranking_response_length,
                settings.judge_ranking_batch_size,
            ):
                responses = model.get_responses_batched(
                    prompts, skip_special_tokens=True
                )
            result = judge_responses(
                client, prompts, responses, judge_settings.refusal_labels
            )
        except JudgeError as error:
            print(
                f"[yellow]The judge failed ({error}); the remaining trials are not judged.[/]"
            )
            break
        except Exception as error:
            print(f"[yellow]{title}: judging failed ({format_exception(error)}).[/]")
            continue

        truncated = [
            len(model.tokenizer(response, add_special_tokens=False)["input_ids"])
            >= settings.judge_ranking_response_length
            for response in responses
        ]
        stored[key] = result_record(result, version, truncated)
        study.set_user_attr(JUDGE_SCORES_ATTR, stored)
        if settings.responses_file is not None:
            append_responses(
                settings.responses_file,
                f"judge:{key}",
                prompts,
                responses,
                result,
            )
        print(
            f"* {title}: [bold]{format_judge_record(stored[key], judge_settings.labels)}[/] "
            "not answered"
        )

    # Leave the model as the menu expects it: the next selection resets it anyway.
    modifier.reset_model(ctx)
    return {k: v for k, v in stored.items() if is_current(k)}
