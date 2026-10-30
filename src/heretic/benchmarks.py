# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Capability benchmarks for the trials that are candidates for export (fork addition).
#
# Abliteration can cost capabilities: the comparison in arXiv 2512.13655 reports GSM8K
# changes between +1.51 and -18.81 points across tools and models. The BenchmarkScore
# scorer can measure that, but it runs on every trial, which is far too slow for a
# benchmark like GSM8K. This runs the configured tasks only on the few candidates the
# judge ranking already looks at (and once on the original model), stores the results in
# the study and shows them next to the trial in the menu.
#
# Each task runs through the Language Model Evaluation Harness on the same model the
# judge ranking uses (merged in memory when judge_ranking_merged is on).

from typing import Any

from optuna.study import Study
from optuna.trial import FrozenTrial

from .config import Settings
from .judge_ranking import BASELINE_KEY, select_candidates, trial_model
from .model import Model
from .modifier import Context, Modifier
from .utils import format_exception, print

# Study user attribute holding the benchmark results, by trial number (as a string) and
# "baseline": {task: {metric: value}}.
BENCHMARK_SCORES_ATTR = "benchmark_scores"


def parse_task(entry: str) -> tuple[str, str]:
    """ "task" or "task:metric" (default metric: acc,none)."""
    task, _, metric = entry.partition(":")
    return task.strip(), (metric.strip() or "acc,none")


def format_benchmarks(record: dict[str, Any]) -> str:
    return ", ".join(f"{task} {value:.3f}" for task, value in sorted(record.items()))


def run_tasks(model: Model, tasks: list[tuple[str, str]]) -> dict[str, float]:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    hflm = HFLM(
        pretrained=model.model,
        tokenizer=model.tokenizer,  # ty:ignore[invalid-argument-type]
        batch_size="auto",
    )
    results = lm_eval.simple_evaluate(model=hflm, tasks=[task for task, _ in tasks])
    scores: dict[str, float] = {}
    for task, metric in tasks:
        task_results = results["results"].get(task, {})
        if metric in task_results:
            scores[task] = float(task_results[metric])
        else:
            print(
                f"[yellow]Benchmark {task}: no metric {metric!r} in its results "
                f"({', '.join(sorted(task_results)) or 'none'}).[/]"
            )
    return scores


def benchmark_trials(
    study: Study,
    sorted_trials: list[FrozenTrial],
    settings: Settings,
    model: Model,
    modifier: Modifier[Any],
) -> dict[str, dict[str, float]]:
    """
    Runs settings.judge_ranking_benchmarks on the original model and on the candidate
    trials that have no stored result yet, stores them in the study and returns all of them.
    """
    stored: dict[str, dict[str, float]] = dict(
        study.user_attrs.get(BENCHMARK_SCORES_ATTR, {})
    )
    if not settings.judge_ranking_benchmarks or settings.n_judge_ranking_trials == 0:
        return stored

    tasks = [parse_task(entry) for entry in settings.judge_ranking_benchmarks]
    candidates = select_candidates(
        sorted_trials, settings.n_judge_ranking_trials, settings.judge_ranking_max_kl
    )
    jobs: list[tuple[str, FrozenTrial | None]] = []
    if BASELINE_KEY not in stored:
        jobs.append((BASELINE_KEY, None))
    jobs += [(str(t.number), t) for t in candidates if str(t.number) not in stored]
    if not jobs:
        return stored

    print()
    print(
        f"Running {', '.join(task for task, _ in tasks)} on the original model and "
        f"[bold]{len(jobs) - (1 if jobs[0][1] is None else 0)}[/] candidate trial(s). "
        "A benchmark takes much longer than an evaluation."
    )
    ctx = Context(settings=settings, model=model)

    for key, trial in jobs:
        title = (
            "Original model" if trial is None else f"Trial {trial.user_attrs['index']}"
        )
        try:
            with trial_model(settings, model, modifier, ctx, trial):
                scores = run_tasks(model, tasks)
        except Exception as error:
            print(f"[yellow]{title}: benchmarks failed ({format_exception(error)}).[/]")
            continue

        stored[key] = scores
        study.set_user_attr(BENCHMARK_SCORES_ATTR, stored)
        baseline = stored.get(BASELINE_KEY, {})
        parts = []
        for task, value in sorted(scores.items()):
            if trial is not None and task in baseline:
                parts.append(f"{task} {value:.3f} ({value - baseline[task]:+.3f})")
            else:
                parts.append(f"{task} {value:.3f}")
        print(f"* {title}: [bold]{', '.join(parts)}[/]")

    modifier.reset_model(ctx)
    return stored
