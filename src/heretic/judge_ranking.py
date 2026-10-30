# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Fork addition: re-rank the best trials by the harness judge after the optimization.
#
# The optimization picks trials by keyword refusals, which cannot see non-answers without
# refusal words (PARTIAL: overview instead of specifics, half the steps, educational pivot).
# Measured with the judge, a trial that the keywords rated well kept 42% refusals on
# Qwen3-14B. Judging every trial is too slow, so the first n_judge_ranking_trials trials
# of the Pareto front within judge_ranking_max_kl are judged, and the menu puts them first,
# sorted by their share of non-answers (REFUSAL + PARTIAL by default), then by KL.

import json
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from optuna.study import Study
from optuna.trial import FrozenTrial

from .config import Settings
from .judge import JudgeError, JudgeResult
from .model import Model
from .modifier import Context, Modifier
from .rows import model_label, response_key, response_row
from .scorers.judge_rate import JudgeRate, judge_responses
from .scorers.judge_rate import Settings as JudgeSettings
from .system import empty_cache
from .utils import Prompt, format_exception, load_prompts, print

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


def select_trial_by_number(study: Study, number: int) -> FrozenTrial:
    """The completed trial shown as "[Trial number]" (its "index" attribute)."""
    from optuna.trial import TrialState

    from .utils import HereticUserError

    for trial in study.get_trials(states=(TrialState.COMPLETE,)):
        if trial.user_attrs.get("index") == number:
            return trial
    raise HereticUserError(
        f"There is no completed trial {number} in this study (--trial-number)."
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


def category_rates(
    result: JudgeResult, prompts: Sequence["RankingPrompt"]
) -> dict[str, list[int]]:
    """Per prompt category: [not answered, total]. Categories are the ranking prompts'."""
    rates: dict[str, list[int]] = {}
    for ranking_prompt, label in zip(prompts, result.labels):
        category = ranking_prompt.category or ""
        if not category:
            continue
        entry = rates.setdefault(category, [0, 0])
        entry[0] += int(label in result.refusal_labels)
        entry[1] += 1
    return rates


def format_categories(rates: dict[str, list[int]], worst: int = 3) -> str:
    """The categories with the highest share of non-answers, worst first."""
    if not rates:
        return ""
    ranked = sorted(
        rates.items(), key=lambda item: (-item[1][0] / max(item[1][1], 1), item[0])
    )
    shown = ", ".join(
        f"{category} {count}/{total}" for category, (count, total) in ranked[:worst]
    )
    return f"worst categories: {shown}"


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
    model_name: str,
    trial: str,
    set_name: str,
    prompts: list["RankingPrompt"],
    responses: list[str],
    result: JudgeResult,
    prompt_version: str,
) -> None:
    """The judged answers as the evaluation harness rows with the label (as in a judged.jsonl)."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as file:
        for ranking_prompt, response, label in zip(prompts, responses, result.labels):
            line = response_row(
                model_name=model_name,
                trial=trial,
                set_name=set_name,
                prompt_id=ranking_prompt.id,
                category=ranking_prompt.category,
                prompt=ranking_prompt.prompt.user,
                response=response,
                settings={
                    "generator": "heretic",
                    "system_prompt": ranking_prompt.prompt.system,
                },
                label=label,
                judge_prompt=prompt_version,
            )
            file.write(json.dumps(line, ensure_ascii=False) + "\n")


def warn_if_hold_out(path: str, patterns: Sequence[str]) -> bool:
    """
    Warns when the ranking prompts look like an evaluation hold-out: a set that took part in
    selecting a trial no longer measures that trial independently. The patterns are shell
    globs matched against the file name, case insensitively (setting
    judge_ranking_hold_out_patterns), so this is a convention check and names no file of any
    particular evaluation harness.
    """
    name = Path(path).name.lower()
    if not any(fnmatch(name, pattern.lower()) for pattern in patterns):
        return False
    print(
        f"[yellow]WARNING: {path} looks like an evaluation hold-out (it matches "
        "judge_ranking_hold_out_patterns). Selecting trials on it makes its numbers for the "
        "selected trial optimistic; rank on a separate validation set and keep the hold-out "
        "for the final measurement.[/]"
    )
    return True


@dataclass
class RankingPrompt:
    id: str
    category: str
    prompt: Prompt


def load_ranking_prompts(
    settings: Settings, judge_settings: JudgeSettings
) -> list[RankingPrompt]:
    """
    The ranking prompts: from judge_ranking_prompts_file (JSON Lines with id, category,
    prompt), or from the [scorer.JudgeRate] prompts with ids <set>-<line>.
    """
    if settings.judge_ranking_prompts_file is not None:
        warn_if_hold_out(
            settings.judge_ranking_prompts_file,
            settings.judge_ranking_hold_out_patterns,
        )
        prompts = []
        with open(settings.judge_ranking_prompts_file, encoding="utf-8") as file:
            for index, line in enumerate(line for line in file if line.strip()):
                row = json.loads(line)
                prompts.append(
                    RankingPrompt(
                        id=str(
                            row.get("id", f"{settings.judge_ranking_set}-{index:04d}")
                        ),
                        category=str(row.get("category", "")),
                        prompt=Prompt(
                            system=settings.system_prompt, user=row["prompt"]
                        ),
                    )
                )
        return prompts

    return [
        RankingPrompt(
            id=f"{settings.judge_ranking_set}-{index:04d}", category="", prompt=p
        )
        for index, p in enumerate(load_prompts(settings, judge_settings.prompts))
    ]


@contextmanager
def trial_model(
    settings: Settings,
    model: Model,
    modifier: Modifier[Any],
    ctx: Context,
    trial: FrozenTrial | None,
):
    """
    Puts the original model (trial None) or a trial's model in place for the duration of
    the block. With judge_ranking_merged the trial is merged into the weights in memory,
    so what is measured is what an export contains (the live LoRA rounds differently,
    upstream #490). Without quantization, merging consumes the live model (needs_reload is
    set), so the next reset reloads it.
    """
    modifier.reset_model(ctx)
    if trial is not None:
        modifier.modify_model(
            ctx,
            modifier.parameters_class.from_dict(trial.user_attrs["parameters"]),
        )

    if trial is None or not settings.judge_ranking_merged:
        yield model
        return

    merged = model.get_merged_model()
    live = model.model
    model.model = merged
    try:
        yield model
    finally:
        model.model = live
        del merged
        empty_cache()


def generate_answers(
    settings: Settings,
    model: Model,
    modifier: Modifier[Any],
    ctx: Context,
    trial: FrozenTrial | None,
    prompts: list[Prompt],
) -> list[str]:
    """Answers of the original model (trial None) or of a trial, at the ranking length."""
    with (
        trial_model(settings, model, modifier, ctx, trial),
        generation_settings(
            settings,
            settings.judge_ranking_response_length,
            settings.judge_ranking_batch_size,
        ),
    ):
        return model.get_responses_batched(prompts, skip_special_tokens=True)


def write_candidate_answers(
    sorted_trials: list[FrozenTrial],
    settings: Settings,
    model: Model,
    modifier: Modifier[Any],
) -> str | None:
    """
    The "generate" mode: appends the answers of the original model ("base") and of the
    candidate trials (labelled with their trial numbers) to judge_ranking_output, in the
    format of the harness responses.jsonl, skipping keys already there. Returns the path.
    """
    if settings.n_judge_ranking_trials == 0:
        return None
    output = settings.judge_ranking_output
    if output is None:
        raise ValueError('judge_ranking_mode = "generate" needs judge_ranking_output')

    judge_settings = get_judge_settings(settings)
    prompts = load_ranking_prompts(settings, judge_settings)
    model_name = settings.judge_ranking_model_name or model_label(settings.model)
    candidates = select_candidates(
        sorted_trials, settings.n_judge_ranking_trials, settings.judge_ranking_max_kl
    )

    done: set[str] = set()
    if Path(output).exists():
        with open(output, encoding="utf-8") as file:
            for line in file:
                try:
                    done.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    # A torn last line of an interrupted run.
                    continue

    jobs: list[tuple[str, FrozenTrial | None]] = [("base", None)] + [
        (str(t.user_attrs["index"]), t) for t in candidates
    ]
    jobs = [
        (label, trial)
        for label, trial in jobs
        if any(
            response_key(model_name, label, settings.judge_ranking_set, p.id)
            not in done
            for p in prompts
        )
    ]

    print()
    print(
        f"Writing answers of the original model and the best trials for judging to "
        f"[bold]{output}[/] ({len(jobs)} still to generate, "
        f"{settings.judge_ranking_response_length} tokens per answer)..."
    )
    ctx = Context(settings=settings, model=model)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    run_settings = {
        "max_new_tokens": settings.judge_ranking_response_length,
        "system_prompt": settings.system_prompt,
        "response_prefix": settings.response_prefix,
        "temperature": 0.0,
        "generator": "heretic",
        "operator": "merged" if settings.judge_ranking_merged else "live LoRA",
    }

    for label, trial in jobs:
        title = "Original model" if trial is None else f"Trial {label}"
        try:
            responses = generate_answers(
                settings, model, modifier, ctx, trial, [p.prompt for p in prompts]
            )
        except Exception as error:
            print(f"[yellow]{title}: generation failed ({format_exception(error)}).[/]")
            continue

        rows = []
        for ranking_prompt, response in zip(prompts, responses):
            key = response_key(
                model_name, label, settings.judge_ranking_set, ranking_prompt.id
            )
            if key in done:
                continue
            ids = model.tokenizer(response, add_special_tokens=False)["input_ids"]
            rows.append(
                response_row(
                    model_name=model_name,
                    trial=label,
                    set_name=settings.judge_ranking_set,
                    prompt_id=ranking_prompt.id,
                    category=ranking_prompt.category,
                    prompt=ranking_prompt.prompt.user,
                    response=response,
                    settings=run_settings,
                    n_gen_tokens=len(ids),
                    finish="length"
                    if len(ids) >= settings.judge_ranking_response_length
                    else "stop",
                    win100=model.tokenizer.decode(ids[:100]),
                    win256=model.tokenizer.decode(ids[:256]),
                )
            )
        with open(output, "a", encoding="utf-8") as file:
            file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        print(f"* {title}: {len(rows)} answers written")

    modifier.reset_model(ctx)
    return output


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
    ranking_prompts = load_ranking_prompts(settings, judge_settings)
    prompts = [p.prompt for p in ranking_prompts]
    model_name = settings.judge_ranking_model_name or model_label(settings.model)
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
            responses = generate_answers(settings, model, modifier, ctx, trial, prompts)
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
        stored[key]["categories"] = category_rates(result, ranking_prompts)
        study.set_user_attr(JUDGE_SCORES_ATTR, stored)
        if settings.responses_file is not None:
            append_responses(
                settings.responses_file,
                model_name,
                "base" if trial is None else str(trial.user_attrs["index"]),
                settings.judge_ranking_set,
                ranking_prompts,
                responses,
                result,
                version,
            )
        categories = format_categories(stored[key]["categories"])
        print(
            f"* {title}: [bold]{format_judge_record(stored[key], judge_settings.labels)}[/] "
            "not answered" + (f"; {categories}" if categories else "")
        )

    # Leave the model as the menu expects it: the next selection resets it anyway.
    modifier.reset_model(ctx)
    return {k: v for k, v in stored.items() if is_current(k)}
