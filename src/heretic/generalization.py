# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Fork addition: checks whether the refusal reduction found by the optimizer on the
# evaluation prompts (bad_eval) carries over to a held-out prompt set such as OR-Bench-hard.
#
# Every Optuna trial is selected by its score on the same 100 evaluation prompts, so the
# Pareto front can reward edits that remove refusals on exactly those prompts. A held-out
# KeywordRate instance with optimization = "none" is evaluated alongside without steering
# the search; this module compares the two over the stored trials of a study.

import argparse
import statistics
from dataclasses import dataclass
from math import sqrt

DEFAULT_PRIMARY = "Refusals"
DEFAULT_KL = "KL divergence"


@dataclass
class TrialScores:
    index: int
    values: dict[str, float]
    baselines: dict[str, float]
    pareto: bool


@dataclass
class CandidateRow:
    index: int
    primary: float
    holdout: float
    kl: float | None
    primary_removal: float
    holdout_removal: float

    @property
    def gap(self) -> float:
        return self.primary_removal - self.holdout_removal


@dataclass
class GeneralizationReport:
    primary: str
    holdout: str
    primary_baseline: float
    holdout_baseline: float
    rows: list[CandidateRow]
    median_gap: float | None
    correlation: float | None
    reasons: list[str]
    warnings: list[str]

    @property
    def overfit(self) -> bool:
        return bool(self.reasons)


def removal(score: float, baseline: float) -> float | None:
    """Fraction of the baseline matches that are gone (1 = all, 0 = none, < 0 = more)."""
    if baseline <= 0:
        return None
    return (baseline - score) / baseline


def _ranks(values: list[float]) -> list[float]:
    # Average ranks, so that ties get the same rank.
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while (
            end + 1 < len(order) and values[order[end + 1]] == values[order[position]]
        ):
            end += 1
        rank = (position + end) / 2
        for k in range(position, end + 1):
            ranks[order[k]] = rank
        position = end + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation, or None if it is undefined (fewer than 3 points, constant data)."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None

    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / sqrt(sxx * syy)


def detect_holdout(trials: list[TrialScores], primary: str) -> str:
    """The held-out score: the one other score named like `<primary> - <instance>`."""
    names = {name for trial in trials for name in trial.values}
    candidates = sorted(
        name for name in names if name != primary and name.startswith(f"{primary} - ")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot pick the held-out score automatically from {sorted(names)}; "
            "pass it with --holdout"
        )
    return candidates[0]


def analyze(
    trials: list[TrialScores],
    primary: str = DEFAULT_PRIMARY,
    holdout: str | None = None,
    kl: str = DEFAULT_KL,
    max_kl: float = 0.15,
    gap_threshold: float = 0.2,
    min_correlation: float = 0.5,
    min_holdout_baseline: float = 0.1,
) -> GeneralizationReport:
    """
    Compares the primary refusal rate with the held-out one.

    Candidates are the Pareto optimal trials with KL divergence <= `max_kl` (the trials one
    would export); if there are none, all trials within `max_kl`. The result counts as
    overfitting when the median of (primary removal - held-out removal) over the candidates
    exceeds `gap_threshold`, or when the Spearman correlation of the two refusal rates over all
    trials is below `min_correlation` (picking trials by the primary rate does not pick them by
    the held-out rate).
    """
    if holdout is None:
        holdout = detect_holdout(trials, primary)

    scored = [t for t in trials if primary in t.values and holdout in t.values]
    if not scored:
        raise ValueError(f"No trial has both scores {primary!r} and {holdout!r}")

    primary_baseline = scored[0].baselines[primary]
    holdout_baseline = scored[0].baselines[holdout]

    warnings: list[str] = []
    if holdout_baseline < min_holdout_baseline:
        warnings.append(
            f"The baseline of {holdout!r} is only {holdout_baseline:.2f}: the held-out set "
            "hardly triggers refusals, so its reduction says little"
        )

    def within_kl(trial: TrialScores) -> bool:
        return kl not in trial.values or trial.values[kl] <= max_kl

    candidates = [t for t in scored if t.pareto and within_kl(t)]
    if not candidates:
        candidates = [t for t in scored if within_kl(t)]
        if candidates:
            warnings.append(
                f"No Pareto optimal trial has {kl} <= {max_kl}; using all such trials"
            )

    rows: list[CandidateRow] = []
    for trial in sorted(candidates, key=lambda t: (t.values[primary], t.index)):
        primary_removal = removal(trial.values[primary], primary_baseline)
        holdout_removal = removal(trial.values[holdout], holdout_baseline)
        if primary_removal is None or holdout_removal is None:
            continue
        rows.append(
            CandidateRow(
                index=trial.index,
                primary=trial.values[primary],
                holdout=trial.values[holdout],
                kl=trial.values.get(kl),
                primary_removal=primary_removal,
                holdout_removal=holdout_removal,
            )
        )

    median_gap = statistics.median(row.gap for row in rows) if rows else None
    correlation = spearman(
        [t.values[primary] for t in scored],
        [t.values[holdout] for t in scored],
    )

    reasons: list[str] = []
    if median_gap is not None and median_gap > gap_threshold:
        reasons.append(
            f"median removal gap {median_gap:.2f} > {gap_threshold} on the candidate trials"
        )
    if correlation is not None and correlation < min_correlation:
        reasons.append(
            f"Spearman correlation of the two refusal rates {correlation:.2f} < {min_correlation} over {len(scored)} trials"
        )

    if not rows:
        warnings.append(
            "No candidate trial could be compared (zero baselines or none within KL)"
        )

    return GeneralizationReport(
        primary=primary,
        holdout=holdout,
        primary_baseline=primary_baseline,
        holdout_baseline=holdout_baseline,
        rows=rows,
        median_gap=median_gap,
        correlation=correlation,
        reasons=reasons,
        warnings=warnings,
    )


def load_trials(checkpoint_file: str) -> list[TrialScores]:
    """Reads the completed trials of a Heretic study checkpoint (checkpoints/<model>.jsonl)."""
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    from optuna.trial import TrialState

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    backend = JournalFileBackend(
        checkpoint_file, lock_obj=JournalFileOpenLock(checkpoint_file)
    )
    storage = JournalStorage(backend)
    summaries = storage.get_all_studies()
    if not summaries:
        raise ValueError(f"{checkpoint_file} contains no study")
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)

    pareto = {trial.number for trial in study.best_trials}
    trials: list[TrialScores] = []
    for trial in study.get_trials(states=(TrialState.COMPLETE,)):
        records = trial.user_attrs.get("scores", [])
        trials.append(
            TrialScores(
                index=trial.user_attrs.get("index", trial.number + 1),
                values={r["name"]: float(r["score"]["value"]) for r in records},
                baselines={r["name"]: float(r["baseline"]["value"]) for r in records},
                pareto=trial.number in pareto,
            )
        )
    return trials


def print_report(report: GeneralizationReport) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console(highlight=False)
    console.print(
        f"Primary: [bold]{report.primary}[/] (baseline {report.primary_baseline:.2f}), "
        f"held-out: [bold]{report.holdout}[/] (baseline {report.holdout_baseline:.2f})"
    )

    table = Table()
    for column in (
        "Trial",
        "Primary",
        "Held-out",
        "KL",
        "Removed",
        "Removed (held-out)",
        "Gap",
    ):
        table.add_column(column, justify="right")
    for row in report.rows:
        table.add_row(
            str(row.index),
            f"{row.primary:.2f}",
            f"{row.holdout:.2f}",
            "-" if row.kl is None else f"{row.kl:.4f}",
            f"{row.primary_removal:.0%}",
            f"{row.holdout_removal:.0%}",
            f"{row.gap:+.2f}",
        )
    console.print(table)

    if report.median_gap is not None:
        console.print(f"Median removal gap: [bold]{report.median_gap:+.2f}[/]")
    if report.correlation is not None:
        console.print(
            f"Spearman correlation over all trials: [bold]{report.correlation:.2f}[/]"
        )
    for warning in report.warnings:
        console.print(f"[yellow]Warning: {warning}[/]")

    if report.overfit:
        console.print(
            "[red]The refusal reduction does not carry over to the held-out set[/] ("
            + "; ".join(report.reasons)
            + "). Make the held-out KeywordRate an objective "
            '(optimization = "minimize") and start a new study.'
        )
    else:
        console.print(
            "[green]No sign that the optimizer overfits the evaluation prompts.[/] "
            'Keep the held-out KeywordRate at optimization = "none".'
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="heretic-generalization",
        description=(
            "Check whether the refusal reduction of a Heretic study carries over to a held-out "
            "KeywordRate instance (e.g. OR-Bench-hard), using the study checkpoint."
        ),
    )
    parser.add_argument(
        "checkpoint", help="Study checkpoint, e.g. checkpoints/Qwen--Qwen3-8B.jsonl"
    )
    parser.add_argument(
        "--primary",
        default=DEFAULT_PRIMARY,
        help="Score name of the optimized refusal rate",
    )
    parser.add_argument(
        "--holdout",
        default=None,
        help='Score name of the held-out refusal rate (default: the only score named "<primary> - <instance>")',
    )
    parser.add_argument(
        "--kl", default=DEFAULT_KL, help="Score name of the KL divergence"
    )
    parser.add_argument(
        "--max-kl",
        type=float,
        default=0.15,
        help="Largest KL divergence of a candidate trial",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.2,
        help="Median removal gap that counts as overfitting",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.5,
        help="Spearman correlation below which picking does not transfer",
    )
    args = parser.parse_args(argv)

    report = analyze(
        load_trials(args.checkpoint),
        primary=args.primary,
        holdout=args.holdout,
        kl=args.kl,
        max_kl=args.max_kl,
        gap_threshold=args.gap_threshold,
        min_correlation=args.min_correlation,
    )
    print_report(report)
    return 1 if report.overfit else 0


if __name__ == "__main__":
    raise SystemExit(main())
