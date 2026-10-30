# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests of the v4 additions: prompt sampling (OR-Bench), the generalization check,
# the responses file, and the fixes for upstream issues #487, #522, #533 and #535.
# CPU only; the ARA tests use the tiny model of test_fused_experts.

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import optuna
import torch
from optuna.distributions import FloatDistribution
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import create_trial
from peft import PeftModel

from heretic.config import Settings, SingleDatasetSpecification
from heretic.generalization import (
    TrialScores,
    analyze,
    detect_holdout,
    load_trials,
    main,
    removal,
    spearman,
)
from heretic.model import check_not_gguf_only, list_model_files
from heretic.modifier import Context
from heretic.modifiers.ara import ARA, lowrank_frobenius_sq
from heretic.modifiers.ara import Parameters as ARAParameters
from heretic.modifiers.ara import Settings as ARASettings
from heretic.scorers.keyword_rate import KeywordRate
from heretic.scorers.keyword_rate import Settings as KeywordRateSettings
from heretic.utils import (
    HereticUserError,
    Prompt,
    ask,
    ask_if_unset,
    get_cot_prefix_extension,
    load_prompts,
    sample_prompts,
)


class SamplingTests(unittest.TestCase):
    def test_sample_is_reproducible_and_keeps_order(self) -> None:
        prompts = [f"p{i}" for i in range(50)]
        first = sample_prompts(prompts, 10, seed=0)
        self.assertEqual(first, sample_prompts(prompts, 10, seed=0))
        self.assertNotEqual(first, sample_prompts(prompts, 10, seed=1))
        self.assertEqual(len(set(first)), 10)
        self.assertEqual(first, sorted(first, key=prompts.index))

    def test_sample_covers_categories_of_a_sorted_dataset(self) -> None:
        # Like OR-Bench: 10 categories of 100 prompts each, sorted by category.
        prompts = [f"{category}-{i}" for category in range(10) for i in range(100)]
        drawn = sample_prompts(prompts, 100, seed=0)
        categories = {prompt.split("-")[0] for prompt in drawn}
        self.assertEqual(len(categories), 10)

    def test_sample_larger_than_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "larger than the 3 prompts"):
            sample_prompts(["a", "b", "c"], 4, seed=0)

    def test_text_file_with_split_and_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "prompts.txt")
            path.write_text("\n".join(f"line {i}" for i in range(20)), encoding="utf-8")
            specification = SingleDatasetSpecification(
                dataset=str(path),
                split="[:10]",
                sample_size=4,
                sample_seed=3,
            )
            settings = cast(Settings, SimpleNamespace(system_prompt="sys"))
            prompts = load_prompts(settings, specification)

        self.assertEqual(len(prompts), 4)
        self.assertTrue(all(int(p.user.split()[1]) < 10 for p in prompts))
        self.assertEqual(
            [p.user for p in prompts],
            sample_prompts([f"line {i}" for i in range(10)], 4, seed=3),
        )

    def test_sample_size_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            SingleDatasetSpecification(dataset="x", sample_size=0)


def trial(index: int, primary: float, holdout: float, kl: float, pareto: bool = True):
    return TrialScores(
        index=index,
        values={
            "Refusals": primary,
            "Refusals - orbench": holdout,
            "KL divergence": kl,
        },
        baselines={"Refusals": 0.9, "Refusals - orbench": 0.5, "KL divergence": 0.0},
        pareto=pareto,
    )


class GeneralizationTests(unittest.TestCase):
    def test_helpers(self) -> None:
        self.assertAlmostEqual(cast(float, removal(0.3, 0.9)), 2 / 3)
        self.assertIsNone(removal(0.1, 0.0))
        self.assertAlmostEqual(
            cast(float, spearman([1, 2, 3, 4], [10, 20, 30, 40])), 1.0
        )
        self.assertAlmostEqual(cast(float, spearman([1, 2, 3, 4], [4, 3, 2, 1])), -1.0)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(spearman([1, 2], [1, 2]))
        # Ties get average ranks.
        self.assertAlmostEqual(cast(float, spearman([1, 1, 2], [1, 1, 2])), 1.0)

    def test_detect_holdout(self) -> None:
        self.assertEqual(
            detect_holdout([trial(1, 0.1, 0.1, 0.1)], "Refusals"), "Refusals - orbench"
        )
        two = TrialScores(
            1, {"Refusals": 0, "Refusals - a": 0, "Refusals - b": 0}, {}, True
        )
        with self.assertRaisesRegex(ValueError, "--holdout"):
            detect_holdout([two], "Refusals")

    def test_transferring_reduction_is_not_overfitting(self) -> None:
        # The held-out rate falls along with the primary one.
        trials = [
            trial(i, primary, primary * 0.5 / 0.9, 0.02 * i)
            for i, primary in enumerate([0.9, 0.6, 0.4, 0.2, 0.1, 0.05], start=1)
        ]
        report = analyze(trials)
        self.assertFalse(report.overfit, report.reasons)
        self.assertAlmostEqual(cast(float, report.correlation), 1.0)
        self.assertAlmostEqual(cast(float, report.median_gap), 0.0, places=6)

    def test_reduction_that_does_not_transfer_is_overfitting(self) -> None:
        # The primary rate falls to almost zero, the held-out rate stays near its baseline.
        trials = [
            trial(i, primary, holdout, 0.02 * i)
            for i, (primary, holdout) in enumerate(
                [
                    (0.9, 0.5),
                    (0.5, 0.45),
                    (0.3, 0.5),
                    (0.1, 0.48),
                    (0.03, 0.46),
                    (0.02, 0.5),
                ],
                start=1,
            )
        ]
        report = analyze(trials)
        self.assertTrue(report.overfit)
        self.assertTrue(
            any("median removal gap" in reason for reason in report.reasons)
        )

    def test_candidates_respect_max_kl_and_pareto(self) -> None:
        trials = [
            trial(1, 0.1, 0.1, 0.5),  # too much KL
            trial(2, 0.2, 0.2, 0.1),
            trial(3, 0.3, 0.3, 0.1, pareto=False),
        ]
        report = analyze(trials, max_kl=0.15)
        self.assertEqual([row.index for row in report.rows], [2])

    def test_low_holdout_baseline_warns(self) -> None:
        trials = [
            TrialScores(
                i,
                {"Refusals": 0.1 * i, "Refusals - orbench": 0.01, "KL divergence": 0.0},
                {"Refusals": 0.9, "Refusals - orbench": 0.02, "KL divergence": 0.0},
                True,
            )
            for i in range(1, 4)
        ]
        report = analyze(trials)
        self.assertTrue(any("hardly triggers refusals" in w for w in report.warnings))

    def test_checkpoint_round_trip_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.jsonl")
            storage = JournalStorage(
                JournalFileBackend(path, lock_obj=JournalFileOpenLock(path))
            )
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                storage=storage,
                study_name="heretic",
                directions=["minimize", "minimize"],
            )
            values = [
                (0.9, 0.5, 0.0),
                (0.3, 0.5, 0.05),
                (0.05, 0.49, 0.1),
                (0.02, 0.5, 0.12),
            ]
            for index, (primary, holdout, kl) in enumerate(values, start=1):
                records = [
                    {
                        "name": name,
                        "score": {"value": value},
                        "baseline": {"value": base},
                    }
                    for name, value, base in (
                        ("Refusals", primary, 0.9),
                        ("Refusals - orbench", holdout, 0.5),
                        ("KL divergence", kl, 0.0),
                    )
                ]
                study.add_trial(
                    create_trial(
                        values=[primary, kl],
                        params={"x": float(index)},
                        distributions={"x": FloatDistribution(0, 10)},
                        user_attrs={"index": index, "scores": records},
                    )
                )

            trials = load_trials(path)
            self.assertEqual([t.index for t in trials], [1, 2, 3, 4])
            self.assertTrue(all(t.pareto for t in trials))
            self.assertEqual(trials[2].values["Refusals - orbench"], 0.49)

            with redirect_stdout(io.StringIO()):
                exit_code = main([path])
            self.assertEqual(exit_code, 1)


class FakeContext:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def get_responses(self, prompts: list[Prompt]) -> list[str]:
        return self.responses


class ResponsesFileTests(unittest.TestCase):
    def test_every_evaluation_appends_its_responses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sub", "responses.jsonl")
            scorer = KeywordRate(
                heretic_settings=Settings.model_construct(responses_file=path),
                settings=KeywordRateSettings(score_name="Refusals"),
            )
            scorer.prompts = [
                Prompt(system="s", user="u1"),
                Prompt(system="s", user="u2"),
            ]
            scorer._evaluation_count = 0

            ctx = cast(Context, FakeContext(["I'm sorry, no.", "Sure, here it is."]))
            scorer.get_baseline_score(ctx)
            score = scorer.get_score(ctx)

            with open(path, encoding="utf-8") as file:
                lines = [json.loads(line) for line in file]

        self.assertEqual(score.value, 0.5)
        self.assertEqual([line["evaluation"] for line in lines], [0, 0, 1, 1])
        self.assertEqual([line["match"] for line in lines], [True, False, True, False])
        self.assertEqual(lines[1]["prompt"], "u2")
        self.assertEqual(lines[1]["response"], "Sure, here it is.")
        self.assertEqual(lines[0]["score_name"], "Refusals")
        # refusal-eval's responses format, so its judge can label the file directly.
        self.assertEqual(lines[0]["trial"], "base")
        self.assertEqual(lines[2]["trial"], "e1")
        self.assertEqual(
            lines[3]["key"].split("|")[1:],
            ["e1", "greedy", "false", "Refusals", "0001"],
        )

    def test_no_file_without_setting(self) -> None:
        scorer = KeywordRate(
            heretic_settings=Settings.model_construct(),
            settings=KeywordRateSettings(),
        )
        scorer.prompts = [Prompt(system="s", user="u")]
        with tempfile.TemporaryDirectory() as directory:
            previous = os.getcwd()
            os.chdir(directory)
            try:
                scorer.get_score(cast(Context, FakeContext(["ok"])))
                self.assertEqual(os.listdir(directory), [])
            finally:
                os.chdir(previous)

    def test_responses_file_is_not_serialized(self) -> None:
        self.assertTrue(Settings.model_fields["responses_file"].exclude)


class CotPrefixExtensionTests(unittest.TestCase):
    def test_whitespace_extends_the_prefix(self) -> None:
        self.assertEqual(
            get_cot_prefix_extension(["\n\nSure, here", "\n\nOf course"]), "\n\n"
        )

    def test_nothing_common(self) -> None:
        self.assertEqual(get_cot_prefix_extension(["Yes.", "No."]), "")

    def test_reasoning_text_is_not_appended(self) -> None:
        # Qwen3-style self-generated CoT (#487): the common prefix is the start of the
        # reasoning. Only the leading whitespace may extend the prefix.
        responses = [
            f"\nOkay, the user asks about {topic}." for topic in ("cats", "dogs")
        ]
        self.assertEqual(get_cot_prefix_extension(responses), "\n")

    def test_trailing_spaces_are_dropped(self) -> None:
        self.assertEqual(get_cot_prefix_extension(["\n  a", "\n  b"]), "\n")


class GgufTests(unittest.TestCase):
    def test_gguf_only_is_rejected(self) -> None:
        with self.assertRaisesRegex(HereticUserError, "only GGUF files"):
            check_not_gguf_only("org/model-gguf", ["README.md", "model-Q4_0.gguf"])

    def test_safetensors_with_gguf_is_accepted(self) -> None:
        check_not_gguf_only("org/model", ["model.safetensors", "model-Q4_0.gguf"])
        check_not_gguf_only("org/model", ["pytorch_model.bin"])
        check_not_gguf_only("org/model", None)
        check_not_gguf_only("org/model", [])

    def test_local_directory_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "model.gguf").write_bytes(b"GGUF")
            files = list_model_files(directory)
            self.assertEqual(files, ["model.gguf"])
            with self.assertRaises(HereticUserError):
                check_not_gguf_only(directory, files)
            self.assertEqual(
                list_model_files(os.path.join(directory, "model.gguf")), ["model.gguf"]
            )


class NonInteractiveTests(unittest.TestCase):
    def test_question_without_terminal_names_the_setting(self) -> None:
        question = SimpleNamespace(ask=lambda: self.fail("must not ask"))
        with patch.object(sys, "stdin", io.StringIO("")):
            with self.assertRaisesRegex(HereticUserError, "--trial-index"):
                ask_if_unset(None, cast(Any, question), setting="trial_index")

    def test_question_without_setting_uses_the_hint(self) -> None:
        question = SimpleNamespace(ask=lambda: self.fail("must not ask"))
        with patch.object(sys, "stdin", io.StringIO("")):
            with self.assertRaisesRegex(HereticUserError, "HF_TOKEN"):
                ask(cast(Any, question), hint="Set the HF_TOKEN environment variable.")
            with self.assertRaisesRegex(HereticUserError, "in a terminal"):
                ask(cast(Any, question))

    def test_preset_value_needs_no_terminal(self) -> None:
        question = SimpleNamespace(ask=lambda: self.fail("must not ask"))
        with patch.object(sys, "stdin", io.StringIO("")):
            self.assertEqual(
                ask_if_unset(3, cast(Any, question), setting="trial_index"), 3
            )

    def test_terminal_asks(self) -> None:
        question = SimpleNamespace(ask=lambda: "answer")
        terminal = SimpleNamespace(isatty=lambda: True)
        with patch.object(sys, "stdin", terminal):
            self.assertEqual(ask_if_unset(None, cast(Any, question)), "answer")


class RidgeTests(unittest.TestCase):
    def test_lowrank_frobenius_matches_dense(self) -> None:
        torch.manual_seed(0)
        A = torch.randn(6, 40, dtype=torch.float64)
        B = torch.randn(30, 6, dtype=torch.float64)
        dense = torch.linalg.matrix_norm(B @ A) ** 2
        self.assertTrue(torch.allclose(lowrank_frobenius_sq(A, B), dense))

    def test_settings(self) -> None:
        settings = ARASettings()
        self.assertEqual(settings.ridge_weight, 0.0)
        self.assertEqual(settings.line_search, "strong_wolfe")
        with self.assertRaises(ValueError):
            ARASettings(ridge_weight=-1)
        with self.assertRaises(ValueError):
            ARASettings(line_search="backtracking")  # ty:ignore[invalid-argument-type]

    def run_ara(self, **settings: Any) -> float:
        # Imported here: loading the tiny model takes a few seconds.
        from test_fused_experts import get_lora_b, get_model

        model = get_model()
        # The cached model keeps its adapters across tests (reset only zeroes them).
        if not isinstance(model.model, PeftModel):
            model.apply_lora(4)
        good = [Prompt(system="s", user=u) for u in ("a", "bb bb", "ccc ccc ccc")]
        bad = [Prompt(system="s", user=u) for u in ("d d", "eee", "ff ff ff ff")]
        with redirect_stdout(io.StringIO()):
            good_io = model.get_module_io_batched(good)
            bad_io = model.get_module_io_batched(bad)

        modifier = ARA.__new__(ARA)
        model.settings.seed = 1
        modifier.heretic_settings = model.settings
        modifier.settings = ARASettings(
            lora_rank=4, n_optimization_steps=2, max_iter=5, **settings
        )
        modifier.good_module_io = good_io
        modifier.bad_module_io = bad_io
        layer_count = len(model.get_layers())
        try:
            with redirect_stdout(io.StringIO()):
                modifier.modify_model(
                    cast(Context, SimpleNamespace(get_model=lambda: model)),
                    ARAParameters(0, layer_count, 0.5, 0.5, 1.0, 1),
                )
        finally:
            model.settings.seed = None

        total = 0.0
        for layer_index in range(layer_count):
            for modules in model.get_layer_modules(layer_index).values():
                for module in modules:
                    lora_a = cast(Any, module).lora_A["default"].weight
                    lora_b = get_lora_b(module)
                    self.assertTrue(torch.isfinite(lora_a).all())
                    self.assertTrue(torch.isfinite(lora_b).all())
                    total += lowrank_frobenius_sq(
                        lora_a.detach().double(), lora_b.detach().double()
                    ).item()
        return total

    def test_ridge_shrinks_the_edit(self) -> None:
        unregularized = self.run_ara()
        regularized = self.run_ara(ridge_weight=1000.0)
        self.assertGreater(unregularized, 0)
        self.assertLess(regularized, unregularized)

    def test_without_line_search(self) -> None:
        self.assertGreater(self.run_ara(line_search="none", learning_rate=0.1), 0)

    def test_divergence_leaves_modules_unmodified(self) -> None:
        # Without row normalization, huge steps make the loss non-finite.
        total = self.run_ara(
            line_search="none", learning_rate=1e30, preserve_row_magnitudes=False
        )
        self.assertEqual(total, 0.0)


if __name__ == "__main__":
    unittest.main()
