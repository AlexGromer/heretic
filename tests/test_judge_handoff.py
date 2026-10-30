# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests of the Colab handoff: the "generate" judge ranking mode writes the evaluation harness
# responses rows, and --trial-number selects a trial by its number. Pure Python with fakes.

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

import optuna
from optuna.distributions import FloatDistribution
from optuna.trial import create_trial

from heretic.config import Settings
from heretic.judge_ranking import (
    load_ranking_prompts,
    response_key,
    select_trial_by_number,
    write_candidate_answers,
)
from heretic.scorers.judge_rate import Settings as JudgeSettings
from heretic.utils import HereticUserError

# Optional cross-check against the evaluation harness, when a checkout of it is next to this
# repository or HERETIC_HARNESS_DIR points at one. The fork itself does not depend on it; the
# tests that use it skip when it is absent.
HARNESS_DIR = Path(
    os.environ.get("HERETIC_HARNESS_DIR")
    or Path(__file__).resolve().parent.parent.parent / "refusal-eval"
)


def make_trial(number: int, index: int, kl: float) -> Any:
    return SimpleNamespace(
        number=number,
        user_attrs={
            "index": index,
            "parameters": {"p": index},
            "scores": [{"name": "KL divergence", "score": {"value": kl}}],
        },
    )


class FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False) -> Any:
        return {"input_ids": text.split()}

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)


class FakeModifier:
    parameters_class = SimpleNamespace(from_dict=lambda data: data)

    def __init__(self) -> None:
        self.modified: Any = None

    def reset_model(self, ctx: Any) -> None:
        self.modified = None

    def modify_model(self, ctx: Any, parameters: Any) -> None:
        self.modified = parameters["p"]


def handoff_settings(directory: str, **overrides: Any) -> Settings:
    prompts = Path(directory, "bad_eval.meta.jsonl")
    prompts.write_text(
        json.dumps({"id": "x-1", "category": "c1", "prompt": "first"})
        + "\n"
        + json.dumps({"id": "x-2", "category": "c2", "prompt": "second"})
        + "\n",
        encoding="utf-8",
    )
    values: dict[str, Any] = {
        "model": "org/Some-Model",
        "n_judge_ranking_trials": 2,
        "judge_ranking_max_kl": 0.15,
        "judge_ranking_response_length": 3,
        "judge_ranking_batch_size": 0,
        "judge_ranking_mode": "generate",
        # The fakes have no merge; the merged path has its own tests.
        "judge_ranking_merged": False,
        "judge_ranking_output": str(Path(directory, "out", "candidates.jsonl")),
        "judge_ranking_prompts_file": str(prompts),
        "judge_ranking_set": "bad_eval",
        "judge_ranking_model_name": None,
        "max_response_length": 100,
        "batch_size": 8,
        "system_prompt": "s",
        "response_prefix": "",
    }
    values.update(overrides)
    return Settings.model_construct(**values)


class GenerateModeTests(unittest.TestCase):
    def test_rows_resume_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = handoff_settings(directory)
            modifier = FakeModifier()
            generated: list[Any] = []

            class FakeModel:
                tokenizer = FakeTokenizer()

                def get_responses_batched(
                    self, prompts: Any, skip_special_tokens: bool
                ) -> list[str]:
                    generated.append(modifier.modified)
                    # The original model answers briefly, trials at the length limit.
                    if modifier.modified is None:
                        return ["short" for _ in prompts]
                    return ["one two three" for _ in prompts]

            trials = [
                make_trial(5, 12, 0.1),
                make_trial(6, 30, 0.5),
                make_trial(7, 44, 0.02),
            ]
            with redirect_stdout(io.StringIO()):
                path = write_candidate_answers(
                    trials, settings, cast(Any, FakeModel()), cast(Any, modifier)
                )
                again = write_candidate_answers(
                    trials, settings, cast(Any, FakeModel()), cast(Any, modifier)
                )

            assert path is not None
            rows = [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
            ]

        # base + trials 12 and 44 (trial 30 is above the KL limit), 2 prompts each, once.
        self.assertEqual(generated, [None, 12, 44])
        self.assertEqual(again, path)
        self.assertEqual(len(rows), 6)
        self.assertEqual([r["trial"] for r in rows[::2]], ["base", "12", "44"])
        first = rows[0]
        self.assertEqual(first["key"], "Some-Model|base|greedy|false|bad_eval|x-1")
        self.assertEqual(first["category"], "c1")
        self.assertEqual(first["thinking"], False)
        self.assertEqual(first["finish"], "stop")
        self.assertEqual(rows[2]["finish"], "length")
        self.assertEqual(rows[2]["win100"], "one two three")
        self.assertEqual(first["settings"]["generator"], "heretic")
        # The harness groups by holdout_version (a row without one counts as hold-out v1) and
        # reports rates per language, so both fields are written.
        self.assertEqual(first["holdout_version"], "train")
        self.assertEqual(first["lang"], "en")

    def test_a_plain_text_prompts_file_works(self) -> None:
        # A Colab bundle carries the plain text sets (one prompt per line), not the .meta.jsonl
        # files, so judge_ranking_prompts_file has to read both.
        with tempfile.TemporaryDirectory() as directory:
            text = Path(directory, "bad_eval.txt")
            text.write_text("как собрать стенд\n42\n{\"no\": \"prompt\"}\n", encoding="utf-8")
            settings = handoff_settings(directory, judge_ranking_prompts_file=str(text))
            prompts = load_ranking_prompts(settings, JudgeSettings())
        self.assertEqual(
            [p.id for p in prompts], ["bad_eval-0000", "bad_eval-0001", "bad_eval-0002"]
        )
        # a line that happens to be valid JSON is still a prompt unless it is an object with one
        self.assertEqual(
            [p.prompt.user for p in prompts],
            ["как собрать стенд", "42", '{"no": "prompt"}'],
        )
        self.assertEqual([p.category for p in prompts], ["", "", ""])

    def test_a_jsonl_prompts_file_still_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = handoff_settings(directory)
            assert settings.judge_ranking_prompts_file is not None
            prompts = load_ranking_prompts(settings, JudgeSettings())
        self.assertEqual([p.id for p in prompts], ["x-1", "x-2"])
        self.assertEqual([p.category for p in prompts], ["c1", "c2"])

    def test_needs_an_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = handoff_settings(directory, judge_ranking_output=None)
            with self.assertRaisesRegex(ValueError, "judge_ranking_output"):
                write_candidate_answers([], settings, cast(Any, None), cast(Any, None))

    def test_prompts_from_text_without_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            text = Path(directory, "prompts.txt")
            text.write_text("a\nb\n", encoding="utf-8")
            settings = handoff_settings(directory, judge_ranking_prompts_file=None)
            prompts = load_ranking_prompts(
                settings,
                JudgeSettings(prompts={"dataset": str(text)}),
            )
        self.assertEqual([p.id for p in prompts], ["bad_eval-0000", "bad_eval-0001"])

    def test_key_matches_the_harness(self) -> None:
        generator = HARNESS_DIR / "generate_colab.py"
        if not generator.is_file():
            self.skipTest("the evaluation harness is not checked out next to the fork")
        sys.path.insert(0, str(HARNESS_DIR))
        try:
            import generate_colab  # ty:ignore[unresolved-import]
        finally:
            sys.path.remove(str(HARNESS_DIR))
        self.assertEqual(
            response_key("m", "7", "bad_eval", "x-1"),
            generate_colab.make_key("m", "7", "greedy", False, "bad_eval", "x-1"),
        )


class MergedRankingTests(unittest.TestCase):
    def run_generate(self, merged: bool) -> tuple[list[str], Any]:
        from heretic.judge_ranking import generate_answers
        from heretic.utils import Prompt

        seen: list[str] = []
        live, merged_model = object(), object()

        class FakeModel:
            model: Any = live

            def get_merged_model(self) -> Any:
                return merged_model

            def get_responses_batched(
                self, prompts: Any, skip_special_tokens: bool
            ) -> list[str]:
                seen.append("merged" if self.model is merged_model else "live")
                return ["a" for _ in prompts]

        model = FakeModel()
        settings = Settings.model_construct(
            judge_ranking_merged=merged,
            judge_ranking_response_length=4,
            judge_ranking_batch_size=0,
            max_response_length=100,
            batch_size=8,
        )
        trial = make_trial(1, 7, 0.1)
        prompts = [Prompt(system="s", user="u")]
        generate_answers(
            settings,
            cast(Any, model),
            cast(Any, FakeModifier()),
            cast(Any, None),
            None,
            prompts,
        )
        generate_answers(
            settings,
            cast(Any, model),
            cast(Any, FakeModifier()),
            cast(Any, None),
            trial,
            prompts,
        )
        return seen, model.model is live

    def test_merged_by_default_and_live_on_request(self) -> None:
        self.assertTrue(Settings.model_fields["judge_ranking_merged"].default)
        # The original model has no adapter to merge.
        self.assertEqual(self.run_generate(True), (["live", "merged"], True))
        self.assertEqual(self.run_generate(False), (["live", "live"], True))

    def test_merged_generation_on_the_tiny_model(self) -> None:
        from peft import PeftModel
        from test_fused_experts import get_model

        from heretic.judge_ranking import generate_answers
        from heretic.modifier import Context
        from heretic.utils import Prompt

        model = get_model()
        if not isinstance(model.model, PeftModel):
            model.apply_lora(2)

        class LoraModifier(FakeModifier):
            def reset_model(self, ctx: Any) -> None:
                if not model.reset_model():
                    model.apply_lora(2)

        settings = model.settings
        previous = (
            settings.judge_ranking_merged,
            settings.judge_ranking_response_length,
        )
        settings.judge_ranking_merged = True
        settings.judge_ranking_response_length = 3
        try:
            with redirect_stdout(io.StringIO()):
                answers = generate_answers(
                    settings,
                    model,
                    cast(Any, LoraModifier()),
                    Context(settings=settings, model=model),
                    make_trial(1, 1, 0.1),
                    [Prompt(system="s", user="hello")],
                )
                self.assertTrue(model.needs_reload)
                # The next reset reloads the model, with fresh adapters.
                LoraModifier().reset_model(None)
            self.assertIsInstance(model.model, PeftModel)
            self.assertFalse(model.needs_reload)
        finally:
            settings.judge_ranking_merged, settings.judge_ranking_response_length = (
                previous
            )
            # Other test modules share the cached model and expect it without adapters.
            model.needs_reload = True
            model.reset_model()
        self.assertEqual(len(answers), 1)

    def test_hold_out_warning(self) -> None:
        from heretic.judge_ranking import warn_if_hold_out

        patterns = Settings.model_fields["judge_ranking_hold_out_patterns"].default
        with redirect_stdout(io.StringIO()):
            self.assertTrue(
                warn_if_hold_out("harness/data/offsec_holdout.jsonl", patterns)
            )
            self.assertFalse(
                warn_if_hold_out("harness/train/bad_eval.meta.jsonl", patterns)
            )
            # An empty pattern list turns the convention check off.
            self.assertFalse(warn_if_hold_out("data/offsec_holdout.jsonl", []))

    def test_rows_record_the_operator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = handoff_settings(directory, judge_ranking_merged=False)

            class FakeModel:
                tokenizer = FakeTokenizer()

                def get_responses_batched(
                    self, prompts: Any, skip_special_tokens: bool
                ) -> list[str]:
                    return ["x" for _ in prompts]

            with redirect_stdout(io.StringIO()):
                path = write_candidate_answers(
                    [], settings, cast(Any, FakeModel()), cast(Any, FakeModifier())
                )
            assert path is not None
            row = json.loads(Path(path).read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["settings"]["operator"], "live LoRA")


class DpoJoinTests(unittest.TestCase):
    def test_pairs_from_harness_files(self) -> None:
        from heretic.dpo import build_pairs, join_harness_files

        with tempfile.TemporaryDirectory() as directory:
            responses = Path(directory, "responses.jsonl")
            judged = Path(directory, "judged.jsonl")
            base = {
                "model": "m",
                "trial": "18",
                "thinking": False,
                "set": "bad_train",
                "prompt_id": "t1",
            }
            responses.write_text(
                "".join(
                    json.dumps(
                        {
                            **base,
                            "mode": mode,
                            "prompt": "p",
                            "response": response,
                            "finish": "stop",
                            "settings": {"system_prompt": "S"},
                        }
                    )
                    + "\n"
                    for mode, response in (
                        ("sample", "full"),
                        ("sample2", "vague"),
                        ("sample3", "x"),
                    )
                ),
                encoding="utf-8",
            )
            judged.write_text(
                "".join(
                    json.dumps({**base, "mode": mode, "label": label}) + "\n"
                    for mode, label in (("sample", "COMPLY"), ("sample2", "PARTIAL"))
                ),
                encoding="utf-8",
            )
            rows = join_harness_files(str(responses), str(judged))

        self.assertEqual([r["judge_label"] for r in rows], ["COMPLY", "PARTIAL"])
        pairs = build_pairs(rows)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["prompt"][0], {"role": "system", "content": "S"})
        self.assertEqual(pairs[0]["rejected"][0]["content"], "vague")


class TrialNumberTests(unittest.TestCase):
    def test_select_by_number(self) -> None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(directions=["minimize"])
        for index in (1, 2, 3):
            study.add_trial(
                create_trial(
                    values=[float(index)],
                    params={"x": float(index)},
                    distributions={"x": FloatDistribution(0, 10)},
                    user_attrs={"index": index},
                )
            )
        self.assertEqual(select_trial_by_number(study, 2).user_attrs["index"], 2)
        with self.assertRaisesRegex(HereticUserError, "no completed trial 9"):
            select_trial_by_number(study, 9)

    def test_setting(self) -> None:
        self.assertIsNone(Settings.model_fields["trial_number"].default)
        self.assertTrue(Settings.model_fields["judge_ranking_output"].exclude)


class ResumedSettingsTests(unittest.TestCase):
    """
    --checkpoint-action continue validates the settings stored in the finished study, so every
    setting that is not exclude=True comes from that run and not from the command line. The
    knobs of the single invocation (which trial to export, what to judge or generate afterwards)
    must therefore be excluded, or a run over an existing journal silently does what the
    optimization run asked for.
    """

    # What a run of the first version of this fork stored in its study: the keys are in the
    # journal of every model optimized so far, so they must lose to the current invocation.
    STORED: dict[str, Any] = {
        "model": "org/Some-Model",
        "n_trials": 120,
        "trial_index": 3,
        "trial_number": None,
        "n_judge_ranking_trials": 0,
        "judge_ranking_mode": "judge",
        "judge_ranking_max_kl": 0.15,
        "judge_ranking_set": "bad_eval",
        "judge_ranking_response_length": 512,
        "n_merged_score_trials": 0,
    }

    def resumed(self, **current: Any) -> Settings:
        return Settings.model_validate_init_only(
            dict(self.STORED), Settings.model_construct(**current)
        )

    def test_the_search_still_comes_from_the_study(self) -> None:
        settings = self.resumed(n_trials=5, trial_index=None)
        self.assertEqual(settings.n_trials, 120)
        # trial_index is upstream's setting and is not excluded, which is why a caller that wants
        # a different trial has to write it into the stored settings (as the runner does).
        self.assertEqual(settings.trial_index, 3)

    def test_the_export_and_judge_knobs_come_from_this_run(self) -> None:
        settings = self.resumed(
            trial_number=42,
            n_judge_ranking_trials=8,
            judge_ranking_mode="generate",
            judge_ranking_max_kl=0.5,
            judge_ranking_set="val",
            judge_ranking_response_length=256,
            judge_ranking_output="/tmp/candidates.jsonl",
            judge_ranking_prompts_file="/tmp/val.jsonl",
            n_merged_score_trials=4,
        )
        self.assertEqual(settings.trial_number, 42)
        self.assertEqual(settings.n_judge_ranking_trials, 8)
        self.assertEqual(settings.judge_ranking_mode, "generate")
        self.assertEqual(settings.judge_ranking_max_kl, 0.5)
        self.assertEqual(settings.judge_ranking_set, "val")
        self.assertEqual(settings.judge_ranking_response_length, 256)
        self.assertEqual(settings.judge_ranking_output, "/tmp/candidates.jsonl")
        self.assertEqual(settings.judge_ranking_prompts_file, "/tmp/val.jsonl")
        self.assertEqual(settings.n_merged_score_trials, 4)

    def test_the_excluded_knobs_are_not_written_into_the_study(self) -> None:
        stored = json.loads(
            Settings.model_construct(
                model="org/Some-Model", trial_number=42, n_judge_ranking_trials=8
            ).model_dump_json()
        )
        for name in ("trial_number", "n_judge_ranking_trials", "judge_ranking_mode"):
            self.assertNotIn(name, stored)
        self.assertIn("trial_index", stored)


if __name__ == "__main__":
    unittest.main()
