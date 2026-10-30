# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests of the Colab handoff: the "generate" judge ranking mode writes refusal-eval
# responses rows, and --trial-number selects a trial by its number. Pure Python with fakes.

import io
import json
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
    refusal_eval_key,
    select_trial_by_number,
    write_candidate_answers,
)
from heretic.scorers.judge_rate import Settings as JudgeSettings
from heretic.utils import HereticUserError

REFUSAL_EVAL = Path(__file__).resolve().parent.parent.parent / "refusal-eval"


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

    def test_key_matches_refusal_eval(self) -> None:
        generator = REFUSAL_EVAL / "generate_colab.py"
        if not generator.is_file():
            self.skipTest("refusal-eval is not checked out next to the fork")
        sys.path.insert(0, str(REFUSAL_EVAL))
        try:
            import generate_colab  # ty:ignore[unresolved-import]
        finally:
            sys.path.remove(str(REFUSAL_EVAL))
        self.assertEqual(
            refusal_eval_key("m", "7", "bad_eval", "x-1"),
            generate_colab.make_key("m", "7", "greedy", False, "bad_eval", "x-1"),
        )


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


if __name__ == "__main__":
    unittest.main()
