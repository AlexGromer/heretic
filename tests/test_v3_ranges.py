# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests for the configurable search ranges of the Abliteration and ARA modifiers.
# Pure Python: a stub model, no downloads, no GPU.

import unittest
from types import SimpleNamespace
from typing import Any, cast

import optuna
from pydantic import ValidationError

from heretic.modifier import Context
from heretic.modifiers.abliteration import Abliteration
from heretic.modifiers.abliteration import Settings as AbliterationSettings
from heretic.modifiers.ara import ARA
from heretic.modifiers.ara import Settings as ARASettings

LAYER_COUNT = 12
LAST_LAYER_INDEX = LAYER_COUNT - 1

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Assertion helpers for the module-level (pytest) tests.
check = unittest.TestCase()
check.maxDiff = None


class StubModel:
    def get_layers(self) -> list[int]:
        return list(range(LAYER_COUNT))

    def get_abliterable_components(self) -> list[str]:
        return ["attn.o_proj", "mlp.down_proj"]


def make_context() -> Context:
    return cast(Context, SimpleNamespace(get_model=lambda: StubModel()))


def make_abliteration(settings: AbliterationSettings) -> Abliteration:
    modifier = Abliteration.__new__(Abliteration)
    modifier.settings = settings
    return modifier


def make_ara(settings: ARASettings) -> ARA:
    modifier = ARA.__new__(ARA)
    modifier.settings = settings
    modifier.good_prompt_count = 40
    modifier.bad_prompt_count = 40
    return modifier


class RecordingTrial:
    """Records every suggest_* call and returns the lower bound."""

    def __init__(self) -> None:
        self.calls: dict[str, tuple[Any, Any, bool]] = {}

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        self.calls[name] = (tuple(choices), None, False)
        return choices[0]

    def suggest_float(
        self, name: str, low: float, high: float, log: bool = False
    ) -> float:
        self.calls[name] = (low, high, log)
        return low

    def suggest_int(self, name: str, low: int, high: int) -> int:
        self.calls[name] = (low, high, False)
        return low


def sample(modifier: Any, count: int = 80) -> list[optuna.trial.FrozenTrial]:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

    def objective(trial: optuna.Trial) -> float:
        parameters = modifier.suggest_parameters(make_context(), trial)
        trial.set_user_attr("parameters", parameters.to_dict())
        return 0.0

    study.optimize(objective, n_trials=count)
    return study.trials


WIDE_ABLITERATION: dict[str, Any] = {
    "direction_index_range": (0.1, 1.0),
    "max_weight_range": (2.0, 4.0),
    "mlp_max_weight_range": (1.5, 3.0),
    "max_weight_position_range": (0.0, 1.0),
    "min_weight_range": (0.5, 1.0),
    "min_weight_distance_min_layers": 3.0,
    "min_weight_distance_max_fraction": 1.5,
}


class RangesTests(unittest.TestCase):
    def test_defaults_equal_v2_ranges(self) -> None:
        # The bounds that upstream Heretic (and v2 of this fork) hard-codes.
        trial = RecordingTrial()
        make_abliteration(AbliterationSettings()).suggest_parameters(
            make_context(), cast(Any, trial)
        )
        calls = trial.calls
        self.assertEqual(
            calls["direction_index"],
            (0.4 * LAST_LAYER_INDEX, 0.9 * LAST_LAYER_INDEX, False),
        )
        self.assertEqual(calls["attn.o_proj.max_weight"], (0.8, 1.5, False))
        self.assertEqual(calls["mlp.down_proj.max_weight"], (-0.25, 1.5, False))
        for component in ("attn.o_proj", "mlp.down_proj"):
            self.assertEqual(
                calls[f"{component}.max_weight_position"],
                (0.6 * LAST_LAYER_INDEX, 1.0 * LAST_LAYER_INDEX, False),
            )
            self.assertEqual(calls[f"{component}.min_weight"], (0.0, 1.0, False))
            self.assertEqual(
                calls[f"{component}.min_weight_distance"],
                (1.0, max(0.6 * LAST_LAYER_INDEX, 1.0), False),
            )

        trial = RecordingTrial()
        make_ara(ARASettings()).suggest_parameters(make_context(), cast(Any, trial))
        calls = trial.calls
        self.assertEqual(calls["start_layer_index"], (0, LAYER_COUNT // 2, False))
        self.assertEqual(
            calls["end_layer_index"], (LAYER_COUNT // 2 + 1, LAYER_COUNT, False)
        )
        self.assertEqual(calls["preserve_good_behavior_weight"], (0.0, 1.0, False))
        self.assertEqual(calls["steer_bad_behavior_weight"], (0.0001, 1.0, True))
        self.assertEqual(calls["overcorrect_relative_weight"], (0.0, 1.3, False))
        self.assertEqual(calls["neighbor_count"], (1, 15, False))

    def test_defaults_equal_v2_for_odd_layer_counts(self) -> None:
        for layer_count in (1, 2, 3, 7, 30, 61):
            stub = SimpleNamespace(
                get_layers=lambda n=layer_count: list(range(n)),
                get_abliterable_components=lambda: ["attn.o_proj"],
            )
            trial = RecordingTrial()
            make_ara(ARASettings()).suggest_parameters(
                cast(Context, SimpleNamespace(get_model=lambda s=stub: s)),
                cast(Any, trial),
            )
            self.assertEqual(
                trial.calls["start_layer_index"][:2],
                (0, min(layer_count // 2, layer_count - 1)),
                layer_count,
            )
            self.assertEqual(
                trial.calls["end_layer_index"][:2],
                (min(layer_count // 2 + 1, layer_count), layer_count),
                layer_count,
            )


def test_wide_ranges_are_used() -> None:
    # Abliteration: every sampled parameter lies in the configured bounds,
    # and the samples leave the upstream ranges.
    modifier = make_abliteration(AbliterationSettings.model_validate(WIDE_ABLITERATION))
    trials = sample(modifier)
    eps = 1e-9

    for trial in trials:
        params = trial.params
        check.assertTrue(
            0.1 * LAST_LAYER_INDEX - eps
            <= params["direction_index"]
            <= 1.0 * LAST_LAYER_INDEX + eps
        )
        check.assertTrue(2.0 <= params["attn.o_proj.max_weight"] <= 4.0)
        check.assertTrue(1.5 <= params["mlp.down_proj.max_weight"] <= 3.0)
        for component in ("attn.o_proj", "mlp.down_proj"):
            check.assertTrue(0.5 <= params[f"{component}.min_weight"] <= 1.0)
            check.assertTrue(
                3.0
                <= params[f"{component}.min_weight_distance"]
                <= 1.5 * LAST_LAYER_INDEX
            )
            check.assertTrue(
                0.0 <= params[f"{component}.max_weight_position"] <= LAST_LAYER_INDEX
            )

    # The widened ranges are really used: values beyond upstream's 1.5 occur.
    check.assertGreater(max(t.params["attn.o_proj.max_weight"] for t in trials), 1.5)
    check.assertGreater(
        max(t.params["attn.o_proj.min_weight_distance"] for t in trials),
        0.6 * LAST_LAYER_INDEX,
    )

    # ARA, including a layer span that was not reachable before.
    ara = make_ara(
        ARASettings(
            start_layer_range=(0.0, 0.9),
            end_layer_range=(0.1, 1.0),
            preserve_good_weight_range=(0.2, 3.0),
            steer_bad_weight_range=(0.001, 5.0),
            overcorrect_range=(1.0, 2.5),
            neighbor_count_max=30,
        )
    )
    ara_trials = sample(ara)

    for trial in ara_trials:
        params = trial.params
        # start 0..floor(0.9 * 12) = 10, end floor(0.1 * 12) + 1 = 2..12
        check.assertTrue(0 <= params["start_layer_index"] <= 10)
        check.assertTrue(2 <= params["end_layer_index"] <= LAYER_COUNT)
        check.assertTrue(0.2 <= params["preserve_good_behavior_weight"] <= 3.0)
        check.assertTrue(0.001 <= params["steer_bad_behavior_weight"] <= 5.0)
        check.assertTrue(1.0 <= params["overcorrect_relative_weight"] <= 2.5)
        check.assertTrue(1 <= params["neighbor_count"] <= 30)
        # The layer range that the modifier actually uses is never empty.
        used = trial.user_attrs["parameters"]
        check.assertLess(used["start_layer_index"], used["end_layer_index"])
        check.assertLessEqual(used["end_layer_index"], LAYER_COUNT)

    check.assertGreater(
        max(t.params["overcorrect_relative_weight"] for t in ara_trials), 1.3
    )
    check.assertGreater(
        max(t.params["start_layer_index"] for t in ara_trials), LAYER_COUNT // 2
    )


def test_invalid_range_rejected() -> None:
    invalid_abliteration: list[tuple[dict[str, Any], str]] = [
        ({"max_weight_range": (1.5, 0.8)}, "greater than upper bound"),
        ({"max_weight_range": (-0.5, 1.5)}, "below the allowed minimum"),
        ({"min_weight_range": (0.9, 0.1)}, "greater than upper bound"),
        ({"min_weight_range": (0.0, 1.5)}, "above the allowed maximum"),
        ({"max_weight_position_range": (0.2, 1.2)}, "above the allowed maximum"),
        ({"direction_index_range": (0.9, 0.4)}, "greater than upper bound"),
        ({"mlp_max_weight_range": (1.0, 0.5)}, "greater than upper bound"),
        ({"mlp_max_weight_range": (-1.0, -0.5)}, "must not be negative"),
        ({"min_weight_distance_min_layers": 0.0}, "greater than 0"),
        ({"min_weight_distance_max_fraction": -1.0}, "below the allowed minimum"),
        ({"max_weight_range": (float("nan"), 1.0)}, "finite"),
    ]
    for kwargs, message in invalid_abliteration:
        with check.subTest(kwargs=kwargs):
            with check.assertRaisesRegex(ValidationError, message):
                AbliterationSettings.model_validate(kwargs)

    invalid_ara: list[tuple[dict[str, Any], str]] = [
        ({"overcorrect_range": (2.0, 1.0)}, "greater than upper bound"),
        ({"overcorrect_range": (-0.1, 1.0)}, "below the allowed minimum"),
        ({"preserve_good_weight_range": (-1.0, 1.0)}, "below the allowed minimum"),
        ({"steer_bad_weight_range": (0.0, 1.0)}, "greater than 0"),
        ({"steer_bad_weight_range": (1.0, 0.1)}, "greater than upper bound"),
        ({"start_layer_range": (0.6, 0.2)}, "greater than upper bound"),
        ({"end_layer_range": (0.5, 1.5)}, "above the allowed maximum"),
        ({"neighbor_count_max": 0}, "greater than 0"),
    ]
    for kwargs, message in invalid_ara:
        with check.subTest(kwargs=kwargs):
            with check.assertRaisesRegex(ValidationError, message):
                ARASettings.model_validate(kwargs)

    # The same check runs when settings come from a config table.
    with check.assertRaisesRegex(ValidationError, "max_weight_range"):
        Abliteration.validate_settings({"max_weight_range": [3.0, 1.0]})

    # A negative lower bound of the MLP range is legal (it is clamped to 0),
    # and a pinned range (low == high) is allowed.
    AbliterationSettings(mlp_max_weight_range=(-1.0, 1.0))
    AbliterationSettings(max_weight_range=(1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
