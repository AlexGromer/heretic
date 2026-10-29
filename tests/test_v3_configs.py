# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# The example configs of the fork must stay valid. Pure Python.

import sys
import unittest
from pathlib import Path
from typing import Any, cast

from heretic.config import Settings
from heretic.modifiers.abliteration import Abliteration
from heretic.modifiers.ara import ARA
from heretic.scorers.keyword_rate import KeywordRate

if sys.version_info >= (3, 11):
    import tomllib

ROOT = Path(__file__).resolve().parent.parent


@unittest.skipIf(sys.version_info < (3, 11), "tomllib needs Python 3.11")
class ExampleConfigTests(unittest.TestCase):
    def load(self, name: str) -> dict[str, Any]:
        with open(ROOT / name, "rb") as file:
            return tomllib.load(file)

    def check(self, name: str) -> dict[str, Any]:
        config = self.load(name)

        # Top-level keys are real settings.
        for key in config:
            if key not in ("scorer", "modifier"):
                self.assertIn(key, Settings.model_fields, f"{name}: unknown key {key}")

        for plugin_class, namespace in (
            (KeywordRate, "scorer"),
            (ARA, "modifier"),
            (Abliteration, "modifier"),
        ):
            raw = plugin_class.get_settings_raw(config, namespace, None)
            plugin_class.validate_settings(raw)

        return config

    def test_default_config_equals_defaults(self) -> None:
        # The documented values in config.default.toml are the defaults of the settings.
        config = self.check("config.default.toml")
        for plugin_class, namespace in (
            (KeywordRate, "scorer"),
            (ARA, "modifier"),
            (Abliteration, "modifier"),
        ):
            documented = plugin_class.validate_settings(
                plugin_class.get_settings_raw(config, namespace, None)
            )
            defaults = plugin_class.validate_settings({})
            assert documented is not None and defaults is not None
            for key in (
                "strict",
                "strict_window_tokens",
                "strict_markers",
                "strict_empty_answer_is_match",
                "start_layer_range",
                "end_layer_range",
                "preserve_good_weight_range",
                "steer_bad_weight_range",
                "overcorrect_range",
                "neighbor_count_max",
                "direction_index_range",
                "max_weight_range",
                "mlp_max_weight_range",
                "max_weight_position_range",
                "min_weight_range",
                "min_weight_distance_min_layers",
                "min_weight_distance_max_fraction",
                "ega",
                "ega_last_positions",
                "ega_min_tokens",
                "ega_frequency_scaling",
            ):
                if hasattr(defaults, key):
                    self.assertEqual(
                        getattr(documented, key), getattr(defaults, key), key
                    )

    def test_wide_config(self) -> None:
        config = self.check("config.wide.toml")
        settings = Abliteration.validate_settings(
            Abliteration.get_settings_raw(config, "modifier", None)
        )
        assert settings is not None
        self.assertEqual(cast(Any, settings).max_weight_range, (1.0, 3.0))
        self.assertFalse(cast(Any, settings).ega)

    def test_moe_ega_config(self) -> None:
        config = self.check("config.moe-ega.toml")
        settings = Abliteration.validate_settings(
            Abliteration.get_settings_raw(config, "modifier", None)
        )
        assert settings is not None
        self.assertTrue(cast(Any, settings).ega)
        scorer = KeywordRate.validate_settings(
            KeywordRate.get_settings_raw(config, "scorer", None)
        )
        assert scorer is not None
        self.assertTrue(cast(Any, scorer).strict)


if __name__ == "__main__":
    unittest.main()
