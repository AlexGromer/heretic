# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests of the v8 additions, which use counterfactual prompt pairs (refusal-eval
# train/bad_train and train/good_train are twins at the same index): the whitened
# direction, the directions from the singular vectors of the pair differences, and ARA's
# index pairing. CPU only; prompts are neutral placeholders.

import io
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as F

from heretic.directions import (
    paired_differences,
    paired_svd_directions,
    whiten,
)
from heretic.modifiers.abliteration import Abliteration, DirectionEstimator
from heretic.modifiers.abliteration import Settings as AbliterationSettings
from heretic.modifiers.ara import ARA, Pairing, ara_loss, distances_to_twin
from heretic.modifiers.ara import Parameters as ARAParameters
from heretic.modifiers.ara import Settings as ARASettings
from heretic.plugin import Context


class PairedStatisticsTests(unittest.TestCase):
    def test_paired_differences_need_matching_shapes(self) -> None:
        bad = torch.zeros(4, 3, 8)
        self.assertEqual(tuple(paired_differences(bad, bad).shape), (4, 3, 8))
        with self.assertRaisesRegex(ValueError, "same number of bad and good"):
            paired_differences(bad, torch.zeros(5, 3, 8))

    def test_mean_of_differences_is_the_difference_of_means(self) -> None:
        torch.manual_seed(0)
        bad, good = torch.randn(16, 2, 6), torch.randn(16, 2, 6)
        self.assertTrue(
            torch.allclose(
                paired_differences(bad, good).mean(dim=0),
                bad.mean(dim=0) - good.mean(dim=0),
                atol=1e-6,
            )
        )

    def test_whitening_turns_towards_the_low_variance_axis(self) -> None:
        # The differences vary a lot along axis 0 and little along axis 1, and the mean
        # difference points between them: whitening must favour axis 1.
        torch.manual_seed(0)
        pairs = 64
        differences = torch.zeros(pairs, 2)
        differences[:, 0] = torch.randn(pairs) * 5.0
        differences[:, 1] = torch.randn(pairs) * 0.1
        differences = differences + torch.tensor([1.0, 1.0])
        direction = F.normalize(differences.mean(dim=0), p=2, dim=0)
        whitened = whiten(direction, differences, shrinkage=0.01)

        self.assertAlmostEqual(float(whitened.norm()), 1.0, places=5)
        self.assertGreater(float(whitened[1].abs()), float(whitened[0].abs()))
        self.assertLess(float(direction[1].abs()), float(direction[0].abs()) + 1e-6)
        # A large ridge goes back towards the plain mean difference.
        heavy = whiten(direction, differences, shrinkage=1e6)
        self.assertGreater(float((heavy * direction).sum()), 0.999)
        # Too few pairs: the direction is returned unchanged.
        self.assertTrue(
            torch.allclose(
                whiten(direction, differences[:1], 0.1), direction, atol=1e-6
            )
        )

    def test_paired_svd_finds_the_axis_of_the_spread(self) -> None:
        torch.manual_seed(0)
        pairs = 32
        differences = torch.zeros(pairs, 4)
        differences[:, 2] = torch.randn(pairs) * 3.0
        directions = paired_svd_directions(differences, 2)
        self.assertEqual(tuple(directions.shape), (2, 4))
        self.assertGreater(float(directions[0].abs()[2]), 0.99)
        # Fewer pairs than requested directions: the rest stays zero.
        self.assertTrue(
            torch.equal(paired_svd_directions(differences[:1], 2), torch.zeros(2, 4))
        )


def tiny_model():
    from test_fused_experts import get_model

    model = get_model()
    model.reset_model()
    return model


def write_pairs(directory: str, count: int = 12) -> tuple[str, str]:
    bad = Path(directory, "bad.txt")
    good = Path(directory, "good.txt")
    # Twins: the same opening words and the same term per index, one "bad", one "good".
    bad.write_text(
        "".join(f"review the term{i} setup for weakness {i}.\n" for i in range(count)),
        encoding="utf-8",
    )
    good.write_text(
        "".join(f"review the term{i} setup for hardening {i}.\n" for i in range(count)),
        encoding="utf-8",
    )
    return str(bad), str(good)


class AbliterationPairedTests(unittest.TestCase):
    def build(self, directory: str, **settings: Any) -> Abliteration:
        model = tiny_model()
        bad, good = write_pairs(directory)
        modifier = Abliteration.__new__(Abliteration)
        modifier.heretic_settings = model.settings
        modifier.settings = AbliterationSettings(
            good_prompts={"dataset": good},
            bad_prompts={"dataset": bad},
            **settings,
        )
        with (
            unittest.mock.patch.object(model, "apply_lora", lambda rank: None),
            redirect_stdout(io.StringIO()),
        ):
            modifier.init(Context(settings=model.settings, model=model))
        return modifier

    def test_whitened_directions_are_unit_and_differ_from_mean_diff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plain = self.build(directory)
            whitened = self.build(
                directory, direction_estimator=DirectionEstimator.WHITENED
            )

        # The row of the embeddings holds the embedding of the LAST prompt token, which the
        # pairing makes identical, so its direction is zero for both estimators. Heretic
        # never ablates that row (a layer uses the row after it), so only the rest matters.
        present = plain.residual_directions.norm(dim=1) > 0
        self.assertTrue(bool(present[1:].all()))
        norms = whitened.residual_directions[present].norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))
        self.assertEqual(
            whitened.residual_directions.shape, plain.residual_directions.shape
        )
        cosines = (
            whitened.residual_directions[present] * plain.residual_directions[present]
        ).sum(dim=1)
        self.assertTrue(float(cosines.abs().max()) <= 1.0 + 1e-5)
        self.assertLess(float(cosines.abs().min()), 0.999)
        tiny_model()

    def test_paired_svd_slots_are_orthonormal_to_the_main_direction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            modifier = self.build(
                directory,
                direction_count=3,
                extra_direction_source="paired_svd",
            )

        self.assertEqual(modifier.extra_direction_names, ("direction_1", "direction_2"))
        extras = modifier.extra_directions
        assert extras is not None
        main = modifier.residual_directions
        for row in range(main.shape[0]):
            if not bool(main[row].norm() > 0):
                continue  # see the comment on the embedding row above
            vectors = [main[row]] + [d for d in extras[row] if bool(d.norm() > 0)]
            stacked = torch.stack(vectors)
            gram = stacked @ stacked.T
            self.assertTrue(torch.allclose(gram, torch.eye(len(vectors)), atol=1e-4))
        tiny_model()

    def test_unpaired_sets_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bad, good = write_pairs(directory)
            Path(good).write_text(
                Path(good).read_text(encoding="utf-8") + "one more good prompt.\n",
                encoding="utf-8",
            )
            model = tiny_model()
            modifier = Abliteration.__new__(Abliteration)
            modifier.heretic_settings = model.settings
            modifier.settings = AbliterationSettings(
                good_prompts={"dataset": good},
                bad_prompts={"dataset": bad},
                direction_estimator=DirectionEstimator.WHITENED,
            )
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "same index"):
                    modifier.init(Context(settings=model.settings, model=model))

    def test_settings_validation(self) -> None:
        with self.assertRaises(ValueError):
            AbliterationSettings(whitening_shrinkage=0)
        defaults = AbliterationSettings()
        self.assertEqual(defaults.direction_estimator, DirectionEstimator.MEAN_DIFF)
        self.assertEqual(defaults.extra_direction_source.value, "som")


class AraPairingTests(unittest.TestCase):
    def parameters(self, pairing: str) -> ARAParameters:
        return ARAParameters(
            start_layer_index=0,
            end_layer_index=2,
            preserve_good_behavior_weight=0.5,
            steer_bad_behavior_weight=1.0,
            overcorrect_relative_weight=0.0,
            neighbor_count=2,
            pairing=pairing,
        )

    def test_twin_distance_and_loss_use_the_same_index(self) -> None:
        good = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        bad = torch.tensor([[1.0, 0.0], [11.0, 10.0]])
        self.assertTrue(
            torch.allclose(distances_to_twin(bad, good), torch.tensor([1.0, 1.0]))
        )
        with self.assertRaisesRegex(ValueError, "matching rows"):
            distances_to_twin(bad, good[:1])

        # The twin of row 1 is far from row 0, so k-NN (k=2) and index pairing differ.
        index_loss = ara_loss(good, bad, good, bad, self.parameters("index"))
        knn_loss = ara_loss(good, bad, good, bad, self.parameters("knn"))
        self.assertAlmostEqual(float(index_loss), 1.0, places=5)
        self.assertGreater(float(knn_loss), float(index_loss))

    def test_parameters_round_trip_and_default(self) -> None:
        stored = self.parameters("index").to_dict()
        restored = cast(ARAParameters, ARAParameters.from_dict(stored))
        self.assertEqual(restored.pairing, "index")
        self.assertIn("pairing", restored.to_presentation_dict())
        # Trials stored before this setting existed default to the upstream mode.
        old = cast(
            ARAParameters,
            ARAParameters.from_dict(
                {
                    "start_layer_index": 0,
                    "end_layer_index": 1,
                    "preserve_good_behavior_weight": 0.5,
                    "steer_bad_behavior_weight": 0.5,
                    "overcorrect_relative_weight": 0.0,
                    "neighbor_count": 1,
                }
            ),
        )
        self.assertEqual(old.pairing, Pairing.KNN.value)

    def test_init_refuses_unpaired_sets_and_reports_pairs(self) -> None:
        model = tiny_model()
        with tempfile.TemporaryDirectory() as directory:
            bad, good = write_pairs(directory, count=4)
            modifier = ARA.__new__(ARA)
            modifier.heretic_settings = model.settings
            modifier.settings = ARASettings(
                good_prompts={"dataset": good},
                bad_prompts={"dataset": bad},
                pairing=Pairing.INDEX,
                lora_rank=2,
            )
            ctx = Context(settings=model.settings, model=model)
            with (
                unittest.mock.patch.object(model, "apply_lora", lambda rank: None),
                redirect_stdout(io.StringIO()) as output,
            ):
                modifier.init(ctx)
            self.assertIn("Index pairing: 4 pairs", output.getvalue())
            self.assertEqual(
                modifier.suggest_parameters(
                    cast(Context, SimpleNamespace(get_model=lambda: model)),
                    _trial(),
                ).pairing,
                "index",
            )

            Path(good).write_text(
                Path(good).read_text(encoding="utf-8") + "one more good prompt.\n",
                encoding="utf-8",
            )
            modifier.settings = ARASettings(
                good_prompts={"dataset": good},
                bad_prompts={"dataset": bad},
                pairing=Pairing.INDEX,
                lora_rank=2,
            )
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "same index"):
                    modifier.init(ctx)
        model.reset_model()


def _trial():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study().ask()


if __name__ == "__main__":
    unittest.main()
