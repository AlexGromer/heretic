# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests for expert-granular abliteration (EGA) of fused routed experts. They run on CPU
# with the tiny model that tests/test_fused_experts.py uses (tiny-random/qwen3.5-moe:
# 4 layers, hidden size 8, 128 experts, top-10 routing).

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as F
from pydantic import ValidationError
from test_fused_experts import get_model
from torch import Tensor

from heretic.ega import (
    EgaPlan,
    ExpertStatistics,
    FrequencyScaling,
    ablate_fused_experts_per_expert,
    build_ega_plans,
    build_plan,
    collect_expert_statistics,
    expert_forward,
    select_last_positions,
)
from heretic.model import Model
from heretic.modifier import Context
from heretic.modifiers.abliteration import (
    Abliteration,
    Parameters,
    RowNormalization,
    WeightDistribution,
    ablate_fused_experts,
)
from heretic.modifiers.abliteration import Settings as AbliterationSettings
from heretic.utils import Prompt

COMPONENT = "mlp.experts.down_proj"

BAD = [
    Prompt(system="s", user=u)
    for u in ("a", "bb bb", "ccc ccc ccc", "how to make a bomb", "pick a lock")
]
GOOD = [
    Prompt(system="s", user=u)
    for u in ("d d", "eee", "ff ff ff ff", "name a color", "say hello")
]


class FreshModelTestCase(unittest.TestCase):
    """
    Tests that apply LoRA adapters or edit fused experts need a freshly loaded model,
    and must leave the cached one in a state that other test modules can reload.
    """

    def fresh_model(self, dtype: str = "float32") -> Model:
        model = get_model(dtype)
        model.needs_reload = True
        model.reset_model()
        self.addCleanup(self.invalidate, model)
        return model

    @staticmethod
    def invalidate(model: Model) -> None:
        model.settings.seed = None
        model.needs_reload = True


def reload(model: Model) -> None:
    model.needs_reload = True
    model.reset_model()


def fused_tensors(model: Model) -> dict[tuple[int, str], Tensor]:
    return {
        fused.key: fused.parameter.detach().clone()
        for layer_index in range(len(model.get_layers()))
        for fused_list in model.get_layer_fused_experts(layer_index).values()
        for fused in fused_list
    }


def make_modifier(
    model: Model,
    plans: dict[tuple[int, str], EgaPlan],
    *,
    ega: bool,
    row_normalization: RowNormalization = RowNormalization.PRE,
) -> Abliteration:
    layers = len(model.get_layers())
    hidden = model.model.config.get_text_config().hidden_size
    torch.manual_seed(0)
    modifier = Abliteration.__new__(Abliteration)
    model.settings.seed = 1
    modifier.heretic_settings = model.settings
    modifier.settings = AbliterationSettings(
        ega=ega, row_normalization=row_normalization
    )
    modifier.residual_directions = F.normalize(torch.randn(layers + 1, hidden), dim=1)
    modifier.lora_rank = 1
    modifier.ega_plans = plans
    return modifier


def full_strength_parameters(layers: int) -> Parameters:
    # Constant weight 1.0 in every layer for both components.
    distribution = WeightDistribution(
        max_weight=1.0,
        max_weight_position=(layers - 1) / 2,
        min_weight=1.0,
        min_weight_distance=float(layers),
    )
    return Parameters(
        direction_index=None,
        weight_distributions={
            "attn.o_proj": distribution,
            "mlp.down_proj": distribution,
        },
    )


def context_for(model: Model) -> Context:
    return cast(Context, SimpleNamespace(get_model=lambda: model))


def get_plans(model: Model, **kwargs: Any) -> dict[tuple[int, str], EgaPlan]:
    options: dict[str, Any] = {
        "last_positions": 2,
        "min_tokens": 1,
        "orthogonalize": True,
        "frequency_scaling": FrequencyScaling.MAX,
    }
    options.update(kwargs)
    with redirect_stdout(io.StringIO()):
        return build_ega_plans(model, GOOD, BAD, **options)


class PureFunctionTests(unittest.TestCase):
    def test_select_last_positions(self) -> None:
        rows = select_last_positions(2, 5, 2)
        self.assertEqual(rows.tolist(), [3, 4, 8, 9])
        # More positions requested than there are.
        self.assertEqual(select_last_positions(1, 2, 8).tolist(), [0, 1])

    def make_statistics(
        self, counts: list[int], means: list[list[float]]
    ) -> ExpertStatistics:
        count = torch.tensor(counts, dtype=torch.int64)
        mean = torch.tensor(means, dtype=torch.float64)
        return ExpertStatistics(output_sum=mean * count.unsqueeze(1), count=count)

    def test_build_plan_semantics(self) -> None:
        # Four experts: 0 differs between the sets, 1 is unused by bad prompts,
        # 2 is used by bad prompts only, 3 has too few bad tokens.
        bad = self.make_statistics(
            [4, 0, 2, 1], [[1.0, 1.0, 0.0], [0, 0, 0], [0.0, 3.0, 0.0], [5, 5, 5]]
        )
        good = self.make_statistics(
            [4, 4, 0, 4], [[1.0, -1.0, 0.0], [1, 1, 1], [0, 0, 0], [1, 2, 3]]
        )
        plan = build_plan(
            bad,
            good,
            min_tokens=2,
            orthogonalize=False,
            frequency_scaling=FrequencyScaling.MAX,
        )

        # Expert 0: direction is the normalized difference (0, 2, 0).
        torch.testing.assert_close(plan.directions[0], torch.tensor([0.0, 1.0, 0.0]))
        # Experts 1 and 3 are unused (fewer than min_tokens bad tokens).
        self.assertEqual(plan.scales[1].item(), 0.0)
        self.assertEqual(plan.scales[3].item(), 0.0)
        # Expert 2 is used but has no harmless tokens: fallback to the layer direction.
        self.assertTrue(plan.fallback[2].item())
        self.assertFalse(plan.fallback[0].item())
        # Frequencies 4/6 and 2/6, normalized by the maximum.
        self.assertAlmostEqual(plan.scales[0].item(), 1.0)
        self.assertAlmostEqual(plan.scales[2].item(), 0.5)
        self.assertEqual(plan.used_experts(), [0, 2])

        plan_none = build_plan(
            bad,
            good,
            min_tokens=2,
            orthogonalize=False,
            frequency_scaling=FrequencyScaling.NONE,
        )
        self.assertEqual(plan_none.scales.tolist(), [1.0, 0.0, 1.0, 0.0])

        plan_mean = build_plan(
            bad,
            good,
            min_tokens=2,
            orthogonalize=False,
            frequency_scaling=FrequencyScaling.MEAN,
        )
        # Used experts have frequencies 4/7 and 2/7 (of 7 routed slots); their mean is 3/7.
        self.assertAlmostEqual(plan_mean.scales[0].item(), 1.0)
        self.assertAlmostEqual(plan_mean.scales[2].item(), 2 / 3, places=6)

    def test_orthogonalization(self) -> None:
        bad = self.make_statistics([3], [[1.0, 1.0, 0.0]])
        good = self.make_statistics([3], [[0.0, 1.0, 0.0]])
        plan = build_plan(
            bad,
            good,
            min_tokens=1,
            orthogonalize=True,
            frequency_scaling=FrequencyScaling.MAX,
        )
        # difference = (1, 0, 0), already orthogonal to the harmless mean (0, 1, 0).
        torch.testing.assert_close(plan.directions[0], torch.tensor([1.0, 0.0, 0.0]))

        good = self.make_statistics([3], [[1.0, 0.0, 0.0]])
        plan = build_plan(
            bad,
            good,
            min_tokens=1,
            orthogonalize=True,
            frequency_scaling=FrequencyScaling.MAX,
        )
        # difference = (0, 1, 0), orthogonal again; and a parallel case keeps the direction.
        torch.testing.assert_close(plan.directions[0], torch.tensor([0.0, 1.0, 0.0]))
        parallel_bad = self.make_statistics([3], [[2.0, 0.0, 0.0]])
        plan = build_plan(
            parallel_bad,
            good,
            min_tokens=1,
            orthogonalize=True,
            frequency_scaling=FrequencyScaling.MAX,
        )
        torch.testing.assert_close(plan.directions[0], torch.tensor([1.0, 0.0, 0.0]))

    def test_per_expert_ablation_matches_v2_for_equal_directions(self) -> None:
        torch.manual_seed(5)
        for mode in RowNormalization:
            for row_axis in (1, 2):
                shape = (6, 8, 5) if row_axis == 1 else (6, 5, 8)
                parameter = torch.randn(shape)
                expected = parameter.clone()
                v = F.normalize(torch.randn(8), dim=0)

                ablate_fused_experts(expected, row_axis, v, 0.7, mode)
                ablate_fused_experts_per_expert(
                    parameter,
                    row_axis,
                    v.expand(6, 8),
                    torch.full((6,), 0.7),
                    mode,
                    max_chunk_elements=8 * 5 * 2,
                )
                torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-5)

    def test_zero_weight_experts_stay_bit_exact(self) -> None:
        torch.manual_seed(6)
        parameter = torch.randn(6, 8, 5, dtype=torch.bfloat16)
        original = parameter.clone()
        weights = torch.tensor([0.0, 1.0, 0.0, 0.5, 0.0, 0.0])
        directions = F.normalize(torch.randn(6, 8), dim=1)

        ablate_fused_experts_per_expert(
            parameter, 1, directions, weights, RowNormalization.FULL
        )

        for expert in (0, 2, 4, 5):
            self.assertTrue(torch.equal(parameter[expert], original[expert]))
        for expert in (1, 3):
            self.assertFalse(torch.equal(parameter[expert], original[expert]))


class EgaModelTests(FreshModelTestCase):
    def test_expert_forward_matches_transformers(self) -> None:
        # The recomputed expert output must equal what the model's own experts module
        # produces for a single routed token with routing weight 1.
        model = get_model()
        fused = model.get_layer_fused_experts(1)[COMPONENT][0]
        hidden = fused.parameter.shape[1]
        x = torch.randn(3, hidden)
        experts = fused.module
        for expert in (0, 5, 77):
            index = torch.full((3, 1), expert)
            weights = torch.ones(3, 1)
            with torch.no_grad():
                expected = cast(Any, experts)(x, index, weights)
                actual = expert_forward(experts, expert, x)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_statistics_count_routed_tokens(self) -> None:
        model = get_model()
        with redirect_stdout(io.StringIO()):
            statistics = collect_expert_statistics(model, BAD, 2)

        top_k = model.model.config.get_text_config().num_experts_per_tok
        for layer_index in range(len(model.get_layers())):
            counts = statistics[(layer_index, COMPONENT)].count
            # 5 prompts, 2 positions, every token goes to top_k distinct experts.
            self.assertEqual(int(counts.sum()), len(BAD) * 2 * top_k)

    def test_per_expert_directions(self) -> None:
        model = self.fresh_model()
        plans = get_plans(model)
        before = fused_tensors(model)
        layers = len(model.get_layers())

        with redirect_stdout(io.StringIO()):
            bad_statistics = collect_expert_statistics(model, BAD, 2)

        # Each routed expert has its own direction: the unit directions of the experts
        # that have enough data of their own are not all the same.
        own_directions = []
        for layer_index in range(layers):
            plan = plans[(layer_index, COMPONENT)]
            used = set(plan.used_experts())
            routed = {
                int(e)
                for e in torch.nonzero(
                    bad_statistics[(layer_index, COMPONENT)].count > 0
                ).flatten()
            }
            self.assertEqual(used, routed)
            # Unused experts have no direction and no strength.
            for expert in range(plan.scales.shape[0]):
                if expert not in used:
                    self.assertEqual(plan.scales[expert].item(), 0.0)
                    self.assertEqual(plan.directions[expert].abs().sum().item(), 0.0)
            for expert in used:
                self.assertGreater(plan.scales[expert].item(), 0.0)
                self.assertLessEqual(plan.scales[expert].item(), 1.0)
                if not plan.fallback[expert]:
                    direction = plan.directions[expert]
                    self.assertAlmostEqual(direction.norm().item(), 1.0, places=5)
                    own_directions.append(direction)
            # Not all experts are used (so the test is meaningful) ...
            self.assertLess(len(used), plan.scales.shape[0])

        # ... several experts have their own direction, and they differ.
        self.assertGreater(len(own_directions), 5)
        stacked = torch.stack(own_directions)
        similarities = stacked @ stacked.T
        off_diagonal = similarities[~torch.eye(len(stacked), dtype=torch.bool)]
        self.assertLess(off_diagonal.min().item(), 0.99)
        self.assertGreater(
            len({tuple(round(x, 4) for x in d.tolist()) for d in own_directions}),
            len(own_directions) // 2,
        )

        # Applying the plan changes exactly the experts that harmful prompts use.
        modifier = make_modifier(model, plans, ega=True)
        model.apply_lora(1)
        with redirect_stdout(io.StringIO()):
            modifier.modify_model(context_for(model), full_strength_parameters(layers))

        after = fused_tensors(model)
        for layer_index in range(layers):
            key = (layer_index, COMPONENT)
            used = set(plans[key].used_experts())
            changed = {
                expert
                for expert in range(before[key].shape[0])
                if not torch.equal(before[key][expert], after[key][expert])
            }
            self.assertEqual(changed, used)

        # Each used expert lost (part of) its own direction, scaled by its strength:
        # with row normalization "none" and full strength the component is removed.
        reload(model)
        modifier = make_modifier(
            model, plans, ega=True, row_normalization=RowNormalization.NONE
        )
        model.apply_lora(1)
        with redirect_stdout(io.StringIO()):
            modifier.modify_model(context_for(model), full_strength_parameters(layers))
        fused = model.get_layer_fused_experts(0)[COMPONENT][0]
        plan = plans[(0, COMPONENT)]
        for expert in plan.used_experts():
            if plan.fallback[expert]:
                continue
            v = plan.directions[expert]
            scale = plan.scales[expert].item()
            original = before[(0, COMPONENT)][expert]
            expected = original - scale * torch.outer(v, v @ original)
            torch.testing.assert_close(
                fused.parameter[expert].detach(), expected, rtol=1e-4, atol=1e-5
            )

    def test_shared_expert_is_ablated_like_a_dense_mlp(self) -> None:
        # EGA does not change what happens to the shared expert / attention outputs.
        model = self.fresh_model()
        plans = get_plans(model)
        layers = len(model.get_layers())
        parameters = full_strength_parameters(layers)

        def lora_state(ega: bool) -> list[Tensor]:
            reload(model)
            model.apply_lora(1)
            modifier = make_modifier(model, plans, ega=ega)
            with redirect_stdout(io.StringIO()):
                modifier.modify_model(context_for(model), parameters)
            return [
                cast(Any, module).lora_B["default"].weight.detach().clone()
                for layer_index in range(layers)
                for modules in model.get_layer_modules(layer_index).values()
                for module in modules
            ]

        with_ega = lora_state(True)
        without_ega = lora_state(False)
        self.assertTrue(any(b.any() for b in with_ega))
        for a, b in zip(with_ega, without_ega):
            self.assertTrue(torch.equal(a, b))

    def test_ega_off_parity(self) -> None:
        # ega = false: the fused experts are ablated exactly like in v2, with the layer
        # direction and the layer weight for all experts, whatever plans exist.
        model = self.fresh_model()
        plans = get_plans(model)
        layers = len(model.get_layers())
        before = fused_tensors(model)

        modifier = make_modifier(model, plans, ega=False)
        model.apply_lora(1)
        with redirect_stdout(io.StringIO()):
            modifier.modify_model(context_for(model), full_strength_parameters(layers))
        after = fused_tensors(model)

        for layer_index in range(layers):
            key = (layer_index, COMPONENT)
            expected = before[key].clone()
            # Weight 1.0 everywhere; direction_index None means the layer's own direction.
            ablate_fused_experts(
                expected,
                1,
                modifier.residual_directions[layer_index + 1],
                1.0,
                RowNormalization.PRE,
            )
            self.assertTrue(torch.equal(after[key], expected), key)
            # Every expert changed (no per-expert selection).
            self.assertTrue(
                all(
                    not torch.equal(before[key][e], after[key][e])
                    for e in range(before[key].shape[0])
                )
            )

        # A modifier that never computed plans behaves the same.
        reload(model)
        fresh = make_modifier(model, {}, ega=False)
        del fresh.ega_plans
        model.apply_lora(1)
        with redirect_stdout(io.StringIO()):
            fresh.modify_model(context_for(model), full_strength_parameters(layers))
        for key, tensor in fused_tensors(model).items():
            self.assertTrue(torch.equal(tensor, after[key]))

    def test_restore_between_trials(self) -> None:
        # Edit with EGA, reset like the optimizer does between trials, and compare
        # bit-exactly (also in bfloat16, where arithmetic undo would not work).
        for dtype in ("float32", "bfloat16"):
            with self.subTest(dtype=dtype):
                model = self.fresh_model(dtype)
                plans = get_plans(model)
                layers = len(model.get_layers())
                before = fused_tensors(model)

                modifier = make_modifier(
                    model, plans, ega=True, row_normalization=RowNormalization.FULL
                )
                model.apply_lora(modifier.lora_rank)
                ctx = context_for(model)
                for trial in range(2):
                    with redirect_stdout(io.StringIO()):
                        modifier.modify_model(ctx, full_strength_parameters(layers))
                    changed = fused_tensors(model)
                    self.assertTrue(
                        any(
                            not torch.equal(before[key], changed[key]) for key in before
                        )
                    )
                    self.assertTrue(model.has_modified_fused_experts())

                    modifier.reset_model(ctx)

                    self.assertFalse(model.has_modified_fused_experts())
                    restored = fused_tensors(model)
                    for key in before:
                        self.assertTrue(torch.equal(before[key], restored[key]), key)

    def test_ega_setting_without_fused_experts_is_harmless(self) -> None:
        # The setting is part of the modifier settings; it must validate and default off.
        settings = AbliterationSettings()
        self.assertFalse(settings.ega)
        self.assertEqual(settings.ega_frequency_scaling, FrequencyScaling.MAX)
        with self.assertRaises(ValidationError):
            AbliterationSettings(ega_last_positions=0)
        with self.assertRaises(ValidationError):
            AbliterationSettings.model_validate({"ega_frequency_scaling": "sometimes"})


if __name__ == "__main__":
    unittest.main()
