# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests for the support of fused (3D) routed experts, as used by Transformers v5
# models like Qwen3.5 MoE. They run on CPU with the tiny model that is also used
# by the end-to-end test in tests/qwen3.5-moe.

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
import torch.nn.functional as F

from heretic.config import ExportStrategy, QuantizationMethod, Settings
from heretic.main import obtain_export_strategy
from heretic.model import Model, get_model_class
from heretic.modifiers.abliteration import RowNormalization, ablate_fused_experts
from heretic.modifiers.ara import get_fused_experts_warning

MODEL = "tiny-random/qwen3.5-moe"
MODEL_COMMIT = "2ebfa8d9717238c5dda927008104fa172a149050"

# Cache the loaded models, because loading takes a few seconds each.
_models: dict[str, Model] = {}


def get_model(dtype: str = "float32") -> Model:
    if dtype not in _models:
        previous_directory = os.getcwd()
        # An empty directory ensures that no config.toml is picked up.
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                with patch.object(sys, "argv", ["heretic"]):
                    settings = Settings(
                        model=MODEL,
                        model_commit=MODEL_COMMIT,
                        dtypes=[dtype],
                        device_map="cpu",
                        batch_size=2,
                    )
                _models[dtype] = Model(settings)
            finally:
                os.chdir(previous_directory)

    model = _models[dtype]
    model.reset_model()
    return model


def reference_ablation(
    weight_matrix: torch.Tensor,
    v: torch.Tensor,
    weight: float,
    row_normalization: RowNormalization,
) -> torch.Tensor:
    # Straightforward per-matrix implementation of the Linear-module math
    # in Abliteration.modify_model (without the low-rank approximation
    # that is applied for full normalization there).
    W = weight_matrix.to(torch.float32)
    norms = torch.linalg.vector_norm(W, dim=1, keepdim=True)

    if row_normalization == RowNormalization.NONE:
        return W - weight * torch.outer(v, v @ W)

    W_normalized = F.normalize(W, p=2, dim=1)
    ablated = W_normalized - weight * torch.outer(v, v @ W_normalized)

    if row_normalization == RowNormalization.PRE:
        return W + norms * (ablated - W_normalized)

    return F.normalize(ablated, p=2, dim=1) * norms


class FusedDiscoveryTests(unittest.TestCase):
    def test_fused_experts_are_discovered_in_every_layer(self) -> None:
        model = get_model()
        hidden_size = model.model.config.get_text_config().hidden_size
        num_experts = model.model.config.get_text_config().num_experts

        for layer_index in range(len(model.get_layers())):
            fused = model.get_layer_fused_experts(layer_index)
            self.assertEqual(list(fused.keys()), ["mlp.experts.down_proj"])
            self.assertEqual(len(fused["mlp.experts.down_proj"]), 1)

            handle = fused["mlp.experts.down_proj"][0]
            # Verified against transformers 5.20.2, Qwen3_5MoeExperts:
            # down_proj is (num_experts, hidden_dim, intermediate_dim) and
            # applied as F.linear(x, down_proj[expert]), so axis 1 is the output
            # (hidden) dimension, exactly as the rows of a Linear weight.
            self.assertEqual(handle.row_axis, 1)
            self.assertEqual(handle.num_experts, num_experts)
            self.assertEqual(handle.parameter.shape[0], num_experts)
            self.assertEqual(handle.parameter.shape[1], hidden_size)
            self.assertIs(
                handle.parameter,
                cast(Any, get_model().get_layers()[layer_index]).mlp.experts.down_proj,
            )

    def test_module_based_callers_are_unaffected(self) -> None:
        model = get_model()

        for layer_index in range(len(model.get_layers())):
            modules = model.get_layer_modules(layer_index)
            # Only nn.Module instances (PEFT LoRA targets), no fused tensors.
            self.assertNotIn("mlp.experts.down_proj", modules)
            self.assertEqual(len(modules["mlp.down_proj"]), 1)
            for module_list in modules.values():
                for module in module_list:
                    self.assertIsInstance(module, torch.nn.Module)

        # The fused experts share the mlp.down_proj Optuna parameters.
        self.assertEqual(
            model.get_abliterable_components(),
            ["attn.o_proj", "mlp.down_proj"],
        )
        self.assertTrue(model.has_fused_experts())


class FusedAblationTests(unittest.TestCase):
    def check_matches_reference(self, row_normalization: RowNormalization) -> None:
        model = get_model()
        handle = model.get_layer_fused_experts(1)["mlp.experts.down_proj"][0]
        original = handle.parameter.detach().clone()

        torch.manual_seed(0)
        v = F.normalize(torch.randn(original.shape[1]), dim=0)
        weight = 0.8

        model.begin_fused_edit(handle)
        # A tiny chunk size also exercises the chunking.
        ablate_fused_experts(
            handle.parameter,
            handle.row_axis,
            v,
            weight,
            row_normalization,
            max_chunk_elements=original.shape[1] * original.shape[2] * 3,
        )

        for expert in range(original.shape[0]):
            expected = reference_ablation(
                original[expert], v, weight, row_normalization
            )
            torch.testing.assert_close(
                handle.parameter[expert].detach(), expected, rtol=1e-5, atol=1e-5
            )

        # The parameter was changed in place.
        self.assertFalse(torch.equal(handle.parameter.detach(), original))
        self.assertTrue(model.has_modified_fused_experts())

    def test_none_matches_reference_einsum(self) -> None:
        model = get_model()
        handle = model.get_layer_fused_experts(0)["mlp.experts.down_proj"][0]
        original = handle.parameter.detach().clone()
        torch.manual_seed(1)
        v = F.normalize(torch.randn(original.shape[1]), dim=0)

        model.begin_fused_edit(handle)
        ablate_fused_experts(
            handle.parameter, handle.row_axis, v, 1.0, RowNormalization.NONE
        )

        expected = original - torch.einsum(
            "h,ei->ehi", v, torch.einsum("h,ehi->ei", v, original)
        )
        torch.testing.assert_close(
            handle.parameter.detach(), expected, rtol=1e-5, atol=1e-5
        )
        # Full-strength ablation removes the direction from every expert.
        residual = torch.einsum("h,ehi->ei", v, handle.parameter.detach())
        self.assertLess(residual.abs().max().item(), 1e-5)

    def test_pre_is_consistent_with_linear_math(self) -> None:
        self.check_matches_reference(RowNormalization.PRE)

    def test_full_is_consistent_with_linear_math(self) -> None:
        self.check_matches_reference(RowNormalization.FULL)

    def test_full_preserves_row_norms(self) -> None:
        model = get_model()
        handle = model.get_layer_fused_experts(2)["mlp.experts.down_proj"][0]
        original = handle.parameter.detach().clone()
        torch.manual_seed(2)
        v = F.normalize(torch.randn(original.shape[1]), dim=0)

        model.begin_fused_edit(handle)
        ablate_fused_experts(
            handle.parameter, handle.row_axis, v, 1.0, RowNormalization.FULL
        )

        torch.testing.assert_close(
            torch.linalg.vector_norm(handle.parameter.detach(), dim=2),
            torch.linalg.vector_norm(original, dim=2),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_transposed_orientation(self) -> None:
        # Models like gpt-oss store down_proj as (expert, intermediate, hidden).
        torch.manual_seed(3)
        parameter = torch.randn(5, 7, 4)
        original = parameter.clone()
        v = F.normalize(torch.randn(4), dim=0)

        ablate_fused_experts(parameter, 2, v, 0.5, RowNormalization.PRE)

        for expert in range(5):
            expected = reference_ablation(
                original[expert].T, v, 0.5, RowNormalization.PRE
            ).T
            torch.testing.assert_close(
                parameter[expert], expected, rtol=1e-5, atol=1e-5
            )

    def test_reset_restores_bit_exactly(self) -> None:
        model = get_model("bfloat16")
        before = {
            layer_index: model.get_layer_fused_experts(layer_index)[
                "mlp.experts.down_proj"
            ][0]
            .parameter.detach()
            .clone()
            for layer_index in range(len(model.get_layers()))
        }
        v = F.normalize(torch.randn(8), dim=0)

        for layer_index in (0, 3):
            handle = model.get_layer_fused_experts(layer_index)[
                "mlp.experts.down_proj"
            ][0]
            model.begin_fused_edit(handle)
            ablate_fused_experts(
                handle.parameter, handle.row_axis, v, 1.0, RowNormalization.FULL
            )
            self.assertFalse(
                torch.equal(handle.parameter.detach(), before[layer_index])
            )

        self.assertTrue(model.has_modified_fused_experts())
        self.assertTrue(model.reset_model())
        self.assertFalse(model.has_modified_fused_experts())

        for layer_index, expected in before.items():
            handle = model.get_layer_fused_experts(layer_index)[
                "mlp.experts.down_proj"
            ][0]
            self.assertEqual(handle.parameter.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(handle.parameter.detach(), expected))

    def test_save_pretrained_contains_modified_weights(self) -> None:
        model = get_model()
        model.apply_lora(1)
        handle = model.get_layer_fused_experts(1)["mlp.experts.down_proj"][0]
        v = F.normalize(torch.randn(8), dim=0)
        model.begin_fused_edit(handle)
        ablate_fused_experts(
            handle.parameter, handle.row_axis, v, 1.0, RowNormalization.NONE
        )
        modified = handle.parameter.detach().clone()

        merged = model.get_merged_model()
        with tempfile.TemporaryDirectory() as directory:
            merged.save_pretrained(directory)
            self.assertTrue(any(Path(directory).glob("*.safetensors")))
            reloaded = get_model_class(MODEL).from_pretrained(
                directory, dtype=torch.float32
            )

        reloaded_weight = reloaded.model.language_model.layers[1].mlp.experts.down_proj
        torch.testing.assert_close(reloaded_weight.detach(), modified)

        # The model was merged, so it must be reloaded for the next use.
        self.assertTrue(model.needs_reload)
        self.assertFalse(model.reset_model())
        self.assertFalse(model.has_modified_fused_experts())

    def test_merge_4bit_branch_contains_modified_weights(self) -> None:
        # The 4-bit branch reloads the base model on the CPU and copies the modified
        # fused weights into it. It depends only on the setting, not on bitsandbytes.
        model = get_model()
        model.apply_lora(1)
        handle = model.get_layer_fused_experts(2)["mlp.experts.down_proj"][0]
        model.begin_fused_edit(handle)
        original = model._fused_backups[handle.key].clone()
        ablate_fused_experts(
            handle.parameter,
            handle.row_axis,
            F.normalize(torch.randn(8), dim=0),
            1.0,
            RowNormalization.PRE,
        )
        modified = handle.parameter.detach().clone()
        self.assertFalse(torch.equal(modified, original))

        model.settings.quantization = QuantizationMethod.BNB_4BIT
        try:
            merged = model.get_merged_model()
        finally:
            model.settings.quantization = QuantizationMethod.NONE
            # The model still carries the adapter and the modified fused weights,
            # so the next test must start from a fresh load.
            model.needs_reload = True

        merged_weight = merged.model.language_model.layers[2].mlp.experts.down_proj
        self.assertTrue(torch.equal(merged_weight.detach(), modified))
        with tempfile.TemporaryDirectory() as directory:
            merged.save_pretrained(directory)
            reloaded = get_model_class(MODEL).from_pretrained(
                directory, dtype=torch.float32
            )
        reloaded_weight = reloaded.model.language_model.layers[2].mlp.experts.down_proj
        self.assertTrue(torch.equal(reloaded_weight.detach(), modified))

    def test_adapter_export_is_refused(self) -> None:
        model = get_model()
        handle = model.get_layer_fused_experts(0)["mlp.experts.down_proj"][0]
        model.settings.export_strategy = ExportStrategy.ADAPTER

        try:
            # Nothing modified yet, so an adapter would be exact.
            self.assertEqual(
                obtain_export_strategy(model.settings, model), ExportStrategy.ADAPTER
            )

            model.begin_fused_edit(handle)
            self.assertIsNone(obtain_export_strategy(model.settings, model))

            # Merging remains possible.
            model.settings.export_strategy = ExportStrategy.MERGE
            self.assertEqual(
                obtain_export_strategy(model.settings, model), ExportStrategy.MERGE
            )
        finally:
            model.settings.export_strategy = None


class AraWarningTests(unittest.TestCase):
    def test_ara_warns_about_fused_experts(self) -> None:
        warning = get_fused_experts_warning(get_model())
        assert warning is not None
        self.assertIn("Abliteration", warning)
        self.assertIn("routed experts", warning)


if __name__ == "__main__":
    unittest.main()
