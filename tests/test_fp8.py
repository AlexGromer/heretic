# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests for the support of checkpoints whose weights are stored in FP8
# (quant_method "fp8", as in Devstral-Small-2-24B-Instruct-2512 or Qwen3.5 FP8).
# They run on CPU: a FP8 checkpoint is derived from the tiny model that is also used
# by the end-to-end test in tests/mistral-3 by quantizing its decoder layers by hand,
# in the layout in which transformers saves them (weight, weight_scale_inv and, for
# the static activation scheme, activation_scale).
#
# Not covered here: the GPU kernel path (loading without dequantization), because
# it is not available on CPU, and checkpoints of other FP8 formats (compressed-tensors).

import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import OfflineModeIsEnabled
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText

from heretic.config import QuantizationMethod, Settings
from heretic.model import Model, get_fp8_quantization_config
from heretic.modifier import Context
from heretic.modifiers.abliteration import (
    Abliteration,
    Parameters,
    RowNormalization,
    WeightDistribution,
)
from heretic.modifiers.abliteration import Settings as AbliterationSettings

MODEL = "tiny-random/mistral-3"
MODEL_COMMIT = "931aa2e5c9668fc3679e56aa44972fe18597d55d"

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

# The linear layers of the text decoder that are quantized in the checkpoint.
QUANTIZED = re.compile(
    r"^language_model\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)

_source_directory: Path | None = None


def get_source_directory() -> Path:
    global _source_directory

    if _source_directory is None:
        try:
            path = snapshot_download(MODEL, revision=MODEL_COMMIT)
        except OfflineModeIsEnabled:
            path = snapshot_download(
                MODEL, revision=MODEL_COMMIT, local_files_only=True
            )

        _source_directory = Path(path)

    return _source_directory


def quantize(
    weight: torch.Tensor,
    block_size: tuple[int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the FP8 tensor and the scale that has to be multiplied with it
    (stored as weight_scale_inv by transformers).
    """
    weight = weight.to(torch.float32)

    if block_size is None:
        # One scalar per tensor.
        scale = (weight.abs().max() / FP8_MAX).clamp(min=1e-12)
        return (weight / scale).to(torch.float8_e4m3fn), scale.to(torch.float32)

    block_rows, block_columns = block_size
    rows, columns = weight.shape
    assert rows % block_rows == 0 and columns % block_columns == 0
    blocks = weight.reshape(
        rows // block_rows, block_rows, columns // block_columns, block_columns
    )
    scale = (blocks.abs().amax(dim=(1, 3), keepdim=True) / FP8_MAX).clamp(min=1e-12)
    quantized = (blocks / scale).to(torch.float8_e4m3fn).reshape(rows, columns)
    return quantized, scale.reshape(rows // block_rows, columns // block_columns)


def dequantize(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    block_size: tuple[int, int] | None,
) -> torch.Tensor:
    """Reference dequantization, independent of transformers."""
    if block_size is None:
        return (quantized.to(torch.float32) * scale).to(torch.bfloat16)

    block_rows, block_columns = block_size
    expanded = scale.repeat_interleave(block_rows, dim=0).repeat_interleave(
        block_columns, dim=1
    )
    return (quantized.to(torch.float32) * expanded).to(torch.bfloat16)


def make_fp8_checkpoint(
    directory: Path,
    block_size: tuple[int, int] | None,
    activation_scheme: str = "static",
) -> dict[str, torch.Tensor]:
    """
    Writes an FP8 version of the tiny model to the directory. Returns the reference
    bfloat16 dequantization of every quantized weight, by checkpoint key.
    """
    source = get_source_directory()

    for file in source.iterdir():
        if file.name not in ("model.safetensors", "config.json"):
            shutil.copy(file.resolve(), directory / file.name)

    tensors: dict[str, torch.Tensor] = {}
    reference: dict[str, torch.Tensor] = {}

    with safe_open(source / "model.safetensors", "pt") as file:
        for key in file.keys():  # noqa: SIM118
            tensor = file.get_tensor(key)

            if QUANTIZED.match(key):
                quantized, scale = quantize(tensor, block_size)
                tensors[key] = quantized
                tensors[key + "_scale_inv"] = scale
                if activation_scheme == "static":
                    # Not used when dequantizing, but present in such checkpoints.
                    tensors[key.removesuffix("weight") + "activation_scale"] = (
                        torch.tensor(1.0)
                    )
                reference[key] = dequantize(quantized, scale, block_size)
            else:
                tensors[key] = tensor

    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})

    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": activation_scheme,
        "weight_block_size": list(block_size) if block_size is not None else None,
        "modules_to_not_convert": [
            "language_model.model.embed_tokens",
            "language_model.lm_head",
            "multi_modal_projector",
            "vision_tower",
        ],
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2))

    return reference


def load_model(path: Path | str, **overrides) -> Model:
    previous_directory = os.getcwd()
    # An empty directory ensures that no config.toml is picked up.
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        try:
            with patch.object(sys, "argv", ["heretic"]):
                settings = Settings(
                    model=str(path),
                    dtypes=["bfloat16"],
                    device_map="cpu",
                    batch_size=2,
                    **overrides,
                )
            return Model(settings)
        finally:
            os.chdir(previous_directory)


def get_text_modules(model: Model) -> dict[str, torch.nn.Linear]:
    """Maps checkpoint keys to the modules that hold the quantized weights."""
    modules = {}

    for layer_index, layer in enumerate(model.get_layers()):
        prefix = f"language_model.model.layers.{layer_index}."
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            modules[f"{prefix}self_attn.{name}.weight"] = getattr(layer.self_attn, name)
        for name in ("gate_proj", "up_proj", "down_proj"):
            modules[f"{prefix}mlp.{name}.weight"] = getattr(layer.mlp, name)

    return modules


def get_weight(module: torch.nn.Linear) -> torch.Tensor:
    return module.weight.detach()


def abliterate(model: Model, weight: float = 1.0) -> None:
    """Runs the real Abliteration.modify_model with a random residual direction."""
    layer_count = len(model.get_layers())
    hidden_size = model.model.config.get_text_config().hidden_size

    torch.manual_seed(0)
    modifier = Abliteration.__new__(Abliteration)
    modifier.settings = AbliterationSettings(row_normalization=RowNormalization.NONE)
    modifier.heretic_settings = model.settings
    modifier.residual_directions = torch.nn.functional.normalize(
        torch.randn(layer_count + 1, hidden_size), dim=1
    )

    distribution = WeightDistribution(
        max_weight=weight,
        max_weight_position=layer_count / 2,
        min_weight=weight,
        min_weight_distance=float(layer_count + 1),
    )
    modifier.modify_model(
        cast(Context, SimpleNamespace(get_model=lambda: model)),
        Parameters(
            direction_index=None,
            weight_distributions={
                "attn.o_proj": distribution,
                "mlp.down_proj": distribution,
            },
        ),
    )


class FP8Tests(unittest.TestCase):
    block_size: tuple[int, int] | None = (8, 8)

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary_directory.name)
        cls.reference = make_fp8_checkpoint(cls.directory, cls.block_size)

        # Original (unquantized) weights for error bounds.
        cls.original = {}
        with safe_open(get_source_directory() / "model.safetensors", "pt") as file:
            for key in cls.reference:
                cls.original[key] = file.get_tensor(key)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_checkpoint_really_is_fp8(self) -> None:
        with safe_open(self.directory / "model.safetensors", "pt") as file:
            key = "language_model.model.layers.0.self_attn.q_proj.weight"
            self.assertEqual(file.get_tensor(key).dtype, torch.float8_e4m3fn)
            self.assertIn(key + "_scale_inv", file.keys())

        quantization_config = get_fp8_quantization_config(str(self.directory))
        assert quantization_config is not None
        self.assertEqual(quantization_config["quant_method"], "fp8")

    def test_non_fp8_models_are_not_detected(self) -> None:
        self.assertIsNone(get_fp8_quantization_config(str(get_source_directory())))

        for quantization_config in (
            {"quant_method": "gptq"},
            {"quant_method": "compressed-tensors"},
            {"quant_method": "fbgemm_fp8"},
        ):
            with patch(
                "heretic.model.PretrainedConfig.get_config_dict",
                return_value=({"quantization_config": quantization_config}, {}),
            ):
                self.assertIsNone(get_fp8_quantization_config("any"))

    def test_loads_dequantized_to_bfloat16(self) -> None:
        model = load_model(self.directory)

        self.assertIsNotNone(model.fp8_quantization_config)
        self.assertEqual(model.dtype, torch.bfloat16)
        self.assertFalse(getattr(model.model, "is_quantized", False))

        modules = get_text_modules(model)
        self.assertEqual(set(modules), set(self.reference))

        for key, module in modules.items():
            weight = get_weight(module)
            self.assertEqual(weight.dtype, torch.bfloat16, key)
            # Exactly the reference dequantization ...
            torch.testing.assert_close(weight, self.reference[key], rtol=0, atol=0)
            # ... which is within the FP8 error of the original weights.
            original = self.original[key].to(torch.float32)
            error = (weight.to(torch.float32) - original).abs().max()
            self.assertLess(error.item(), 0.08 * original.abs().max().item(), key)

        # No FP8 tensors or leftover scale parameters anywhere in the model.
        for name, parameter in model.model.named_parameters():
            self.assertNotEqual(parameter.dtype, torch.float8_e4m3fn, name)
            self.assertNotIn("scale", name)

    def test_layer_modules_are_found(self) -> None:
        model = load_model(self.directory)

        layer_count = len(model.get_layers())
        for layer_index in range(layer_count):
            components = model.get_layer_modules(layer_index)
            self.assertEqual(len(components["attn.o_proj"]), 1)
            self.assertEqual(len(components["mlp.down_proj"]), 1)

    def test_abliteration_and_merged_export_are_plain_bfloat16(self) -> None:
        model = load_model(self.directory)
        model.apply_lora(1)
        abliterate(model)

        merged = model.get_merged_model()

        # The merged model no longer carries any trace of FP8.
        self.assertFalse(hasattr(merged.config, "quantization_config"))
        self.assertIsNone(getattr(merged, "hf_quantizer", None))

        with tempfile.TemporaryDirectory() as output:
            merged.save_pretrained(output)

            config = json.loads((Path(output) / "config.json").read_text())
            self.assertNotIn("quantization_config", config)
            self.assertNotIn("quantization_config", config.get("text_config", {}))

            keys = {}
            for file_path in Path(output).glob("*.safetensors"):
                with safe_open(file_path, "pt") as file:
                    for key in file.keys():  # noqa: SIM118
                        keys[key] = file.get_tensor(key)

            for key, tensor in keys.items():
                self.assertNotEqual(tensor.dtype, torch.float8_e4m3fn, key)
                self.assertNotIn("scale", key)

            # The checkpoint is a plain one that loads without any quantization.
            reloaded = AutoModelForImageTextToText.from_pretrained(
                output, dtype=torch.bfloat16
            )
            self.assertFalse(getattr(reloaded, "is_quantized", False))

            # The saved layout may differ from the checkpoint's (Transformers saves
            # the current format), so identify layers by their index and name.
            def identify(key: str) -> str:
                match = re.search(r"layers\.(\d+)\.(.*)$", key)
                assert match is not None
                return f"{match.group(1)}.{match.group(2)}"

            saved = {
                identify(key): tensor
                for key, tensor in keys.items()
                if "language_model" in key and "layers." in key
            }

            changed = []
            unchanged = []
            for key in get_text_modules(model):
                if torch.equal(saved[identify(key)], self.reference[key]):
                    unchanged.append(key)
                else:
                    changed.append(key)

            # Abliteration modified exactly the components it targets.
            self.assertTrue(changed)
            self.assertTrue(
                all(
                    key.endswith(("o_proj.weight", "down_proj.weight"))
                    for key in changed
                )
            )
            self.assertTrue(any("q_proj" in key for key in unchanged))
            self.assertEqual(
                len(changed),
                2 * len(model.get_layers()),
            )

    def test_reset_after_merge_reloads_dequantized(self) -> None:
        model = load_model(self.directory)
        model.apply_lora(1)
        abliterate(model)
        model.get_merged_model()

        # Slow path: the model is loaded again, which must dequantize again.
        self.assertFalse(model.reset_model())
        for key, module in get_text_modules(model).items():
            torch.testing.assert_close(
                module.weight, self.reference[key], rtol=0, atol=0
            )

    def test_bnb_4bit_is_rejected(self) -> None:
        with self.assertRaises(Exception) as context:
            load_model(self.directory, quantization=QuantizationMethod.BNB_4BIT)
        self.assertIn("FP8", str(context.exception))

    def test_not_fp8_models_are_unchanged(self) -> None:
        model = load_model(get_source_directory())

        self.assertIsNone(model.fp8_quantization_config)
        self.assertIsNone(model._get_quantization_config("bfloat16"))


class FP8PerTensorTests(FP8Tests):
    # Checkpoints without weight_block_size (as of Devstral-Small-2) have one
    # scalar scale per tensor.
    block_size = None


if __name__ == "__main__":
    unittest.main()
