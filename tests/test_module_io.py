# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Tests for recording the inputs and outputs of modules (as needed by ARA) when
# the hidden states are flattened inside MoE blocks or modules only see some of the
# tokens, as in Qwen3.5 MoE. They run on CPU with the tiny model that is also used
# by the end-to-end test in tests/qwen3.5-moe.

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import Tensor
from torch.nn import Module

from heretic.config import Settings
from heretic.model import Model, reshape_module_io
from heretic.modifier import Context
from heretic.modifiers.ara import ARA
from heretic.modifiers.ara import Parameters as ARAParameters
from heretic.modifiers.ara import Settings as ARASettings
from heretic.utils import Prompt

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


def get_shared_down_projection(layer: Module) -> Module:
    return cast(Any, layer).mlp.shared_expert.down_proj


def get_lora_b(module: Module) -> Tensor:
    return cast(Any, module).lora_B["default"].weight


class ModuleIOTests(unittest.TestCase):
    def test_reshape_module_io(self) -> None:
        three_dimensional = torch.arange(24.0).view(2, 3, 4)
        torch.testing.assert_close(
            reshape_module_io(three_dimensional, (2, 3)), three_dimensional[:, -1, :]
        )

        flattened = three_dimensional.reshape(6, 4)
        torch.testing.assert_close(
            reshape_module_io(flattened, (2, 3)), three_dimensional[:, -1, :]
        )

        # Sparse input, as seen by routed experts: not one row per token.
        self.assertIsNone(reshape_module_io(torch.zeros(5, 4), (2, 3)))

    def test_get_module_io_handles_flattened_shared_expert(self) -> None:
        model = get_model()
        prompts = [
            Prompt(system="You are helpful.", user="What is 1+1?"),
            Prompt(system="You are helpful.", user="Name a color."),
            Prompt(system="You are helpful.", user="Say hello."),
        ]

        module_io = model.get_module_io_batched(prompts)

        self.assertEqual(len(module_io), len(model.get_layers()))
        for layer in module_io:
            module_input, module_output = layer["mlp.down_proj"][0]
            # (prompt, features): the intermediate size in, the hidden size out.
            self.assertEqual(tuple(module_input.shape), (3, 32))
            self.assertEqual(tuple(module_output.shape), (3, 8))
            self.assertIn("attn.o_proj", layer)

    def test_get_module_io_skips_sparse_inputs_with_one_warning(self) -> None:
        model = get_model()
        prompts = [
            Prompt(system="You are helpful.", user="a"),
            Prompt(system="You are helpful.", user="bb bb"),
            Prompt(system="You are helpful.", user="ccc ccc ccc"),
        ]

        # Make the shared expert's down projection see a single row, like a routed
        # expert that only receives some of the tokens. Broadcasting keeps the
        # forward pass valid.
        handles = [
            get_shared_down_projection(layer).register_forward_pre_hook(
                lambda module, args: (args[0][:1],)
            )
            for layer in model.get_layers()
        ]
        model._warnings_shown.discard("sparse-io:mlp.down_proj")
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                module_io = model.get_module_io_batched(prompts)
        finally:
            for handle in handles:
                handle.remove()

        # Warned exactly once for the component, across all layers and batches.
        self.assertEqual(buffer.getvalue().count("cannot be recorded"), 1)
        for layer in module_io:
            # The skipped component is absent rather than an empty mapping.
            self.assertNotIn("mlp.down_proj", layer)
            # Other components are still recorded.
            self.assertEqual(tuple(layer["attn.o_proj"][0][0].shape[0:1]), (3,))

    def test_ara_skips_modules_without_recorded_io(self) -> None:
        model = get_model()
        model.apply_lora(2)
        good = [Prompt(system="s", user=u) for u in ("a", "bb bb", "ccc ccc ccc")]
        bad = [Prompt(system="s", user=u) for u in ("d d", "eee", "ff ff ff ff")]

        # The shared expert's input is sparse, so its I/O is not recorded.
        handles = [
            get_shared_down_projection(layer).register_forward_pre_hook(
                lambda module, args: (args[0][:1],)
            )
            for layer in model.get_layers()
        ]
        try:
            with redirect_stdout(io.StringIO()):
                good_io = model.get_module_io_batched(good)
                bad_io = model.get_module_io_batched(bad)
        finally:
            for handle in handles:
                handle.remove()

        modifier = ARA.__new__(ARA)
        model.settings.seed = 1
        modifier.heretic_settings = model.settings
        modifier.settings = ARASettings(lora_rank=2, n_optimization_steps=1, max_iter=2)
        modifier.good_module_io = good_io
        modifier.bad_module_io = bad_io
        layer_count = len(model.get_layers())

        # Must not raise a KeyError for the component without recorded I/O.
        with redirect_stdout(io.StringIO()):
            modifier.modify_model(
                cast(Context, SimpleNamespace(get_model=lambda: model)),
                ARAParameters(0, layer_count, 0.5, 0.1, 0.5, 1),
            )

        model.settings.seed = None

        # The skipped modules are untouched (LoRA B starts at zero), the others are optimized.
        shared_b = [
            get_lora_b(module)
            for layer_index in range(layer_count)
            for module in model.get_layer_modules(layer_index)["mlp.down_proj"]
        ]
        self.assertTrue(all(not b.any() for b in shared_b))
        other_b = [
            get_lora_b(module)
            for layer_index in range(layer_count)
            for component, modules in model.get_layer_modules(layer_index).items()
            if component != "mlp.down_proj"
            for module in modules
        ]
        self.assertTrue(any(b.any() for b in other_b))


if __name__ == "__main__":
    unittest.main()
