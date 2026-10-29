# heretic-moe: abliteration of fused routed experts

Fork of [Heretic](https://github.com/p-e-w/heretic) (AGPL-3.0-or-later), branch `moe-fused`,
based on upstream master `efb1be6`. Everything is a patch on top of upstream:
`heretic-moe-fused.patch` (`git diff efb1be6..moe-fused`).

## The problem

`Qwen/Qwen3.6-35B-A3B` (`qwen3_5_moe`, `Qwen3_5MoeForConditionalGeneration`) with
transformers 5.x and the default ARA modifier crashed with

    heretic/model.py: input = inputs[0][:, -1, :]  ->  IndexError: too many indices for tensor of dimension 2

Reproduced on `tiny-random/qwen3.5-moe` with upstream `efb1be6` and transformers 5.20.2.
Two independent causes:

1. **Fused experts are invisible.** In transformers v5 the routed experts of `qwen3_5_moe`
   (and dozens of other MoE models) are one module `Qwen3_5MoeExperts` holding 3D parameters
   `gate_up_proj (E, 2*I, H)` and `down_proj (E, H, I)`; they are applied as
   `F.linear(x, down_proj[e])`, so axis 1 of `down_proj` is the output (hidden)
   dimension, exactly like the rows of a `Linear` weight. `get_layer_modules` only knew
   iterable experts (`for expert in layer.mlp.experts`), the loop raised inside
   `suppress(Exception)`, and the routed experts were **silently skipped** (upstream
   issue #339). Only `attn.o_proj` / `linear_attn.out_proj` and `shared_expert.down_proj`
   were ablated.
2. **The ARA hook assumed 3D tensors.** `Qwen3_5MoeSparseMoeBlock` flattens the hidden
   states to `(prompt*position, hidden)` before calling the shared expert, so the hook's
   `[:, -1, :]` failed.

## What changed

* **Discovery** (`Model.get_layer_fused_experts`): generic. Any 3D parameter named
  `down_proj` / `w2` (or the `weight` of an `output_linear`) inside the layer's MoE block
  is a fused down projection. The hidden axis is determined from the shape and
  `config.hidden_size` and asserted; if it is ambiguous (e.g. hidden size equals the
  intermediate size) the tensor is skipped **with a warning** instead of guessing,
  except for model types with a known layout (`gpt_oss`: `(E, I, H)`, verified from the
  transformers source). Fused experts are returned as `FusedExperts` handles under the
  component `mlp.experts.down_proj`, separately from `get_layer_modules`, which still
  only returns `nn.Module`s (PEFT LoRA targets), so ARA/LoRA code is unaffected.
  The startup printout shows e.g. `mlp.experts.down_proj (fused): 40 tensors with 10240 experts total`.
* **Abliteration** (`ablate_fused_experts`): for each expert `e`,
  `W_e <- W_e - lambda * v (v^T W_e)`, with the layer direction `v` and the
  `mlp.down_proj` weight `lambda` of the current trial. The fused experts reuse the
  `mlp.down_proj` Optuna parameters, so the search space is identical to a model with
  separate experts (`get_abliterable_components` adds `mlp.down_proj` for models whose
  only MLP down projections are fused). `row_normalization` has the same semantics as for
  `Linear`: `none` plain, `pre` ablation on row-normalized weights scaled back by the row
  norms, `full` additionally renormalizes rows to their original norms (MPOA). Unlike the
  Linear path, `full` is exact (no rank-`r` SVD approximation), since the weights are edited
  directly. The edit is in place and chunked (default 2**26 elements, 256 MiB of float32
  temporaries per chunk); no full copy of the tensor is made on the GPU.
* **Reversibility** (`begin_fused_edit` / `restore_fused_experts`): the first edit of a
  tensor stores its original on `fused_experts_backup_device` (default `cpu`), and
  `reset_model` copies the originals back bit-exactly (in-place edits of bf16 weights cannot
  be undone arithmetically). A full reload (after merging) drops the backups.
* **Export**: `merge` saves the modified fused weights (checked with a
  `save_pretrained` / reload test; transformers converts the fused tensors back to the
  checkpoint layout). With bitsandbytes 4-bit, where the base model is reloaded for the merge,
  the modified fused tensors are copied into the reloaded model. `export_strategy =
  "adapter"` is refused with a clear message when fused experts were modified; the
  interactive menu then offers only merging.
* **ARA / `get_module_io`**: the batch shape `(prompt, position)` is recorded per layer
  with a forward pre-hook; flattened `(prompt*position, features)` inputs/outputs are
  reshaped; modules that see a sparse 2D input are skipped with a one-time warning (ARA skips
  modules without recorded I/O). Selecting ARA on a model with fused experts prints a
  prominent warning that ARA leaves routed experts unmodified.
* New setting `fused_experts_backup_device` (config.default.toml).

## Limits

* Only the down projection is modified (as for all other models); `gate_up_proj` is untouched.
* Biases (`down_proj_bias`, e.g. gpt-oss) are not ablated, same as `Linear` biases upstream.
* ARA does not touch routed experts. Use the `Abliteration` modifier on MoE models.
* Backups of modified layers need system RAM roughly equal to the modified fused tensors
  (all layers, bf16: `layers * experts * H * I * 2` bytes; ~10 GB for a 35B-A3B class model;
  check the model config).
* Edits and the Linear LoRA parts of `Abliteration` are independent; Linear `full`
  normalization stays low-rank approximate.
* `gpt_oss` orientation and Granite-style `output_linear` are implemented from the
  transformers source but not covered by CPU tests here. The 4-bit merge path (base model
  reloaded on CPU, modified fused weights copied in) is covered by
  `test_merge_4bit_branch_contains_modified_weights`; it only depends on the setting, not on
  bitsandbytes or a GPU.
* Only validated on CPU with `tiny-random/qwen3.5-moe` and transformers 5.20.2; not run on
  the real 35B model.

## How to run

    git clone <upstream> heretic && cd heretic && git checkout efb1be6
    git apply heretic-moe-fused.patch            # or use branch moe-fused
    uv sync && uv pip install "transformers[kernels]==5.20.2"
    cp config.moe.toml config.toml               # Qwen/Qwen3.6-35B-A3B
    uv run --no-sync heretic

(`uv sync` follows `uv.lock`, which pins transformers 5.18; install 5.20.2 afterwards and
use `--no-sync`, or `UV_NO_SYNC=1`, so uv does not revert it.)

Tests: `pytest tests/test_*.py` (CPU, tiny model, ~1 min). End-to-end hashes:
`UV_NO_SYNC=1 python tests/run_tests.py`; `tests/qwen3.5-moe/SHA256SUMS.{linux,ci}` were
updated because that test uses `Abliteration`, which now actually modifies the routed
experts, so `model.safetensors` changes by design. `config.json` / `generation_config.json`
hashes depend on the transformers version and differ under 5.20.2 for every test case
(identically for upstream).
