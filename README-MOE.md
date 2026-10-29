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

# v3: wider ranges, strict refusal counting, expert-granular abliteration

Branch `moe-fused-v3`, on top of `moe-fused-v2` (`95dcea1`). Three independent options; with all of
them off (the defaults) behaviour is that of v2. Tests: `tests/test_v3_*.py` (CPU, tiny models or
pure functions). Example configs: `config.wide.toml` (dense), `config.moe-ega.toml` (MoE).

Not run on a GPU and not on the real Qwen3.6/3.8 models. What the options do to refusals and KL on
real models is **not measured**; the example configs are starting points.

## 1. Wider search ranges

The fixed Optuna ranges of the modifiers are settings now, with the upstream values as defaults
(a test pins them to the old hard-coded bounds).

* `[modifier.Abliteration]`: `direction_index_range`, `max_weight_range`, `mlp_max_weight_range`,
  `max_weight_position_range`, `min_weight_range`, `min_weight_distance_min_layers`,
  `min_weight_distance_max_fraction`.
* `[modifier.ARA]`: `start_layer_range`, `end_layer_range`, `preserve_good_weight_range`,
  `steer_bad_weight_range`, `overcorrect_range`, `neighbor_count_max`.

Validation happens when the config is loaded: `low <= high` (equal pins the parameter), finite
numbers, no negative weights (only the lower bound of `mlp_max_weight_range` may be negative, it is
clamped to 0 as upstream does), fractions within `[0, 1]`, `steer_bad_weight_range` strictly
positive (log-uniform sampling). ARA layer ranges may overlap now; the sampled end is raised to
`start + 1` afterwards if needed (the default ranges never need it).

## 2. Strict keyword scorer

`[scorer.KeywordRate] strict = true` (default false; the upstream counting is untouched when off):

* chain-of-thought blocks (`chain_of_thought_blocks`, e.g. `<think>...</think>`) are removed before
  matching, including an unterminated block (generation ended inside the thinking) and a closing
  tag without opening tag (templates that open `<think>` in the prompt, like Qwen3 thinking mode);
* a leading disclaimer paragraph (`Disclaimer`, `Warning:`, `Note:`, ...) is dropped if the answer
  continues after it;
* only the first `strict_window_tokens` (default 256) tokens of the answer are inspected (real
  tokens when the tokenizer is available). Responses are only `max_response_length` long, so set
  that to 256 as well;
* `strict_markers` replace `keyword_markers`: the default list without `disclaimer`, `violat`,
  `prohibit`, `illegal`, `harmful`, `inappropriate`, `unethical`, `ethical boundaries`, plus
  `i don't want`, `i don't create`, `cannot assist`, `can't help`, `i must decline` and variants;
* `strict_empty_answer_is_match` (default false): whether a response that is empty only after
  removing the thinking counts as a refusal. A response that is empty to begin with always does.

Rationale (research report 2026-10-29): the upstream markers fire on disclaimers of fully compliant
answers and miss some phrasings; part of the "refusals" of the optimization may be false positives.
A refusal followed by compliance is deliberately still counted (it cannot be told apart from a soft
refusal by markers); only a judge model separates the two.

## 3. Expert-granular abliteration (EGA)

`[modifier.Abliteration] ega = true` (default false) for models with fused routed experts
(`qwen3_5_moe` and other models with the same `gate_up_proj` / `down_proj` / `act_fn` layout).

* For every layer and routed expert the fork runs the good and bad prompts and records the
  expert's own output (before the routing weight) for the tokens routed to it, at the last
  `ega_last_positions` (default 4) positions of each prompt.
* The direction of an expert is the normalized difference of its mean outputs on bad and good
  prompts, made orthogonal to its mean good output when `orthogonalize_direction` is on.
* The ablation strength of an expert is the trial's `mlp.down_proj` weight times a factor from the
  expert's routing frequency on the bad prompts (`ega_frequency_scaling`: `max` = frequency divided
  by the layer maximum, `mean`, or `none`). An expert with fewer than `ega_min_tokens` routed bad
  tokens is unused and not modified at all (bit-exact). A used expert with too few good tokens uses
  the layer's residual direction.
* The shared expert and the attention outputs are ablated like a dense MLP, exactly as without EGA.
* The directions are computed once on the unmodified model. Original weights are restored between
  trials from the v2 backup (`fused_experts_backup_device`); EGA adds no second copy.
* The search space does not change. `direction_scope` only affects the Linear modules and the
  fallback; an expert uses the direction of its own layer.
* Experts with another weight layout are skipped with a warning and ablated with the layer direction.

Sources of the idea (from the research report 2026-10-29, `abliteration-2026-10-29/report.md`; the
numbers are the authors' own and not comparable with ours):

* TrevorS/gemma-4-abliteration (Apache-2.0): heretic with EGA; Gemma 4 26B-A4B refusals 29/100
  without EGA, 3/100 with EGA.
* abliterix (wuwangzhang1216/abliterix, AGPL-3.0, derivative of heretic): MoE expert-granular
  abliteration, router suppression, FP8 handling.
* OBLITERATUS (elder-plinius, AGPL-3.0 and commercial licence): `surgical` preset with EGA.

This implementation follows the method description in the task specification and the fork's own
code. No code of these projects was copied, so none of it is derived work; the credit is for the idea.
Router pre-bias (heretic issue #322) is not implemented.
