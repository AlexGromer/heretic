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
  `direction_index_range` may end at 1.0 (the last layer): the global direction then is the direction
  of the last layer, there is no row to interpolate with beyond it.
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
  continues after it, unless that paragraph itself contains a refusal marker (`Warning: I'm sorry,
  but I can't help with that.` stays a refusal); liability phrasings such as "I can't be held
  responsible" do not count as refusal markers there;
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

# v4: OR-Bench-hard as a second bad_eval, upstream issue fixes

Branch `moe-fused-v4`, on top of `moe-fused-v3` (`15c1048`). Every addition is off by default or
behaves like v3 unless configured. Tests: `tests/test_v4.py` (CPU; the ARA tests use the tiny MoE
model). Nothing here was run on a GPU or on the real Qwen3 / Devstral models.

## 1. OR-Bench-hard as a second refusal evaluation set

Follow-up item 2 of the research report: "OR-Bench-hard as a second bad_eval for Optuna, only if
the first cycle shows that the keyword scorer overfits our bad_eval". Every trial is picked by its
refusal count on the same 100 prompts (`mlabonne/harmful_behaviors` `test[:100]`), so the Pareto
front can reward edits that remove refusals on exactly those prompts. `config.orbench.toml` adds
a second `KeywordRate` instance (`orbench`, shown as `Refusals - orbench`) on 100 prompts of
[OR-Bench-hard-1K](https://huggingface.co/datasets/bench-llm/or-bench) (`or-bench-hard-1k`,
CC-BY-4.0, 1319 seemingly harmful prompts that many models refuse; not used for training).

* **Sampling.** OR-Bench is sorted by category; `train[:100]` covers only `deception` and
  `harassment`. Dataset specifications take `sample_size` and `sample_seed` now: a reproducible
  random draw from the selected split (order kept). Works for HF datasets, local datasets and
  text files.
* **Cycle 1.** The instance runs with `optimization = "none"`: it is evaluated and shown on every
  trial but does not steer the search. Afterwards
  `heretic-generalization checkpoints/<model>.jsonl` reads the study checkpoint and compares, over
  the Pareto trials with KL <= 0.15, the removed fraction of baseline refusals on the evaluation
  prompts with the removed fraction on OR-Bench. It reports overfitting (exit code 1) when the
  median gap exceeds 0.2, or when the Spearman correlation of the two refusal rates over all trials
  is below 0.5 (thresholds are options). It warns when the OR-Bench baseline is below 0.1 (then
  the set hardly triggers refusals on that model and says little).
* **Cycle 2, only if overfitting is reported.** Set the instance to `optimization = "minimize"`
  and use a new `study_checkpoint_dir`: a study cannot change its number of objectives, and
  "Continue the previous run" restores the settings stored in the old checkpoint. Alternative with
  two objectives: list both datasets under `[scorer.KeywordRate.prompts]`.
* `or-bench-toxic` (655 truly harmful prompts) can serve as a held-out set of the same kind as
  the evaluation prompts (`config = "or-bench-toxic"`).

The thresholds are judgement calls, not calibrated on real runs.

## 2. Responses file (audit of counted refusals, upstream #122)

`responses_file = "responses.jsonl"` (top-level setting): every `KeywordRate` evaluation appends
one JSON line per prompt (evaluation number of that scorer, 0 = baseline; score name; system
prompt; prompt; response; match). This is what the research report's first step needs: read the
"refusals" of the 3/100 and 4/100 trials by hand and label false positives (also the dataset
upstream asks for in #290). The path is `exclude=True`, so it is not stored in the checkpoint or
in uploaded configs.

## 3. Fixes for open upstream issues

* **#487** (Qwen3 and other self-thinking models): the "Rechecking with prefix" step appended the
  common beginning of the reasoning (`\nOkay,`) to the response prefix, keeping the model in CoT
  mode while it writes the "answer". Only the leading whitespace of the common prefix extends the
  prefix now; the rest is reported as ignored. Same idea as upstream PR #501 (own code).
* **#535** (crash without a terminal after the optimization): every question first checks that
  standard input is a terminal. If not, Heretic stops with a message naming the flag that answers
  it (`--trial-index`, `--model-action`, `--save-directory`, ...) and lists the Pareto trials with
  their `--trial-index` values; the checkpoint keeps the results. Exit code 1, no traceback.
* **#533** (GGUF repositories): a model whose repository or folder only holds `.gguf` files is
  rejected before loading with a message to use the original repository, instead of a tokenizer
  traceback. If the file list cannot be fetched (offline), loading proceeds as before.
* **#522** ("a deep, subtle flaw in ARA"): two `[modifier.ARA]` options, defaults unchanged.
  `ridge_weight` adds a penalty that keeps the whole matrix close to the original, including the
  directions that the 400+400 recorded inputs do not span (maintainer's option 1). The penalty is
  `||B A||_F^2 * E[||x||^2] / (in * out)`, the mean squared output change on inputs of the
  recorded mean norm in random directions, i.e. in the units of the "preserve good" term, so that
  one weight transfers across models better than a raw Frobenius weight (not measured; try
  0.1 to 1). `line_search = "none"` is the maintainer's option 2; a module whose adapter diverges
  (non-finite adapter or loss) is left unmodified instead of breaking the model.

Not addressed in v4 (see v5): #490, router pre-bias (#322), SOMA (#140). #436 (tencent/Hy3-FP8)
may load with the FP8 dequantization of v2 but was not tried.

# v5: merged-model scores, multi-directional abliteration, router pre-bias, local judge

Same branch `moe-fused-v4`. All four are off by default. Tests: `tests/test_v5.py` (CPU; the
model tests use `tiny-random/qwen3.5-moe`). None of it was run on a GPU or on a real model, so
nothing here says how refusals and KL change.

## 1. Re-scoring on merged models (upstream #490)

Trials are scored on the live LoRA, which applies the base matrix and the adapter separately
and rounds in between; the export runs the merged matrix, so the exported model can score
differently (the maintainer's example: KL 0.15 -> 0.17, 3 -> 5 refusals).
`n_merged_score_trials = N` evaluates the first N trials of the sorted Pareto front again on
their merged models after the optimization and shows `live -> merged` in the menu. If the whole
front was re-scored, the menu is sorted by the merged scores (which changes what `--trial-index`
points to). The merged records are stored in the study, so showing the results again or running
more trials does not recompute them. Cost per trial: one merge, one evaluation, one model reload;
with 4-bit quantization the merge reloads the base model on the CPU. A failed merge keeps the
live scores. Approach as in upstream PR #495, own code.

## 2. Multi-directional abliteration with self-organizing maps (upstream #140)

`[modifier.Abliteration] direction_count = k` (default 1). From "SOM Directions are Better than
One" (arXiv 2511.08379; upstream PR #196 reported gemma-3-12b 3/100 at KL 0.08 vs 0.16, self-reported):

* a batch self-organizing map (`som_grid`, `som_epochs`, `som_sigma`; plain PyTorch, no new
  dependency, deterministic) is fitted to the bad-prompt residuals of every layer;
* the prototypes with the most hits minus the good mean are candidate directions; they are made
  orthogonal (Gram-Schmidt) to the classic mean-difference direction, to the good direction
  (with `orthogonalize_direction`) and to each other; near-duplicates are dropped;
* the classic direction stays direction 0 with weight 1; each additional direction gets a
  relative weight `direction_{i}.weight` in `direction_weight_range` (shared by all components).
  With all weights 0 the result is classic abliteration, so the search space contains it;
* Linear modules get `delta W = -weight * V^T diag(w) V W` (orthonormal V, so it is a weighted
  projection), LoRA rank `k` (`full_normalization_lora_rank + k - 1` with "full" row
  normalization); fused experts are ablated direction by direction (exact for "none"; with row
  normalization each step renormalizes); with `direction_scope = global` the extras of the
  nearest layer are used, made orthogonal to the interpolated direction.

Differences from PR #196: no `minisom`; directions are orthonormal (PR #196 summed rank-1 updates
of non-orthogonal directions); its "top-k neurons" counted winners among the prototypes
themselves, here hits are counted over the residuals; one weight per extra direction instead of a
full weight curve per direction and component (k - 1 new search parameters instead of 4 per
direction and component).

## 3. Router pre-bias for MoE models (upstream #322)

`[modifier.Abliteration] router_bias = true`. The routing of the last
`router_bias_last_positions` positions of the good and bad prompts is counted per expert, and the
router logits get `-alpha * log(p_bad / p_good)` (smoothed, clipped at `router_bias_max_log_ratio`),
with `alpha` searched in `router_bias_alpha_range` (the issue reports a sharp optimum near 0.5 and
worse results at 1.0 on Qwen3.5/3.6-35B-A3B, self-reported). One alpha for all MoE layers; the
issue's best result used four layers with different alphas, which is not searched here.

The issue adds the bias at inference. Here it is stored in the weights, so that it survives the
export to safetensors and GGUF:

* routers with a bias parameter (e.g. gpt-oss) get it added to the bias, exactly;
* routers without one (Qwen3.5 MoE: `mlp.gate`, weight `(experts, hidden)`, no bias) get
  `b u^T / c` added to the weight, with `u` the unit mean direction of the router inputs and `c`
  their mean projection onto `u`. An input `h` then gets `b (u.h) / c`: exactly `b` for the mean
  input, more or less for others. The spread of `u.h / c` over sampled inputs is printed per layer
  at startup; a large spread means the fold is a poor approximation of a constant bias.

Router edits use the same backups as the fused experts (bit-exact restore between trials,
`fused_experts_backup_device`) and force merging on export. Found routers: a `gate` or `router`
module in `mlp` / `block_sparse_moe` / `moe` / `feed_forward` with a `weight` of shape
`(experts, hidden)` and a `top_k` (or `config.num_experts_per_tok`). 4-bit quantized routers are
skipped. Not combined with ARA (ARA does not use the Abliteration settings).

## 4. Local judge

Superseded by v6, section 1 (the judge now follows the evaluation harness exactly).

## Not done

Comparing with abliterix on Devstral-Small-2-24B is an experiment, not a code change: it needs
the BF16 model (about 48 GB) on a large GPU and running both tools on the same prompts with the
same KL definition. Nothing in the fork blocks it.

# v6: against PARTIAL answers

Same branch `moe-fused-v4`. Measured with the the evaluation harness judge (refusal / partial / comply, %):
Qwen3-8B base 52.1 / 32.3 / 15.7, trial 18 1.4 / 46.5 / 52.1; Qwen3-14B base 47.9 / 27.6 / 24.4,
trial 90 41.9 / 35.0 / 12.9; Qwen3.8-27B base 89.9 / 6.9 / 3.2, trial 96 8.8 / 55.8 / 35.5.
Abliteration turns refusals into PARTIAL answers, and the keyword metric cannot see them (on
Qwen3-14B it picked a trial that kept 42% refusals). Everything below is off by default, tested on
CPU with tiny models (`tests/test_v6.py`), and not measured on real models.

## 1. The harness judge in Heretic

`heretic/judge.py` implements the protocol of `the evaluation harness/judge.py`, so numbers are comparable:
the same system prompt (built-in byte-identical copy, prompt version `v1-039fd0fa`; or
`rubric = "../the evaluation harness/judge.py:SYSTEM_PROMPT"`, read by parsing the file, not executing it),
the same user message and clipping, the same CoT stripping, Ollama `/api/chat` with `think: false`,
the JSON schema, temperature 0, seed 0, 3 retries, and the same parser. Labels REFUSAL, PARTIAL,
COMPLY, OFF_TOPIC; REFUSAL + PARTIAL count as not answered. Settings in `[scorer.JudgeRate]`
(endpoint, model, `api = "openai"` for OpenAI-compatible servers, labels, rubric). Used by:

* (removed in branch `repo-split`: the `heretic-judge` command. Files are judged by the harness's
  `judge.py` / the harness's file judge; `responses_file` rows are in its format, see below);
* scorer `JudgeRate` (`optimization = "none"`; judges every trial, slow);
* **judge re-ranking** (`n_judge_ranking_trials = N`): after the optimization, the original model
  and the first N trials of the Pareto front with KL <= `judge_ranking_max_kl` generate answers to
  the `[scorer.JudgeRate]` prompts (`judge_ranking_response_length`, default 512 like the evaluation harness)
  and are judged. The menu lists the judged trials first, by share of non-answers, then by KL, with
  the label counts and the answers cut at the length limit. Stored in the study per prompt version.

## 2. Distilled classifier as an objective

`heretic-train-classifier --responses R.jsonl --judged J.jsonl --output classifiers/nonanswer`
fine-tunes ModernBERT-base (as in upstream PR #500) on judged answers (the evaluation harness files, joined by
row key, or rows with a label); validation holds out whole prompts and reports Cohen's kappa
against the judge (4-class and answered / not answered) and warns below 0.75. Scorer
`heretic.scorers.classifier_rate.ClassifierRate` (`model_path`) then scores 100 answers in seconds,
so the optimization can minimize REFUSAL + PARTIAL instead of keyword refusals. Use
`max_response_length` >= 256.

## 3. Directions where the evasion happens

`heretic/positions.py` takes activations at chosen tokens instead of the end of the prompt:

* **Response direction** (`[modifier.Abliteration] response_direction_file`): residuals averaged
  over the first `response_tokens` tokens of judged answers, PARTIAL minus COMPLY
  (`response_bad_labels`, `response_good_labels`), as an additional direction
  `partial_direction` with its own weight.
* **ARA on answer tokens** (`[modifier.ARA] response_io_file`): the module I/O averaged over the
  first answer tokens of PARTIAL answers is added to the bad I/O and that of COMPLY answers to the
  good I/O (the mean is exact for the linear modules ARA edits), so the loss pulls evasive answers
  towards real ones.
* **Harmfulness direction** (`harmfulness_direction = true`, Zhao et al. 2025, arXiv 2507.11878):
  bad minus good prompts at the last token of the user instruction, as `harmfulness_direction`.
  Experimental.

Additional directions are made orthonormal to the classic one, the good direction and each other,
and each gets a weight in `direction_weight_range`. Answer files: rows with prompt, response
and judge_label / label, or the harness's `responses.jsonl` plus `judged.jsonl`
(`*_judged_file`). Use answers to training prompts, not the evaluation hold-out.

## 4. DPO on the model's own answers

Answers are sampled and judged by the evaluation harness (the harness's generator with several sampled answers per prompt, then the harness's file judge); `heretic-dpo-pairs pairs --responses R --judged J` builds pairs (chosen COMPLY, rejected PARTIAL or
REFUSAL, same prompt; cut-off chosen answers dropped) in TRL's conversational format, and trains a
DPO LoRA (mlabonne's recipe: r64, alpha 32, lr 5e-6, 1 epoch, beta 0.1; `--merge` saves the merged
model). `train` needs TRL (`uv pip install trl`, not a Heretic dependency). Whether DPO lowers
PARTIAL is a hypothesis: measure judge rates, KL and GSM8K.

## 5. Smaller fixes

* KL divergence masks tokens with -inf baseline logits (reserved tokens, e.g. Gemma 4 12B in
  TrevorS's recipe); without non-finite values it is computed as before.
* `config.partial.toml`: ARA with judge re-ranking, answer I/O and technical good prompts, plus the
  second-pass recipe in comments.
* the evaluation harness (branch `partial-diagnostics`): judged rows keep `finish`, the summary counts PARTIAL
  answers cut at the token limit, its generator's system-prompt option for an
  authorized-pentest system prompt.

# Handing ranked trials to an external judge

For setups where the judge runs in its own process and must not share the GPU with the evaluated
model (e.g. the harness's judge process); how a pipeline uses this is the pipeline's business:

* `judge_ranking_mode = "generate"`: after the optimization, Heretic writes the answers of the
  original model (`trial` "base") and of the best `n_judge_ranking_trials` trials within
  `judge_ranking_max_kl` to `judge_ranking_output`, in the harness's `responses.jsonl` format
  (same key as the harness's key format, greedy, thinking off, `finish`, `win100` / `win256`,
  `settings.generator = "heretic"`). Keys already in the file are skipped, so an interrupted run
  resumes. Without a terminal, Heretic exits after writing. Prompts:
  `judge_ranking_prompts_file` (JSON Lines with id, category, prompt, e.g.
  `train/bad_eval.meta.jsonl`), set name `judge_ranking_set`, model label `judge_ranking_model_name`.
* `judge_ranking_pool = "all"` (default `"front"`): in the `generate` mode the candidates are taken
  from every completed trial within `judge_ranking_max_kl`, highest KL divergence first (ties by
  trial number), not only from the Pareto front, because the keyword scorer cannot see the
  non-answers without refusal words that decide whether a trial off the front is better.
* `judge_ranking_merged = true` (default): each ranked trial is merged into the weights in memory
  before its answers are generated, so they come from the same weights an export contains (the
  live LoRA rounds differently, upstream #490). One model reload per trial, no export to disk.
  Rows record `settings.operator` ("merged" or "live LoRA").
* Ranking on an evaluation harness evaluation hold-out (`data/offsec_holdout*.jsonl`, `harmless*.jsonl`,
  the v3 monitor files) prints a warning: rank on a validation set such as `train/bad_eval` and keep
  the hold-out for the final measurement of the selected trial.
* `--trial-number N` selects the trial shown as `[Trial N]` (the `trial` field of the rows), from
  all completed trials, independent of the menu order; `--trial-index` still counts menu positions.
* the runner configs (paths, Drive layout, the two-run recipe) belong to the runner repository, which
  generates its own `config.*.cyber.toml`; this branch only adds the Heretic code they can call.

Tests: `tests/test_judge_handoff.py`.

# Repository boundaries

This fork changes models and judges only inside its own process. It contains no paths, settings or
file names of any particular evaluation harness or runner, so it can be published or offered
upstream without carrying private ones:

* **This fork** reads prompts from files or Hugging Face datasets, writes its answers as JSON Lines
  rows (`heretic/rows.py`: key, model, trial, mode, thinking, set, prompt_id, category, prompt,
  response, settings), and judges only through the endpoint of `[scorer.JudgeRate]`. Judging a
  file, generating evaluation answers, summarizing and comparing runs are not its job. The example
  configs use placeholder paths (`prompts/...`); whoever runs it supplies the real ones.
* **The evaluation harness** (a separate repository) owns the prompt sets, the generators, the file
  judge, the summaries and the run comparison. Its row format is what `heretic/rows.py` writes, so
  its judge reads Heretic's files unchanged; the built-in rubric is a copy of its judge prompt, and
  `[scorer.JudgeRate] rubric = "<harness>/judge.py:SYSTEM_PROMPT"` reads that prompt from there
  instead (parsed, not executed).
* **The runner** (another repository) owns the environment: paths, drive layout, the loop over
  models, the stages and which config a model gets.

Removed from this fork for that reason: the `heretic-judge` command, `heretic-dpo-pairs generate`
and `judge` (the harness generates and judges), the runner config, and the hard-coded hold-out file
names, which became `judge_ranking_hold_out_patterns` (globs like `*holdout*`, so the convention
check names no file of anyone's harness).

# v7: metrics that see more, and more levers

Same fork, branch `metrics-depth`. Everything is off by default. Tests: `tests/test_v7.py`
(CPU, tiny models). Nothing here was run on a GPU or on a real model, so the effect on refusals,
PARTIAL and capabilities is unmeasured.

## 1. KL over an answer, not only its first token

`heretic.scorers.sequence_kl.SequenceKLDivergence`: the built-in KL compares the distribution of
the FIRST generated token on harmless prompts, so damage that only shows later in an answer never
enters it. This records, once on the original model, the greedy continuation of every prompt
(`tokens`, default 32) with the original distributions along it, and measures the divergence at the
same positions by teacher forcing. Only the `top_k` (default 128) most likely original tokens are
kept per position, with both sides renormalized over that support; the covered share of the
original mass is printed at startup (normally above 0.99). The display also names the worst prompt,
since a mean hides a single derailed answer. Cost: one forward pass over prompt + `tokens`.

## 2. Capability benchmarks on the candidates

`judge_ranking_benchmarks = ["gsm8k:exact_match,strict-match", "mmlu"]` runs those lm-eval tasks on
the original model and on the trials the judge ranking already looks at, prints each candidate's
difference from the original model, and stores the numbers in the study. The `BenchmarkScore`
scorer runs on every trial, which GSM8K cannot afford; this runs on the few finalists. Motivation:
arXiv 2512.13655 reports GSM8K changes from +1.51 to -18.81 points across tools and models.

## 3. Directions where the evasion is, by language and by kind

All three add a slot in the same mechanism as v5/v6 (orthonormal to the classic direction, the good
direction and each other, each with its own weight in `direction_weight_range`):

* `language_directions = true`: the bad prompts are split by script (`ru` / `en`, the same rule as
  the harness's `detect_lang`), and every language with at least `language_min_prompts` prompts
  gets its own direction. The classic direction is dominated by whichever language most of the
  prompts are in - in `train/bad_train` 436 of 612 prompts are Russian while `harmful_behaviors` is
  English.
* `response_subtype_patterns`: the judge writes a one-line reason for every label, so the
  non-answers can be split by it ("general overview", "asks for clarification", "educational
  pivot") and each kind gets its own direction instead of one for all of them. A kind with fewer
  than `response_subtype_min_answers` matching answers is skipped.
* `response_thinking_direction = true`: a direction from the tokens inside the reasoning blocks of
  the judged answers. Thinking models decide there whether to answer (upstream #360). Answers
  without a reasoning block are skipped.

## 4. Router pre-bias per layer, and Gemma 4

* `router_bias_per_layer = true` searches one alpha per MoE layer (`router_bias_alpha.<layer>`)
  instead of one for all of them. An alpha of 0 leaves a layer's router untouched, so the search
  can pick a subset of layers, as the best result in upstream #322 did (four layers, different
  alphas). One search parameter per MoE layer, so raise `n_trials` with it.
* Gemma 4 text keeps its experts and router on the decoder layer (`layer.experts`, `layer.router`)
  instead of inside an MLP block, and the router holds its projection in a child module
  (`router.proj`) behind an RMS norm. Both are found now, and the bias is folded into that
  projection, whose recorded inputs are what the weight multiplies. The expert layout is the one
  EGA already supports (`gate_up_proj` / `down_proj` / `act_fn`), so EGA covers Gemma 4 MoE too -
  verified against the Transformers 5.20.2 source and a stub of that layout, not on the real model.

## 5. Optimizing non-answers, and several studies

`config.judge-objective.toml` makes `ClassifierRate` (the classifier distilled from the judge) the
objective instead of the keyword counter, with both KL scorers beside it and the keyword counter
kept for comparison. Its header has the order of the prerequisites (judged answers, judge
calibration kappa >= 0.75, trained classifier) and the recipe for running two or three independent
studies into separate `study_checkpoint_dir` journals and comparing their candidates with
the harness's paired run comparison; TPE runs differ a lot (upstream #88, #202), while more trials in one
study do not bring refusals down by themselves (#404).

Not done here: wiring the two-stage selection, the new fork patch and `--trial-number` into
the runner repository (that repository is being changed elsewhere).

# v8: statistics over counterfactual pairs

Branch `paired-directions`. the harness's paired training sets are
counterfactual twins at the SAME index: 612 pairs, the same language on both sides in all of
them, the same three technical terms in 522, the same opening words in 522 and the same final
punctuation in 583. Inside a pair everything but the refusal cancels, which the mean of the two
sets cannot do. Three settings use that; all default to the previous behaviour, and all of them
require both sets to have the same length and order (an unpaired pair of sets is refused with a
message naming the counts).

Note what does **not** change: the mean of the per-pair differences is exactly the difference of
the means, so the classic direction is the same whether the sets are paired or not (a test pins
this). What the pairing buys is everything computed from the *spread* of the differences.

* `[modifier.Abliteration] direction_estimator = "whitened"`: the mean-difference direction
  corrected by the covariance of the per-pair differences (LDA-like, the "whitened diff" of the
  representation-engineering literature). The plain difference treats every axis alike, although
  the pairs vary far more along some of them; whitening favours the axes along which the pairs
  barely vary, which is where a clean refusal signal sits. With fewer pairs than dimensions the
  covariance is singular, so `whitening_shrinkage` (default 0.1 of the mean eigenvalue) is added;
  a large value goes back to `mean_diff`. Computed from the thin SVD of the differences, never
  from a hidden x hidden matrix. Cost: the per-prompt residuals of both sets are kept in memory
  (`prompts * (layers + 1) * hidden * 4` bytes each).
* `extra_direction_source = "paired_svd"`: the additional directions of `direction_count > 1` come
  from the singular vectors of the per-pair differences instead of the self-organizing map. A SOM
  cluster is a group of harmful prompts (so language and topic are still in it); a singular vector
  of the differences is a direction of refusal the pairing has already cleaned.
* `[modifier.ARA] pairing = "index"`: every bad output is pulled towards the good output at the
  same index, its own counterfactual twin, instead of towards the mean of its `neighbor_count`
  nearest good outputs. The twin is the safe version of the very same request, so the target is
  exact rather than an average over unrelated requests; `neighbor_count` is then unused. With
  `response_io_file` the judged answers must add equally many rows to both sides, or the run stops.

The row of the embeddings has no direction under pairing (it holds the embedding of the last
prompt token, which the pairing makes identical); Heretic never ablates that row, since a layer
uses the row after it.

Tests: `tests/test_v8.py`. Not run on a GPU or on a real model: whether a whitened direction or
index pairing lowers refusals or KL is unmeasured.

# v9: the checks themselves

Branch `polish`. Housekeeping that makes the other versions trustworthy, plus two small
diagnostics.

* **CI runs pytest.** Some of this fork's tests are plain test functions, which
  `unittest discover` cannot collect, so until now they ran only locally: eight of them were
  invisible to CI. The workflow runs `pytest -q tests/` instead, which also runs every unittest
  class, and `pytest` is a dev dependency.
* **Questions are built lazily.** `ask_if_unset` and `ask` accept a callable that builds the
  question, and every menu in `main.py` / `reproduce.py` passes one. Building a question creates a
  prompt_toolkit application, which needs a usable console, so a fully configured run (every answer
  given as a setting) no longer constructs any. That also fixes the only test that failed on
  Windows for the whole series (`test_adapter_export_is_refused`), which had nothing to do with
  export strategies: the question was built as an argument before `ask_if_unset` could see the
  value. A test now pins that no question is built eagerly.
* **Pair diagnostics at the token level.** The paired estimators and ARA's index pairing print how
  many pairs share a language and how many share the LAST TOKEN OF THE INSTRUCTION. The hidden
  state the directions are read from sits at the end of the rendered prompt, whose final tokens
  come from the chat template and are identical for every prompt, so what differs there is the
  instruction's own last token. This is feedback for whoever builds the prompt sets.
* **Memory of the paired estimators.** The estimate (both sets of per-prompt residuals) is printed
  before they are computed, and `paired_residuals_dtype = "float16"` halves it. Whitening is
  sensitive to that precision (see the setting's description), so the default stays float32.
* **Per-category non-answers.** The judge ranking stores the share of non-answers per prompt
  category (the categories of the ranking prompts) and prints the worst three next to each trial,
  so a single domain dragging the rate down is visible without opening the files.
* **`tools/make_fork_patch.sh`** builds the `src/` patch that turns an upstream checkout into this
  fork and prints its provenance (fork commit, upstream base, file count, sha256), which is what a
  runner needs to vendor it. Only `src/` travels: configs and tests stay here.

Tests: `tests/test_v9.py`.

# v10: the handoff as it is actually called

Branch `resume-settings`. Four things that the end-to-end path through a Colab runner needs and
that the previous versions got wrong or left out. All of them were found by reading the calls, not
by a failing test; the tests came after.

* **A resumed study no longer overrides the knobs of the single run.** `--checkpoint-action
  continue` validates the settings stored in the finished study, so everything that is not
  `exclude=True` comes from the optimization run and not from the command line (that is why a
  runner has to write `trial_index` into the stored settings to pick a trial). The settings that
  only describe what this one invocation should do after the optimization are therefore excluded
  now: `trial_number`, `n_merged_score_trials` and every `judge_ranking_*` setting
  (`n_judge_ranking_trials`, `judge_ranking_mode`, `..._max_kl`, `..._set`, `..._model_name`,
  `..._response_length`, `..._batch_size`, `..._merged`, `..._benchmarks`,
  `..._hold_out_patterns`, `..._holdout_version`). Before this, `heretic --trial-number N
  --judge-ranking-mode generate ...` over an existing journal silently did what the optimization
  run had asked for: the trial number was dropped and nothing was generated. The study's journals
  written so far carry the old keys, and the excluded settings now lose to the current run, which
  is what a second invocation means.
* **A plain text prompts file.** `judge_ranking_prompts_file` reads JSON Lines with `id`,
  `category` and `prompt` (the harness's `.meta.jsonl`) **or** one prompt per line, because a
  Colab bundle carries the plain text sets and not the meta files. A text line that happens to be
  valid JSON is only read as a row when it is an object with a `prompt`.
* **The generated rows carry `holdout_version` and `lang`.** The harness groups its summaries by
  `(model, trial, mode, thinking, set, holdout_version)` and reads a row without a
  `holdout_version` as hold-out v1, which ranking answers are not; it also reports rates per
  prompt language from `lang`. The new `judge_ranking_holdout_version` defaults to `"train"`, the
  harness's own stamp for the training sets, and `lang` comes from the same rule as the harness's
  `detect_lang`.
* **What still cannot be called in Colab** (runner work, not this repository): the notebook bundle
  carries `eval/data/*.jsonl` (the hold-out) and `cyber/*.txt` (the training sets) only, so
  `config.orbench.toml`'s validation set (`val/orbench_val.*`) and the `.meta.jsonl` ids and
  categories never arrive there. Ranking in Colab therefore runs on `cyber/bad_eval.txt` with
  synthetic ids and no categories, which the per-category diagnostics of v9 cannot use.

Tests: the new cases in `tests/test_judge_handoff.py` (what a resumed run keeps, both prompt file
formats, the two new row fields).
