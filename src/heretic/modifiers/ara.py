# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Arbitrary-Rank Ablation (ARA) (Weidmann 2026)
# See https://github.com/p-e-w/heretic/pull/211 for more information.

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Literal, cast

import bitsandbytes.functional as BNB_F
import torch
import torch.linalg as LA
from optuna import Trial
from peft.tuners.lora.layer import Linear
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    PositiveInt,
    ValidationInfo,
    field_validator,
)
from torch import Tensor
from torch.optim import LBFGS

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.model import Model
from heretic.modifier import Context, Modifier, Serializable
from heretic.positions import (
    append_module_io,
    format_pair_diagnostics,
    load_answer_items,
    module_io_means,
    pair_diagnostics,
)
from heretic.ranges import check_range, scaled_floor
from heretic.utils import format_dataset_specification, print


class Pairing(str, Enum):
    # Each bad output is pulled towards its k nearest good outputs (upstream).
    KNN = "knn"
    # Each bad output is pulled towards the good output at the same index (its twin).
    INDEX = "index"


@dataclass
class Parameters(Serializable):
    start_layer_index: int
    end_layer_index: int
    preserve_good_behavior_weight: float
    steer_bad_behavior_weight: float
    overcorrect_relative_weight: float
    neighbor_count: int
    # "knn" (upstream) or "index" (fork addition); stored so a trial restores its own mode.
    pairing: str = Pairing.KNN.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_presentation_dict(self) -> dict[str, str]:
        return {
            name: (f"{value:.4f}" if isinstance(value, float) else f"{value}")
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        return Parameters(**data)


class Settings(BaseModel):
    good_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
        ),
        description="Dataset of prompts that tend to produce desirable responses.",
    )

    bad_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
        ),
        description="Dataset of prompts that tend to produce undesirable responses.",
    )

    preserve_row_magnitudes: bool = Field(
        default=True,
        description=(
            "Whether to renormalize the rows of the modified matrices to preserve "
            "the original matrices' row magnitudes. This is believed to improve "
            'intelligence retention (see Lai 2025, "Magnitude-Preserving Orthogonal Ablation").'
        ),
    )

    lora_rank: PositiveInt = Field(
        default=50,
        description=(
            "The rank of the LoRA adapter to use. "
            'While mathematically, ARA is of "arbitrary" rank, experiments have shown that '
            "singular values tend to drop rapidly after a few dozen dimensions, and approximating "
            "the full transformation with a LoRA has many practical advantages."
        ),
    )

    n_optimization_steps: PositiveInt = Field(
        default=5,
        description="Number of (outer) L-BFGS optimization steps to perform.",
    )

    learning_rate: float = Field(
        default=1.0,
        description="Learning rate to use in the L-BFGS optimizer.",
    )

    max_iter: PositiveInt = Field(
        default=20,
        description="Maximum number of (inner) iterations to perform per (outer) L-BFGS optimization step.",
    )

    history_size: PositiveInt = Field(
        default=10,
        description="Number of past updates to store for approximating the Hessian matrix in the L-BFGS optimizer.",
    )

    print_loss: bool = Field(
        default=False,
        description="Whether to print the loss value for each L-BFGS optimization step.",
    )

    pairing: Pairing = Field(
        default=Pairing.KNN,
        description=(
            'How a bad output finds its target. "knn": the neighbor_count nearest good outputs '
            '(upstream). "index": the good output at the SAME index, i.e. the counterfactual '
            "twin of that request (the harness paired training set and train/good_train are built "
            "that way). The twin differs from its bad prompt only in the refusal, so the target "
            "is exact instead of an average of unrelated requests, and neighbor_count is then "
            "unused. Both prompt sets must have the same length and order."
        ),
    )

    # Judged answers whose module I/O is added to the prompt I/O (fork addition, see
    # heretic/positions.py and the Abliteration settings of the same names).
    response_io_file: str | None = Field(
        default=None,
        description=(
            "JSON Lines file of judged answers (prompt, response and judge_label, as written by "
            "the harness responses.jsonl with response_io_judged_file, or rows with judge_label). The "
            "module I/O averaged over the first response_tokens answer tokens of answers labelled "
            "response_bad_labels is added to the bad I/O, that of response_good_labels to the "
            "good I/O (by default PARTIAL and COMPLY)."
        ),
    )

    response_io_judged_file: str | None = Field(
        default=None,
        description="harness judged.jsonl whose labels are joined to response_io_file by row key.",
    )

    response_bad_labels: list[str] = Field(
        default=["PARTIAL"],
        description="Judge labels of the answers added to the bad I/O.",
    )

    response_good_labels: list[str] = Field(
        default=["COMPLY"],
        description="Judge labels of the answers added to the good I/O.",
    )

    response_tokens: PositiveInt = Field(
        default=32,
        description="Number of answer tokens averaged per answer.",
    )

    response_max_samples: PositiveInt = Field(
        default=200,
        description="Maximum number of answers per side (a fixed random subset is used beyond).",
    )

    # Fork additions for upstream issue #522: the loss only sees the subspace spanned by the
    # recorded inputs, so the optimizer can change the matrix freely outside of it, and the
    # line search can amplify such changes. They do not affect the loss but do affect the model.
    ridge_weight: NonNegativeFloat = Field(
        default=0.0,
        description=(
            "Weight of a ridge penalty that keeps the whole matrix close to the original, "
            "including the directions the recorded inputs do not cover (upstream issue #522). "
            "The penalty is the mean squared output change the edit would cause on inputs of "
            "the recorded inputs' mean norm pointing in random directions, i.e. it has the units "
            "of the term that preserves the outputs for good prompts. 0 disables it (upstream behavior)."
        ),
    )

    line_search: Literal["strong_wolfe", "none"] = Field(
        default="strong_wolfe",
        description=(
            'Line search of the L-BFGS optimizer. "none" uses fixed steps of learning_rate, '
            "which avoids the arbitrarily long steps that can blow up changes outside the "
            "recorded subspace (upstream issue #522), but can diverge; a module whose adapter "
            "becomes non-finite is left unmodified."
        ),
    )

    # Search ranges (fork addition). The defaults are the fixed ranges of upstream Heretic;
    # widen them to let the optimizer explore stronger steering.
    start_layer_range: tuple[float, float] = Field(
        default=(0.0, 0.5),
        description=(
            "Range of the first modified layer, as fractions of the number of layers."
        ),
    )

    end_layer_range: tuple[float, float] = Field(
        default=(0.5, 1.0),
        description=(
            "Range of the end (exclusive) of the modified layers, as fractions of the number "
            "of layers. The lowest sampled end is floor(lower * layers) + 1. "
            "If the sampled end does not exceed the sampled start (possible only when "
            "the two ranges overlap), it is raised to start + 1."
        ),
    )

    preserve_good_weight_range: tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Range of the weight of the term that preserves the outputs for good prompts.",
    )

    steer_bad_weight_range: tuple[float, float] = Field(
        default=(0.0001, 1.0),
        description=(
            "Range of the weight of the term that steers the outputs for bad prompts "
            "(sampled log-uniformly, so the lower bound must be positive)."
        ),
    )

    overcorrect_range: tuple[float, float] = Field(
        default=(0.0, 1.3),
        description=(
            "Range of the relative weight of the overcorrection (pushing bad outputs "
            "away from their original position). Values above 1.3 steer more strongly "
            "but can damage the model."
        ),
    )

    neighbor_count_max: PositiveInt = Field(
        default=15,
        description=(
            "Upper bound of the number of nearest neighbors "
            "(also limited by the number of prompts)."
        ),
    )

    @field_validator(
        "start_layer_range",
        "end_layer_range",
    )
    @classmethod
    def validate_layer_ranges(
        cls, value: tuple[float, float], info: ValidationInfo
    ) -> tuple[float, float]:
        return check_range(info.field_name, value, lower_limit=0.0, upper_limit=1.0)

    @field_validator("preserve_good_weight_range", "overcorrect_range")
    @classmethod
    def validate_weight_ranges(
        cls, value: tuple[float, float], info: ValidationInfo
    ) -> tuple[float, float]:
        return check_range(info.field_name, value, lower_limit=0.0)

    @field_validator("steer_bad_weight_range")
    @classmethod
    def validate_steer_bad_weight_range(
        cls, value: tuple[float, float]
    ) -> tuple[float, float]:
        return check_range("steer_bad_weight_range", value, strictly_positive=True)


# For each vector in the 2D-tensor `a`, computes the mean Euclidean distance
# to the `k` nearest neighbors of the vector among the vectors in the 2D-tensor `b`.
def mean_distances_to_knn(a: Tensor, b: Tensor, k: int) -> Tensor:
    distances = torch.cdist(a, b)
    nearest_distances, _ = distances.topk(k, dim=1, largest=False)
    return nearest_distances.mean(1)


def distances_to_twin(a: Tensor, b: Tensor) -> Tensor:
    """
    The distance of every vector in `a` to the vector at the same index in `b` (fork
    addition). With counterfactual pairs (the harness paired training set and
    train/good_train are twins at the same index) this is the distance to the safe version
    of the very same request, instead of to an average of k unrelated safe requests.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Index pairing needs matching rows, got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    return (a - b).norm(dim=1)


def steering_distances(
    new_bad_output: Tensor,
    target: Tensor,
    parameters: Parameters,
) -> Tensor:
    if parameters.pairing == Pairing.INDEX.value:
        return distances_to_twin(new_bad_output, target)
    return mean_distances_to_knn(new_bad_output, target, parameters.neighbor_count)


def lowrank_frobenius_sq(A: Tensor, B: Tensor) -> Tensor:
    """
    ||B @ A||_F^2 = trace((B^T B) (A A^T)), computed from two rank x rank matrices
    instead of the dense out_features x in_features product.
    """
    return ((B.T @ B) * (A @ A.T)).sum()


# The objective function at the heart of ARA.
def ara_loss(
    good_output: Tensor,
    bad_output: Tensor,
    new_good_output: Tensor,
    new_bad_output: Tensor,
    parameters: Parameters,
) -> Tensor:
    # The outputs for "good" prompts should change as little as possible.
    preserve_good_behavior = ((new_good_output - good_output) ** 2).mean()

    steer_bad_behavior = (
        # Pull the outputs for "bad" prompts towards
        # the original outputs for "good" prompts.
        steering_distances(
            new_bad_output,
            good_output,
            parameters,
        ).mean()
        # Push the outputs for "bad" prompts away from
        # the original outputs for "bad" prompts.
        # In combination with the above, this overcorrects
        # away from the original residuals, which results
        # in stronger steering that can overcome more complex
        # refusal mechanisms.
        + parameters.overcorrect_relative_weight
        * -steering_distances(
            new_bad_output,
            bad_output,
            parameters,
        ).mean()
    )

    return (
        parameters.preserve_good_behavior_weight * preserve_good_behavior
        + parameters.steer_bad_behavior_weight * steer_bad_behavior
    )


def get_fused_experts_warning(model: Model) -> str | None:
    """
    Returns a warning for models with fused routed experts, or None for other models.
    """
    if not model.has_fused_experts():
        return None

    return (
        "[yellow bold]WARNING: This model has fused routed experts "
        "(one 3D weight tensor per layer instead of a Linear module per expert).[/]\n"
        "[yellow]ARA operates on Linear modules only and leaves the routed experts unmodified; "
        "only the attention outputs and the shared expert are ablated, "
        "which is usually not enough to remove refusals from a MoE model. "
        "The Abliteration modifier handles fused routed experts "
        '(modifiers = [ { plugin = "heretic.modifiers.abliteration.Abliteration" } ]).[/]'
    )


class ARA(Modifier[Parameters]):
    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def modifier_name(self) -> str:
        return "Arbitrary-Rank Ablation (ARA)"

    def init(self, ctx: Context) -> None:
        model = ctx.get_model()

        print()
        print(
            f"Loading good prompts from [bold]{format_dataset_specification(self.settings.good_prompts)}[/]..."
        )
        good_prompts = ctx.load_prompts(self.settings.good_prompts)
        print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

        print()
        print(
            f"Loading bad prompts from [bold]{format_dataset_specification(self.settings.bad_prompts)}[/]..."
        )
        bad_prompts = ctx.load_prompts(self.settings.bad_prompts)
        print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

        if self.settings.pairing == Pairing.INDEX:
            if len(bad_prompts) != len(good_prompts):
                raise ValueError(
                    'pairing = "index" compares the prompts at the same index, but there are '
                    f"{len(bad_prompts)} bad and {len(good_prompts)} good prompts. Use paired "
                    'sets (the harness paired training set and train/good_train) or pairing = "knn".'
                )
            print()
            print(
                "* Index pairing: "
                + format_pair_diagnostics(
                    pair_diagnostics(model, bad_prompts, good_prompts)
                )
            )

        fused_experts_warning = get_fused_experts_warning(model)
        if fused_experts_warning is not None:
            print()
            print(fused_experts_warning)

        self.good_prompt_count = len(good_prompts)
        self.bad_prompt_count = len(bad_prompts)

        print()
        print("Obtaining module I/O for good prompts...")
        self.good_module_io = model.get_module_io_batched(good_prompts)

        print()
        print("Obtaining module I/O for bad prompts...")
        self.bad_module_io = model.get_module_io_batched(bad_prompts)

        if self.settings.response_io_file is not None:
            # Fork addition: the module I/O of judged answers, averaged over their first
            # answer tokens, is added to the bad (e.g. PARTIAL) and good (e.g. COMPLY)
            # I/O, so that the loss also pulls the outputs of evasive answers towards
            # those of real answers, where the evasion happens (not only at the end of
            # the prompt).
            print()
            print("Obtaining module I/O for judged answers...")
            bad_items, good_items = load_answer_items(
                model,
                self.settings.response_io_file,
                self.settings.response_io_judged_file,
                self.settings.response_bad_labels,
                self.settings.response_good_labels,
                self.settings.response_tokens,
                self.settings.response_max_samples,
            )
            self.bad_module_io = append_module_io(
                self.bad_module_io, module_io_means(model, bad_items)
            )
            self.good_module_io = append_module_io(
                self.good_module_io, module_io_means(model, good_items)
            )
            self.bad_prompt_count += len(bad_items)
            self.good_prompt_count += len(good_items)
            if self.settings.pairing == Pairing.INDEX and len(bad_items) != len(
                good_items
            ):
                raise ValueError(
                    'pairing = "index" needs the same number of rows on both sides, but the '
                    f"judged answers added {len(bad_items)} bad and {len(good_items)} good "
                    'ones. Use pairing = "knn" with judged answers, or add equally many.'
                )

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.
        model.apply_lora(self.settings.lora_rank)

    def suggest_parameters(self, ctx: Context, trial: Trial) -> Parameters:
        layer_count = len(ctx.get_model().get_layers())

        start_layer_index = trial.suggest_int(
            "start_layer_index",
            min(
                scaled_floor(self.settings.start_layer_range[0], layer_count),
                layer_count - 1,
            ),
            # At least one layer must remain after the start.
            min(
                scaled_floor(self.settings.start_layer_range[1], layer_count),
                layer_count - 1,
            ),
        )
        # The end index is exclusive, so starting its range one above the lower bound
        # (the start range's upper bound in the default configuration) ensures that
        # the layer range is never empty.
        end_layer_index = trial.suggest_int(
            "end_layer_index",
            min(
                scaled_floor(self.settings.end_layer_range[0], layer_count) + 1,
                layer_count,
            ),
            max(
                scaled_floor(self.settings.end_layer_range[1], layer_count),
                min(
                    scaled_floor(self.settings.end_layer_range[0], layer_count) + 1,
                    layer_count,
                ),
            ),
        )
        # With overlapping ranges, the sampled end can fall at or below the start.
        # Raise it afterwards (instead of using a conditional range, which
        # multivariate TPE does not support). Never triggers with the defaults.
        end_layer_index = min(max(end_layer_index, start_layer_index + 1), layer_count)
        preserve_good_behavior_weight = trial.suggest_float(
            "preserve_good_behavior_weight",
            self.settings.preserve_good_weight_range[0],
            self.settings.preserve_good_weight_range[1],
        )
        steer_bad_behavior_weight = trial.suggest_float(
            "steer_bad_behavior_weight",
            self.settings.steer_bad_weight_range[0],
            self.settings.steer_bad_weight_range[1],
            log=True,
        )
        overcorrect_relative_weight = trial.suggest_float(
            "overcorrect_relative_weight",
            self.settings.overcorrect_range[0],
            self.settings.overcorrect_range[1],
        )
        # The nearest neighbors are selected from the outputs for the good and bad prompts,
        # which contain one vector per prompt, so there can't be more neighbors than prompts.
        neighbor_count = trial.suggest_int(
            "neighbor_count",
            1,
            min(
                self.settings.neighbor_count_max,
                self.good_prompt_count,
                self.bad_prompt_count,
            ),
        )

        return Parameters(
            start_layer_index=start_layer_index,
            end_layer_index=end_layer_index,
            preserve_good_behavior_weight=preserve_good_behavior_weight,
            steer_bad_behavior_weight=steer_bad_behavior_weight,
            overcorrect_relative_weight=overcorrect_relative_weight,
            neighbor_count=neighbor_count,
            pairing=self.settings.pairing.value,
        )

    def modify_model(self, ctx: Context, parameters: Parameters) -> None:
        model = ctx.get_model()

        # The model's reset zeroes both A and B, where all gradients vanish,
        # so A must be initialized below for the optimization to make progress.
        # Generating on the CPU keeps A independent of the device and the global RNG state.
        generator = torch.Generator().manual_seed(cast(int, self.heretic_settings.seed))

        for layer_index in range(
            parameters.start_layer_index,
            parameters.end_layer_index,
        ):
            for component, modules in model.get_layer_modules(layer_index).items():
                for module_index, module in enumerate(modules):
                    # Modules whose I/O could not be recorded (e.g. because they only
                    # see some of the tokens) cannot be optimized.
                    if module_index not in self.good_module_io[layer_index].get(
                        component, {}
                    ) or module_index not in self.bad_module_io[layer_index].get(
                        component, {}
                    ):
                        continue

                    # Cast to Linear to access weights and LoRA adapters.
                    module = cast(Linear, module)

                    # We need the base weight in float32 to compute the effective weight.
                    base_weight = cast(Tensor, module.base_layer.weight)
                    quant_state = getattr(base_weight, "quant_state", None)

                    if quant_state is None:
                        W_base = base_weight.to(torch.float32)
                    else:
                        # Use the original dequantization logic from bitsandbytes.
                        W_base = BNB_F.dequantize_4bit(
                            base_weight.data,
                            quant_state,
                        ).to(torch.float32)

                    # Pre-calculate the original row norms to preserve them.
                    # See https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
                    W_row_norms = cast(
                        Tensor,
                        LA.vector_norm(W_base, dim=1, keepdim=True).detach(),
                    )

                    # We optimize the LoRA weights A and B.
                    lora_A = cast(Tensor, module.lora_A["default"].weight)
                    lora_B = cast(Tensor, module.lora_B["default"].weight)

                    # Initialize A the same way PEFT does, but deterministically. See
                    # https://github.com/huggingface/peft/blob/v0.21.2/src/peft/tuners/lora/layer.py#L338
                    initial_A = torch.empty(lora_A.shape)
                    torch.nn.init.kaiming_uniform_(
                        initial_A,
                        a=math.sqrt(5),
                        generator=generator,
                    )
                    with torch.no_grad():
                        lora_A.copy_(initial_A)

                    # Move I/O tensors to the device of the adapter weights.
                    good_input, good_output = self.good_module_io[layer_index][
                        component
                    ][module_index]
                    bad_input, bad_output = self.bad_module_io[layer_index][component][
                        module_index
                    ]

                    good_input = good_input.float().to(lora_A.device)
                    good_output = good_output.float().to(lora_A.device)
                    bad_input = bad_input.float().to(lora_A.device)
                    bad_output = bad_output.float().to(lora_A.device)

                    # The base product input @ W_base.T does not depend on the adapter,
                    # so compute it once here (in float32, from the float32 weight)
                    # instead of on every closure evaluation. It must not be taken from
                    # the captured module output: that was computed in the model's dtype
                    # (e.g. bfloat16) and would change the objective.
                    good_product = good_input @ W_base.T
                    bad_product = bad_input @ W_base.T

                    # ARA only modifies the weight. If the module has a bias, the
                    # captured output includes it, so add it back to the modified output
                    # (after row normalization, which must not rescale the bias).
                    bias = cast(Tensor | None, module.base_layer.bias)
                    if bias is not None:
                        bias = bias.detach().float().to(lora_A.device)

                    # For the ridge penalty: inputs x with E[||x||^2] = input_norm_sq in
                    # uniformly random directions change the outputs of W by (B @ A) @ x,
                    # with E[||(B @ A) @ x||^2] = ||B @ A||_F^2 * input_norm_sq / in_features.
                    # Dividing by out_features gives the mean over output components,
                    # matching the mean in the preserve term of ara_loss.
                    ridge_scale = 0.0
                    if self.settings.ridge_weight > 0:
                        input_norm_sq = (
                            torch.cat([good_input, bad_input]).pow(2).sum(dim=1).mean()
                        )
                        ridge_scale = (
                            self.settings.ridge_weight
                            * input_norm_sq.item()
                            / (W_base.shape[0] * W_base.shape[1])
                        )

                    def objective(A: Tensor, B: Tensor) -> Tensor:
                        # input @ (W_base + B @ A).T
                        #   = input @ W_base.T + (input @ A.T) @ B.T
                        # The first term is the precomputed base product, so only the
                        # low-rank second term is recomputed. This avoids materializing
                        # the dense effective weight and the full input @ W_eff.T product
                        # on every closure evaluation.
                        new_good_output = good_product + (good_input @ A.T) @ B.T
                        new_bad_output = bad_product + (bad_input @ A.T) @ B.T

                        if self.settings.preserve_row_magnitudes:
                            # Row normalization rescales each output row by
                            # (original row norm / effective row norm), which applies
                            # directly to the outputs as a per-column factor.
                            eff_row_norms = LA.vector_norm(W_base + (B @ A), dim=1)
                            scale = W_row_norms.squeeze(1) / eff_row_norms
                            new_good_output = new_good_output * scale
                            new_bad_output = new_bad_output * scale

                        if bias is not None:
                            new_good_output = new_good_output + bias
                            new_bad_output = new_bad_output + bias

                        loss = ara_loss(
                            good_output,
                            bad_output,
                            new_good_output,
                            new_bad_output,
                            parameters,
                        )

                        if ridge_scale > 0:
                            loss = loss + ridge_scale * lowrank_frobenius_sq(A, B)

                        return loss

                    optimizer = LBFGS(
                        [lora_A, lora_B],
                        lr=self.settings.learning_rate,
                        max_iter=self.settings.max_iter,
                        history_size=self.settings.history_size,
                        line_search_fn=None
                        if self.settings.line_search == "none"
                        else self.settings.line_search,
                    )

                    def closure() -> Tensor:
                        optimizer.zero_grad()
                        loss = objective(lora_A, lora_B)
                        loss.backward()
                        return loss

                    for step in range(self.settings.n_optimization_steps):
                        loss = optimizer.step(closure)
                        if self.settings.print_loss:
                            print(
                                f"\\[{layer_index}/{component}/{module_index}] Step: {step + 1}, Loss: {loss.item():.6f}"
                            )

                    # Without a line search, the optimizer can diverge to adapters that are
                    # non-finite or so large that the loss is. Such a module is left
                    # unmodified instead of breaking the model.
                    with torch.no_grad():
                        diverged = not (
                            torch.isfinite(lora_A).all()
                            and torch.isfinite(lora_B).all()
                            and torch.isfinite(objective(lora_A, lora_B))
                        )
                    if diverged:
                        with torch.no_grad():
                            lora_A.zero_()
                            lora_B.zero_()
                        print(
                            f"[yellow]ARA diverged for {component} in layer {layer_index}; "
                            "that module is left unmodified.[/]"
                        )

                    # Free the gradient buffers accumulated during optimization.
                    # Without this, they persist on the model (one full-size gradient
                    # per processed weight) and can easily consume tens of GB of VRAM,
                    # causing out-of-memory errors during the subsequent evaluation.
                    optimizer.zero_grad(set_to_none=True)

    def reset_model(self, ctx: Context) -> None:
        model = ctx.get_model()
        fast_path = model.reset_model()
        if not fast_path:
            model.apply_lora(self.settings.lora_rank)
