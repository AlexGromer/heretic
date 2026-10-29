# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, cast

import bitsandbytes.functional as BNB_F
import torch
import torch.linalg as LA
import torch.nn.functional as F
from optuna import Trial
from peft.tuners.lora.layer import Linear
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
)
from torch import Tensor

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.model import FUSED_EXPERTS_WEIGHT_COMPONENT
from heretic.modifier import Context, Modifier, Serializable
from heretic.utils import format_dataset_specification, print


@dataclass
class WeightDistribution:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


@dataclass
class Parameters(Serializable):
    direction_index: float | None
    weight_distributions: dict[str, WeightDistribution]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_presentation_dict(self) -> dict[str, str]:
        parameters = {}

        parameters["direction_index"] = (
            "per layer"
            if (self.direction_index is None)
            else f"{self.direction_index:.2f}"
        )

        for component, weight_distribution in self.weight_distributions.items():
            for name, value in asdict(weight_distribution).items():
                parameters[f"{component}.{name}"] = f"{value:.2f}"

        return parameters

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        return Parameters(
            direction_index=data["direction_index"],
            weight_distributions={
                component: WeightDistribution(**weight_distribution)
                for component, weight_distribution in data[
                    "weight_distributions"
                ].items()
            },
        )


class RowNormalization(str, Enum):
    NONE = "none"
    PRE = "pre"
    # POST = "post"  # Theoretically possible, but provides no advantage.
    FULL = "full"


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

    orthogonalize_direction: bool = Field(
        default=True,
        description=(
            "Whether to adjust the residual directions so that only the component that is "
            "orthogonal to the good direction is subtracted during abliteration."
        ),
    )

    row_normalization: RowNormalization = Field(
        default=RowNormalization.FULL,
        description=(
            "How to apply row normalization of the weights. Options: "
            '"none" (no normalization), '
            '"pre" (compute LoRA adapter relative to row-normalized weights), '
            '"full" (like "pre", but renormalizes to preserve original row magnitudes).'
        ),
    )

    full_normalization_lora_rank: PositiveInt = Field(
        default=3,
        description=(
            'The rank of the LoRA adapter to use when "full" row normalization is used. '
            "Row magnitude preservation is approximate due to non-linear effects, "
            "and this determines the rank of that approximation. Higher ranks produce "
            "larger output files and may slow down evaluation."
        ),
    )

    winsorization_quantile: float = Field(
        default=1.0,
        description=(
            "The symmetric winsorization to apply to the per-prompt, per-layer residual vectors, "
            "expressed as the quantile to clamp to (between 0 and 1). Disabled by default. "
            'This can tame so-called "massive activations" that occur in some models. '
            "Example: winsorization_quantile = 0.95 computes the 0.95-quantile of the absolute values "
            "of the components, then clamps the magnitudes of all components to that quantile."
        ),
    )


# The maximum number of elements of the temporary float32 tensors that are created
# per chunk when modifying fused experts. A chunk of 2**26 elements takes 256 MiB
# in float32, and a handful of those are alive at the same time.
FUSED_CHUNK_ELEMENTS = 2**26


def ablate_fused_experts(
    parameter: Tensor,
    row_axis: int,
    v: Tensor,
    weight: float,
    row_normalization: RowNormalization,
    max_chunk_elements: int = FUSED_CHUNK_ELEMENTS,
) -> None:
    """
    Orthogonally ablates the direction v from the down projections of all experts
    held in a fused 3D parameter, in place.

    This applies to each expert's weight matrix W (with the hidden dimension as its rows,
    exactly like the weight of a Linear layer) the same ablation as the LoRA path
    does for Linear modules: W <- W - weight * v (v^T W).
    The row normalization modes have the same semantics as there: "pre" applies
    the ablation to the row-normalized matrix and scales the change back by the row norms,
    "full" additionally renormalizes the rows to their original norms. Because the weights
    are modified directly, "full" is exact here, instead of being approximated by a
    low-rank adapter.

    The experts are processed in chunks, so that only a small part of the parameter
    is converted to float32 at any time, and the result is written back
    to the original storage, so no second copy of the parameter is ever created.
    """
    assert parameter.dim() == 3 and row_axis in (1, 2)

    # Put the hidden dimension on axis 1, for (expert, row, column) matrices.
    # This is a view, so writing to it modifies the parameter.
    matrices = parameter.data if row_axis == 1 else parameter.data.transpose(1, 2)

    expert_count, row_count, column_count = matrices.shape
    assert v.shape == (row_count,), (
        f"Direction of shape {tuple(v.shape)} does not match "
        f"the hidden size {row_count} of the fused experts"
    )

    v = v.to(device=parameter.device, dtype=torch.float32)

    chunk_size = max(1, max_chunk_elements // (row_count * column_count))

    with torch.no_grad():
        for start in range(0, expert_count, chunk_size):
            chunk = matrices[start : start + chunk_size]
            W = chunk.to(torch.float32)

            if row_normalization == RowNormalization.NONE:
                # coefficients[e] = v^T W[e], with shape (column,).
                coefficients = torch.einsum("h,ehc->ec", v, W)
                W -= weight * torch.einsum("h,ec->ehc", v, coefficients)
            else:
                W_row_norms = LA.vector_norm(W, dim=2, keepdim=True)
                W_normalized = F.normalize(W, p=2, dim=2)
                coefficients = torch.einsum("h,ehc->ec", v, W_normalized)
                ablated = W_normalized - weight * torch.einsum(
                    "h,ec->ehc", v, coefficients
                )

                if row_normalization == RowNormalization.PRE:
                    # Same as scaling the change by the row norms: W + norms * (ablated - W_normalized).
                    W += W_row_norms * (ablated - W_normalized)
                else:
                    # Normalize the adjusted matrix along the rows, then restore
                    # the original row norms.
                    W = F.normalize(ablated, p=2, dim=2) * W_row_norms

            chunk.copy_(W.to(chunk.dtype))


class Abliteration(Modifier[Parameters]):
    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def modifier_name(self) -> str:
        if (
            self.settings.orthogonalize_direction
            and self.settings.row_normalization == RowNormalization.FULL
        ):
            return "Magnitude-Preserving Orthogonal Ablation (MPOA)"
        elif self.settings.orthogonalize_direction:
            return "Projected Abliteration"
        else:
            return "Abliteration"

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

        print()
        print("Calculating per-layer residual directions...")

        print("* Obtaining residual mean for good prompts...")
        good_means = model.get_residuals_mean(
            good_prompts,
            winsorization_quantile=self.settings.winsorization_quantile,
        )
        print("* Obtaining residual mean for bad prompts...")
        bad_means = model.get_residuals_mean(
            bad_prompts,
            winsorization_quantile=self.settings.winsorization_quantile,
        )

        self.residual_directions = F.normalize(
            bad_means - good_means,
            p=2,
            dim=1,
        )

        if self.settings.orthogonalize_direction:
            # Implements https://huggingface.co/blog/grimjim/projected-abliteration
            # Adjust the residual directions so that only the component that is
            # orthogonal to the good direction is subtracted during abliteration.
            good_directions = F.normalize(
                good_means,
                p=2,
                dim=1,
            )
            projection_vector = torch.sum(
                self.residual_directions * good_directions,
                dim=1,
            )
            self.residual_directions = (
                self.residual_directions
                - projection_vector.unsqueeze(1) * good_directions
            )
            self.residual_directions = F.normalize(
                self.residual_directions,
                p=2,
                dim=1,
            )

        if self.settings.row_normalization != RowNormalization.FULL:
            # Rank 1 is sufficient for directional ablation without renormalization.
            self.lora_rank = 1
        else:
            # Row magnitude preservation introduces nonlinear effects.
            self.lora_rank = self.settings.full_normalization_lora_rank

        if model.has_fused_experts():
            print()
            print(
                "* Fused routed experts found. They are ablated by modifying their weights "
                "directly (using the parameters of [bold]mlp.down_proj[/]), "
                "which requires merging when exporting."
            )

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.
        model.apply_lora(self.lora_rank)

    def suggest_parameters(self, ctx: Context, trial: Trial) -> Parameters:
        model = ctx.get_model()

        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        last_layer_index = len(model.get_layers()) - 1

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * last_layer_index,
            0.9 * last_layer_index,
        )

        if direction_scope == "per layer":
            direction_index = None

        weight_distributions = {}

        for component in model.get_abliterable_components():
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
            #
            # The MLP gets a negative lower bound that is then clamped to 0, so the
            # optimizer can fully disable its ablation. The clamp puts a positive
            # probability mass on exactly 0 (the continuous sampler would otherwise
            # reach 0 with probability zero). Ablating the MLP is often unnecessary for
            # removing refusals and tends to damage model intelligence more than
            # ablating the attention output, so on many models the optimum is to leave
            # it (mostly) untouched. See issue #202.
            max_weight_lower_bound = -0.25 if component == "mlp.down_proj" else 0.8
            max_weight = max(
                0.0,
                trial.suggest_float(
                    f"{component}.max_weight",
                    max_weight_lower_bound,
                    1.5,
                ),
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            # For sampling purposes, min_weight is expressed as a fraction of max_weight,
            # again because multivariate TPE doesn't support variable-range parameters.
            # The value is transformed into the actual min_weight value below.
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                1.0,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                max(0.6 * last_layer_index, 1.0),
            )

            weight_distributions[component] = WeightDistribution(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        return Parameters(
            direction_index=direction_index,
            weight_distributions=weight_distributions,
        )

    def modify_model(self, ctx: Context, parameters: Parameters) -> None:
        model = ctx.get_model()

        if parameters.direction_index is None:
            residual_direction = None
        else:
            # The index must be shifted by 1 because the first element
            # of residual_directions is the direction for the embeddings.
            weight, index = math.modf(parameters.direction_index + 1)
            residual_direction = F.normalize(
                self.residual_directions[int(index)].lerp(
                    self.residual_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        def get_layer_weight(layer_index: int, component: str) -> float | None:
            # Returns the weight of the ablation of the component in the layer,
            # or None if the component is not to be ablated in that layer.
            weight_distribution = parameters.weight_distributions[component]

            # Type inference fails here for some reason.
            distance = abs(layer_index - weight_distribution.max_weight_position)

            # Don't orthogonalize layers that are more than
            # min_weight_distance away from max_weight_position.
            if distance > weight_distribution.min_weight_distance:
                return None

            # Interpolate linearly between max_weight and min_weight
            # over min_weight_distance.
            weight = weight_distribution.max_weight + (
                distance / weight_distribution.min_weight_distance
            ) * (weight_distribution.min_weight - weight_distribution.max_weight)

            # A weight of 0 disables this component's ablation. reset_model() has
            # already left the adapter at identity, so abort before the otherwise
            # wasteful decomposition (which would also be operating on a zero matrix).
            if weight == 0:
                return None

            return weight

        def get_layer_residual_direction(layer_index: int) -> Tensor:
            if residual_direction is None:
                # The index must be shifted by 1 because the first element
                # of residual_directions is the direction for the embeddings.
                return self.residual_directions[layer_index + 1]
            else:
                return residual_direction

        # Note that some implementations of abliteration also orthogonalize
        # the embedding matrix, but it's unclear if that has any benefits.
        for layer_index in range(len(model.get_layers())):
            # Fused routed experts (Transformers v5) are no LoRA targets, so they
            # are ablated by modifying their weights directly. They use the parameters
            # of the "mlp.down_proj" component, so the search space is the same
            # as for models with separate experts.
            for fused_list in model.get_layer_fused_experts(layer_index).values():
                weight = get_layer_weight(layer_index, FUSED_EXPERTS_WEIGHT_COMPONENT)

                if weight is None:
                    continue

                for fused in fused_list:
                    # Keeps the original weights, so reset_model can restore them.
                    model.begin_fused_edit(fused)
                    ablate_fused_experts(
                        fused.parameter,
                        fused.row_axis,
                        get_layer_residual_direction(layer_index),
                        weight,
                        self.settings.row_normalization,
                    )

            for component, modules in model.get_layer_modules(layer_index).items():
                weight = get_layer_weight(layer_index, component)

                if weight is None:
                    continue

                layer_residual_direction = get_layer_residual_direction(layer_index)

                for module in modules:
                    # FIXME: This cast is potentially invalid, because the program logic
                    #        does not guarantee that the module is of type Linear, and in fact
                    #        the retrieved modules might not conform to the interface assumed
                    #        below (though they do in practice). However, this is difficult
                    #        to fix cleanly, because get_layer_modules is called twice on
                    #        different model configurations, and PEFT employs different
                    #        module types depending on the chosen quantization.
                    module = cast(Linear, module)

                    # LoRA abliteration: delta W = -lambda * v * (v^T W)
                    # lora_B = -lambda * v
                    # lora_A = v^T W

                    # Use the FP32 residual direction directly (no downcast/upcast)
                    # and move to the correct device.
                    v = layer_residual_direction.to(module.weight.device)

                    # Get W (dequantize if necessary).
                    #
                    # FIXME: This cast is valid only under the assumption that the original
                    #        module wrapped by the LoRA adapter has a weight attribute.
                    #        See the comment above for why this is currently not guaranteed.
                    base_weight = cast(Tensor, module.base_layer.weight)
                    quant_state = getattr(base_weight, "quant_state", None)

                    if quant_state is None:
                        W = base_weight.to(torch.float32)
                    else:
                        # 4-bit quantization.
                        W = BNB_F.dequantize_4bit(
                            base_weight.data,
                            quant_state,
                        ).to(torch.float32)

                    # Flatten weight matrix to (out_features, in_features).
                    W = W.view(W.shape[0], -1)

                    if self.settings.row_normalization == RowNormalization.FULL:
                        # Keep a reference to the original weight matrix so we can subtract it later.
                        W_org = W

                    if self.settings.row_normalization != RowNormalization.NONE:
                        # Get the row norms.
                        W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
                        # Normalize the weight matrix along the rows.
                        W = F.normalize(W, p=2, dim=1)

                    # Calculate lora_A = v^T W
                    # v is (d_out,), W is (d_out, d_in)
                    # v @ W -> (d_in,)
                    lora_A = (v @ W).view(1, -1)

                    # Calculate lora_B = -weight * v
                    # v is (d_out,)
                    lora_B = (-weight * v).view(-1, 1)

                    if self.settings.row_normalization == RowNormalization.PRE:
                        # Make the LoRA adapter apply to the original weight matrix.
                        lora_B = W_row_norms * lora_B
                    elif self.settings.row_normalization == RowNormalization.FULL:
                        # Approximates https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
                        W = W + lora_B @ lora_A
                        # Normalize the adjusted weight matrix along the rows.
                        W = F.normalize(W, p=2, dim=1)
                        # Restore the original row norms of the weight matrix.
                        W = W * W_row_norms
                        # Subtract the original matrix to turn W into a delta.
                        W = W - W_org
                        # Use a low-rank SVD to get an approximation of the matrix.
                        r = model.peft_config.r

                        # svd_lowrank is randomized:
                        # https://github.com/pytorch/pytorch/blob/20919052303c0b5ba87f8bf7e19237dc33ab09d3/torch/_lowrank.py#L108-L109
                        # Reseed immediately before the call so restoring a trial is independent of RNG history.
                        torch.manual_seed(self.heretic_settings.seed)
                        # "It's safe to call this function if CUDA is not available;
                        # in that case, it is silently ignored."
                        torch.cuda.manual_seed_all(self.heretic_settings.seed)  # ty:ignore[invalid-argument-type]
                        U, S, Vh = torch.svd_lowrank(W, q=2 * r + 4, niter=6)

                        # Truncate it to the part we want to store in the LoRA adapter.
                        # Note: svd_lowrank actually returns V, so transpose it to get Vh.
                        U = U[:, :r]
                        S = S[:r]
                        Vh = Vh[:, :r].T
                        # Transfer it into the LoRA adapter components. Split the singular values
                        # evenly between the two components to keep their norms balanced and avoid
                        # potential issues with numerical stability.
                        sqrt_S = torch.sqrt(S)
                        lora_B = U @ torch.diag(sqrt_S)
                        lora_A = torch.diag(sqrt_S) @ Vh

                    # Assign to adapters. The adapter name is "default", because that's
                    # what PEFT uses when no name is explicitly specified, as above.
                    # These casts are therefore valid.
                    weight_A = cast(Tensor, module.lora_A["default"].weight)
                    weight_B = cast(Tensor, module.lora_B["default"].weight)
                    weight_A.data = lora_A.to(weight_A.dtype)
                    weight_B.data = lora_B.to(weight_B.dtype)

    def reset_model(self, ctx: Context) -> None:
        model = ctx.get_model()
        fast_path = model.reset_model()
        if not fast_path:
            model.apply_lora(self.lora_rank)
