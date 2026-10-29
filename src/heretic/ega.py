# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Expert-granular abliteration (EGA) for fused routed experts (fork addition, v3).
#
# Plain abliteration applies one refusal direction per layer (computed from the
# residual stream) to all experts of a layer alike. EGA gives every routed expert
# its own direction, computed from what that expert itself writes into the residual
# stream on the tokens that are routed to it, and scales the strength per expert
# by how often harmful prompts use the expert. Experts that harmful prompts do not
# use are left untouched. See README-MOE.md for the sources of the idea.

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import torch
import torch.linalg as LA
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .model import FusedExperts, Model
from .utils import Prompt, batchify, print

__all__ = [
    "EgaPlan",
    "ExpertStatistics",
    "FrequencyScaling",
    "ablate_fused_experts_per_expert",
    "build_ega_plans",
    "build_plan",
    "collect_expert_statistics",
    "expert_forward",
    "supports_ega",
]


class FrequencyScaling(str, Enum):
    # The strength of an expert is its routing frequency on harmful prompts
    # divided by the highest frequency of any expert in the layer.
    MAX = "max"
    # The frequency divided by the mean frequency of the used experts, capped at 1.
    MEAN = "mean"
    # Every expert that harmful prompts use gets the full strength.
    NONE = "none"


@dataclass
class ExpertStatistics:
    """
    Accumulated outputs of the experts of one fused tensor on a set of prompts,
    summed over the routed tokens only.
    """

    # (expert, hidden): sum of the expert's outputs (before the routing weight).
    output_sum: Tensor
    # (expert,): number of tokens that were routed to the expert.
    count: Tensor

    @classmethod
    def empty(cls, num_experts: int, hidden_size: int) -> "ExpertStatistics":
        return cls(
            output_sum=torch.zeros(num_experts, hidden_size, dtype=torch.float64),
            count=torch.zeros(num_experts, dtype=torch.int64),
        )


@dataclass
class EgaPlan:
    """
    Per-expert ablation plan of one fused tensor. It depends only on the original
    model and the prompts, not on the trial, so it is computed once.
    """

    # (expert, hidden): unit refusal direction of each expert. Zero where unused.
    directions: Tensor
    # (expert,): strength factor in [0, 1]. 0 means the expert is not modified at all.
    scales: Tensor
    # (expert,): True where the expert has too little data for its own direction
    # and uses the layer's residual direction instead.
    fallback: Tensor
    # (expert,): number of routed harmful tokens, for reporting.
    harmful_counts: Tensor

    def used_experts(self) -> list[int]:
        return [int(i) for i in torch.nonzero(self.scales > 0).flatten().tolist()]


def supports_ega(fused: FusedExperts) -> bool:
    """
    Whether the expert outputs of the fused tensor can be recomputed generically.
    That is the layout of Qwen3.5 MoE (and many other Transformers v5 models):
    `gate_up_proj` (expert, 2 * intermediate, hidden), `down_proj` (expert, hidden,
    intermediate) and an `act_fn`, applied as
    `linear(act(gate) * up, down_proj[e])` with `gate, up = linear(x, gate_up_proj[e]).chunk(2)`.
    """
    module = fused.module
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)

    if (
        fused.parameter_name != "down_proj"
        or fused.row_axis != 1
        or not isinstance(gate_up, Tensor)
        or not isinstance(down, Tensor)
        or not callable(getattr(module, "act_fn", None))
        or gate_up.dim() != 3
        or down.dim() != 3
    ):
        return False

    experts, rows, columns = down.shape
    return tuple(gate_up.shape) == (experts, 2 * columns, rows)


def expert_forward(module: Module, expert_index: int, x: Tensor) -> Tensor:
    """
    Output of one routed expert for the tokens `x` (token, hidden), before the
    routing weight is applied. Computed in float32.
    """
    gate_up = cast(Any, module).gate_up_proj[expert_index]
    down = cast(Any, module).down_proj[expert_index]
    act_fn = cast(Callable[[Tensor], Tensor], cast(Any, module).act_fn)
    gate_up = gate_up.to(torch.float32)
    down = down.to(torch.float32)

    gate, up = F.linear(x.to(torch.float32), gate_up).chunk(2, dim=-1)
    return F.linear(act_fn(gate) * up, down)


def select_last_positions(
    batch_size: int, sequence_length: int, last_positions: int, device: Any = None
) -> Tensor:
    """
    Flat token indices (for hidden states flattened to (prompt * position, ...))
    of the last `last_positions` positions of every prompt. Prompts are
    left-padded, so these positions always hold real tokens.
    """
    count = min(last_positions, sequence_length)
    return (
        torch.arange(batch_size, device=device)[:, None] * sequence_length
        + torch.arange(sequence_length - count, sequence_length, device=device)[None, :]
    ).reshape(-1)


def collect_expert_statistics(
    model: Model,
    prompts: list[Prompt],
    last_positions: int,
) -> dict[tuple[int, str], ExpertStatistics]:
    """
    Runs the prompts through the model and sums, for every expert of every
    supported fused tensor, the expert's own output over the tokens that are routed
    to it. Only the last `last_positions` positions of each prompt (where the model
    decides whether to refuse) are considered, like the residual directions
    use the last position.
    """
    statistics: dict[tuple[int, str], ExpertStatistics] = {}
    handles = []
    batch_shapes: dict[int, tuple[int, int]] = {}

    def get_layer_hook(layer_index: int):
        def hook(module: Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if isinstance(hidden_states, Tensor) and hidden_states.dim() >= 3:
                batch_shapes[layer_index] = (
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                )

        return hook

    def get_experts_hook(layer_index: int, fused: FusedExperts):
        key = fused.key
        num_experts = fused.num_experts
        hidden_size = fused.parameter.shape[fused.row_axis]
        statistics[key] = ExpertStatistics.empty(num_experts, hidden_size)

        def hook(module: Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            top_k_index = args[1] if len(args) > 1 else kwargs["top_k_index"]
            batch_shape = batch_shapes.get(layer_index)

            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            top_k_index = top_k_index.reshape(-1, top_k_index.shape[-1])

            if batch_shape is None or hidden_states.shape[0] != (
                batch_shape[0] * batch_shape[1]
            ):
                model._warn_once(
                    f"ega-shape:{key}",
                    f"[yellow]WARNING: The inputs of the experts [bold]{fused.path}[/] "
                    f"in layer {layer_index} do not have one row per token; "
                    "they are skipped by expert-granular abliteration.[/]",
                )
                return

            rows = select_last_positions(
                batch_shape[0],
                batch_shape[1],
                last_positions,
                device=hidden_states.device,
            )
            selected_states = hidden_states[rows]
            selected_index = top_k_index[rows]

            with torch.no_grad():
                for expert in torch.unique(selected_index).tolist():
                    # Newer Transformers versions use index num_experts for
                    # tokens that are not routed anywhere.
                    if expert < 0 or expert >= num_experts:
                        continue
                    mask = (selected_index == expert).any(dim=1)
                    outputs = expert_forward(module, expert, selected_states[mask])
                    statistics[key].output_sum[expert] += (
                        outputs.sum(dim=0).to(torch.float64).cpu()
                    )
                    statistics[key].count[expert] += int(mask.sum().item())

        return hook

    for layer_index, layer in enumerate(model.get_layers()):
        registered = False
        for fused_list in model.get_layer_fused_experts(layer_index).values():
            for fused in fused_list:
                if not supports_ega(fused):
                    model._warn_once(
                        f"ega-unsupported:{fused.path}",
                        f"[yellow]WARNING: The experts [bold]{fused.path}[/] have a layout "
                        "that expert-granular abliteration cannot recompute; "
                        "they are abliterated with the layer direction instead.[/]",
                    )
                    continue
                handles.append(
                    fused.module.register_forward_pre_hook(
                        get_experts_hook(layer_index, fused), with_kwargs=True
                    )
                )
                registered = True
        if registered:
            handles.append(
                layer.register_forward_pre_hook(
                    get_layer_hook(layer_index), with_kwargs=True
                )
            )

    try:
        for batch in batchify(prompts, model.settings.batch_size):
            model.generate(batch, max_new_tokens=1)
    finally:
        for handle in handles:
            handle.remove()

    return statistics


def build_plan(
    harmful: ExpertStatistics,
    harmless: ExpertStatistics,
    *,
    min_tokens: int,
    orthogonalize: bool,
    frequency_scaling: FrequencyScaling,
) -> EgaPlan:
    """
    Turns the statistics of one fused tensor into per-expert directions and strengths.

    * An expert with fewer than `min_tokens` routed harmful tokens is unused:
      strength 0, never modified.
    * A used expert with at least `min_tokens` routed harmless tokens gets its own
      direction: the normalized difference of its mean output on harmful and harmless
      tokens, optionally made orthogonal to its mean harmless output (like the
      projected abliteration of the layer direction).
    * A used expert without enough harmless tokens cannot form a difference and is
      marked `fallback`: the modifier then uses the layer's residual direction for it.
    * The strength factor is the expert's routing frequency on the harmful prompts,
      normalized according to `frequency_scaling`.
    """
    num_experts, hidden_size = harmful.output_sum.shape
    harmful_count = harmful.count
    harmless_count = harmless.count

    used = harmful_count >= min_tokens
    own = used & (harmless_count >= min_tokens)

    harmful_mean = harmful.output_sum / harmful_count.clamp(min=1).unsqueeze(1)
    harmless_mean = harmless.output_sum / harmless_count.clamp(min=1).unsqueeze(1)

    difference = harmful_mean - harmless_mean
    difference_norm = LA.vector_norm(difference, dim=1)
    # A vanishing difference carries no direction.
    own = own & (difference_norm > 1e-12)

    directions = torch.zeros(num_experts, hidden_size, dtype=torch.float64)
    directions[own] = difference[own] / difference_norm[own].unsqueeze(1)

    if orthogonalize and bool(own.any()):
        good_norm = LA.vector_norm(harmless_mean[own], dim=1, keepdim=True)
        good_direction = harmless_mean[own] / good_norm.clamp(min=1e-12)
        own_directions = directions[own]
        projection = (own_directions * good_direction).sum(dim=1, keepdim=True)
        orthogonal = own_directions - projection * good_direction
        orthogonal_norm = LA.vector_norm(orthogonal, dim=1, keepdim=True)
        # If nothing remains after removing the harmless component, keep the
        # plain direction instead of a zero vector.
        keep = orthogonal_norm.squeeze(1) > 1e-12
        own_directions[keep] = orthogonal[keep] / orthogonal_norm[keep]
        directions[own] = own_directions

    fallback = used & ~own

    frequencies = harmful_count.to(torch.float64) / max(
        int(harmful_count.sum().item()), 1
    )
    scales = torch.zeros(num_experts, dtype=torch.float64)

    if bool(used.any()):
        used_frequencies = frequencies[used]
        if frequency_scaling == FrequencyScaling.MAX:
            scales[used] = used_frequencies / used_frequencies.max()
        elif frequency_scaling == FrequencyScaling.MEAN:
            scales[used] = (used_frequencies / used_frequencies.mean()).clamp(max=1.0)
        else:
            scales[used] = 1.0

    return EgaPlan(
        directions=directions.to(torch.float32),
        scales=scales.to(torch.float32),
        fallback=fallback,
        harmful_counts=harmful_count.clone(),
    )


def build_ega_plans(
    model: Model,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    *,
    last_positions: int,
    min_tokens: int,
    orthogonalize: bool,
    frequency_scaling: FrequencyScaling,
) -> dict[tuple[int, str], EgaPlan]:
    """
    Computes the EGA plans of all supported fused tensors, keyed by
    FusedExperts.key. Must be called on the unmodified model.
    """
    print("* Obtaining per-expert outputs for good prompts...")
    good = collect_expert_statistics(model, good_prompts, last_positions)
    print("* Obtaining per-expert outputs for bad prompts...")
    bad = collect_expert_statistics(model, bad_prompts, last_positions)

    return {
        key: build_plan(
            bad[key],
            good[key],
            min_tokens=min_tokens,
            orthogonalize=orthogonalize,
            frequency_scaling=frequency_scaling,
        )
        for key in bad
    }


def ablate_fused_experts_per_expert(
    parameter: Tensor,
    row_axis: int,
    directions: Tensor,
    weights: Tensor,
    row_normalization: Any,
    max_chunk_elements: int = 2**26,
) -> None:
    """
    Like `ablate_fused_experts`, but with a separate direction and strength for
    every expert: W_e <- W_e - weights[e] * v_e (v_e^T W_e), in place.

    Experts with weight 0 are not touched at all (not even rewritten), so they stay
    bit-exactly equal to the original. `row_normalization` has the same meaning
    as there ("none", "pre", "full").
    """
    # RowNormalization is a str enum ("none", "pre", "full"); comparing the values
    # avoids importing modifiers.abliteration, which imports this module.
    mode = getattr(row_normalization, "value", row_normalization)

    assert parameter.dim() == 3 and row_axis in (1, 2)

    matrices = parameter.data if row_axis == 1 else parameter.data.transpose(1, 2)
    expert_count, row_count, column_count = matrices.shape
    assert directions.shape == (expert_count, row_count), (
        f"Directions of shape {tuple(directions.shape)} do not match "
        f"{expert_count} experts with hidden size {row_count}"
    )
    assert weights.shape == (expert_count,)

    selected = torch.nonzero(weights != 0).flatten()
    chunk_size = max(1, max_chunk_elements // (row_count * column_count))

    with torch.no_grad():
        for start in range(0, len(selected), chunk_size):
            index = selected[start : start + chunk_size].to(matrices.device)
            W = matrices[index].to(torch.float32)
            v = directions[index.cpu()].to(device=matrices.device, dtype=torch.float32)
            w = weights[index.cpu()].to(device=matrices.device, dtype=torch.float32)
            w = w.view(-1, 1, 1)

            if mode == "none":
                coefficients = torch.einsum("eh,ehc->ec", v, W)
                W -= w * torch.einsum("eh,ec->ehc", v, coefficients)
            else:
                W_row_norms = LA.vector_norm(W, dim=2, keepdim=True)
                W_normalized = F.normalize(W, p=2, dim=2)
                coefficients = torch.einsum("eh,ehc->ec", v, W_normalized)
                ablated = W_normalized - w * torch.einsum("eh,ec->ehc", v, coefficients)

                if mode == "pre":
                    W += W_row_norms * (ablated - W_normalized)
                else:
                    W = F.normalize(ablated, p=2, dim=2) * W_row_norms

            matrices[index] = W.to(matrices.dtype)
