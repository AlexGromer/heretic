# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Router pre-bias for MoE models (fork addition, upstream issue #322).
#
# Experts that the router prefers on harmful prompts are pushed down and experts it prefers
# on harmless prompts are pushed up, by adding b = -alpha * log(p_bad(e) / p_good(e)) to the
# router logits before the top-k selection, where p(e) is how often expert e is selected.
# The issue reports flipped refusals on Qwen3.5/3.6-35B-A3B without MMLU/GSM8K loss and a
# sharp optimum near alpha = 0.5 (self-reported, not reproduced).
#
# The issue adds the bias at inference time. To make it part of an exported checkpoint
# (and GGUF), it is stored in the weights here:
# - a router with a bias parameter (e.g. gpt-oss) gets b added to its bias (exact);
# - a router without one (e.g. Qwen3.5 MoE) gets the outer product b u^T / c added to its
#   weight, where u is the unit mean direction of the router inputs and c the mean
#   projection of the inputs onto u. For an input h, the logits change by b (u.h) / c,
#   i.e. by b on average and proportionally to how much of the mean direction the input
#   contains. This is an approximation; its spread is printed when the plan is built.

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .model import Model, ParameterHandle
from .utils import Prompt, batchify

ROUTER_ATTRIBUTES = ("gate", "router")
MOE_BLOCK_ATTRIBUTES = ("mlp", "block_sparse_moe", "moe", "feed_forward")

# Number of router inputs per layer kept to measure the spread of the weight fold.
SAMPLE_ROWS = 512


@dataclass
class Router:
    layer_index: int
    # Path of the router module relative to the decoder layer, e.g. "mlp.gate".
    path: str
    module: Module
    num_experts: int
    top_k: int

    @property
    def weight(self) -> Tensor:
        return cast(Any, self.module).weight

    @property
    def bias(self) -> Tensor | None:
        bias = getattr(self.module, "bias", None)
        return bias if isinstance(bias, Tensor) else None


@dataclass
class RouterBiasPlan:
    """Per-layer data of the router pre-bias, computed once on the unmodified model."""

    router: Router
    # (expert,): log(p_bad / p_good), clipped.
    log_ratio: Tensor
    # Unit mean direction of the router inputs and the mean projection onto it.
    # Unused for routers with a bias parameter.
    direction: Tensor
    mean_projection: float
    # Relative spread (std / mean) of the projection over sampled inputs.
    projection_spread: float


def find_router(model: Model, layer_index: int) -> Router | None:
    """
    The router of a MoE layer: a module named gate/router in the layer's MoE block whose
    `weight` has shape (experts, hidden size), or None for dense layers.
    """
    layer = model.get_layers()[layer_index]
    config = model.model.config
    text_config = getattr(config, "text_config", None) or config
    hidden_size = getattr(text_config, "hidden_size", None)
    top_k = getattr(text_config, "num_experts_per_tok", None)

    for block_name in MOE_BLOCK_ATTRIBUTES:
        block = getattr(layer, block_name, None)
        if not isinstance(block, Module):
            continue
        for router_name in ROUTER_ATTRIBUTES:
            module = getattr(block, router_name, None)
            weight = getattr(module, "weight", None)
            if not isinstance(module, Module) or not isinstance(weight, Tensor):
                continue
            if weight.dim() != 2 or weight.shape[1] != hidden_size:
                continue
            if not weight.is_floating_point():
                # Quantized (e.g. bitsandbytes 4-bit) routers cannot be edited directly.
                return None
            k = getattr(module, "top_k", None) or top_k
            if not isinstance(k, int) or not 1 <= k < weight.shape[0]:
                continue
            return Router(
                layer_index=layer_index,
                path=f"{block_name}.{router_name}",
                module=module,
                num_experts=weight.shape[0],
                top_k=k,
            )

    return None


def log_ratio(
    bad_counts: Tensor,
    bad_tokens: int,
    good_counts: Tensor,
    good_tokens: int,
    max_abs: float,
) -> Tensor:
    """
    log(p_bad / p_good) per expert, from selection counts over the given numbers of
    tokens, with add-one-half smoothing (so that experts never selected by one side get a
    finite value) and clipped to [-max_abs, max_abs].
    """
    p_bad = (bad_counts.to(torch.float64) + 0.5) / (bad_tokens + 1)
    p_good = (good_counts.to(torch.float64) + 0.5) / (good_tokens + 1)
    return torch.log(p_bad / p_good).clamp(-max_abs, max_abs).to(torch.float32)


@dataclass
class _RoutingStatistics:
    counts: Tensor
    tokens: int
    input_sum: Tensor
    input_count: int
    samples: list[Tensor]


def collect_routing_statistics(
    model: Model,
    routers: list[Router],
    prompts: list[Prompt],
    last_positions: int,
) -> dict[int, _RoutingStatistics]:
    """
    Counts how often each expert is selected at the last `last_positions` positions of
    each prompt, and accumulates the router inputs there.
    """
    statistics = {
        router.layer_index: _RoutingStatistics(
            counts=torch.zeros(router.num_experts, dtype=torch.int64),
            tokens=0,
            input_sum=torch.zeros(router.weight.shape[1], dtype=torch.float64),
            input_count=0,
            samples=[],
        )
        for router in routers
    }
    batch_shapes: dict[int, tuple[int, int]] = {}
    handles = []

    def get_layer_hook(layer_index: int):
        def hook(module: Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if isinstance(hidden_states, Tensor) and hidden_states.dim() >= 3:
                batch_shapes[layer_index] = (
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                )

        return hook

    def get_router_hook(router: Router):
        def hook(module: Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            batch_shape = batch_shapes.get(router.layer_index)
            if batch_shape is None or hidden_states.shape[0] != (
                batch_shape[0] * batch_shape[1]
            ):
                model._warn_once(
                    f"router-shape:{router.layer_index}",
                    f"[yellow]WARNING: The router inputs in layer {router.layer_index} "
                    "do not have one row per token; that layer gets no router bias.[/]",
                )
                return

            count = min(last_positions, batch_shape[1])
            rows = (
                torch.arange(batch_shape[0], device=hidden_states.device)[:, None]
                * batch_shape[1]
                + torch.arange(
                    batch_shape[1] - count,
                    batch_shape[1],
                    device=hidden_states.device,
                )[None, :]
            ).reshape(-1)
            selected = hidden_states[rows].to(torch.float32)

            with torch.no_grad():
                bias = router.bias
                logits = F.linear(
                    selected,
                    router.weight.to(torch.float32),
                    None if bias is None else bias.to(torch.float32),
                )
                chosen = logits.topk(router.top_k, dim=-1).indices
                entry = statistics[router.layer_index]
                entry.counts += torch.bincount(
                    chosen.reshape(-1).cpu(), minlength=router.num_experts
                )
                entry.tokens += selected.shape[0]
                entry.input_sum += selected.sum(dim=0).to(torch.float64).cpu()
                entry.input_count += selected.shape[0]
                kept = sum(sample.shape[0] for sample in entry.samples)
                if kept < SAMPLE_ROWS:
                    entry.samples.append(selected[: SAMPLE_ROWS - kept].cpu())

        return hook

    layers = model.get_layers()
    for router in routers:
        handles.append(
            layers[router.layer_index].register_forward_pre_hook(
                get_layer_hook(router.layer_index), with_kwargs=True
            )
        )
        handles.append(
            router.module.register_forward_pre_hook(
                get_router_hook(router), with_kwargs=True
            )
        )

    try:
        for batch in batchify(prompts, model.settings.batch_size):
            model.generate(batch, max_new_tokens=1)
    finally:
        for handle in handles:
            handle.remove()

    return statistics


def build_router_bias_plans(
    model: Model,
    good_prompts: list[Prompt],
    bad_prompts: list[Prompt],
    last_positions: int,
    max_log_ratio: float,
) -> list[RouterBiasPlan]:
    routers = [
        router
        for layer_index in range(len(model.get_layers()))
        if (router := find_router(model, layer_index)) is not None
    ]
    if not routers:
        return []

    good = collect_routing_statistics(model, routers, good_prompts, last_positions)
    bad = collect_routing_statistics(model, routers, bad_prompts, last_positions)

    plans = []
    for router in routers:
        g = good[router.layer_index]
        b = bad[router.layer_index]
        if g.tokens == 0 or b.tokens == 0:
            continue

        mean_input = (g.input_sum + b.input_sum) / (g.input_count + b.input_count)
        mean_projection = float(mean_input.norm())
        direction = (mean_input / max(mean_projection, 1e-30)).to(torch.float32)

        samples = torch.cat(g.samples + b.samples)
        projections = samples @ direction
        spread = (
            float(projections.std() / projections.mean())
            if projections.numel() > 1 and projections.mean() != 0
            else float("inf")
        )

        plans.append(
            RouterBiasPlan(
                router=router,
                log_ratio=log_ratio(
                    b.counts, b.tokens, g.counts, g.tokens, max_log_ratio
                ),
                direction=direction,
                mean_projection=mean_projection,
                projection_spread=spread,
            )
        )

    return plans


def apply_router_bias(model: Model, plan: RouterBiasPlan, alpha: float) -> None:
    """Adds the pre-bias -alpha * log_ratio to the router, in place (see module docstring)."""
    if alpha == 0:
        return

    router = plan.router
    bias = -alpha * plan.log_ratio

    if router.bias is not None:
        handle = ParameterHandle(
            router.layer_index, f"{router.path}.bias", router.module, "bias"
        )
        model.begin_fused_edit(handle)
        with torch.no_grad():
            parameter = handle.parameter
            parameter += bias.to(device=parameter.device, dtype=parameter.dtype)
        return

    if plan.mean_projection <= 1e-6:
        return

    handle = ParameterHandle(
        router.layer_index, f"{router.path}.weight", router.module, "weight"
    )
    model.begin_fused_edit(handle)
    with torch.no_grad():
        parameter = handle.parameter
        delta = torch.outer(bias, plan.direction) / plan.mean_projection
        updated = parameter.to(torch.float32) + delta.to(parameter.device)
        parameter.copy_(updated.to(parameter.dtype))
