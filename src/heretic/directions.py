# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Multi-directional refusal suppression with self-organizing maps (fork addition, upstream
# issue #140). Piras et al. 2025, "SOM Directions are Better than One"
# (https://arxiv.org/abs/2511.08379): the residuals of harmful prompts form clusters, so
# one mean-difference direction covers them only on average. A self-organizing map fitted
# to the harmful residuals of a layer finds cluster prototypes; the differences between the
# most used prototypes and the harmless mean are additional refusal directions.
#
# Unlike upstream PR #196 (closed), this needs no extra dependency (the SOM is a batch SOM
# in PyTorch), keeps the classic mean-difference direction as the first direction (so that
# the search space contains plain abliteration), and makes the directions of a layer
# orthonormal, so that ablating them together is a projection onto their complement.

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def train_som(
    data: Tensor,
    grid: tuple[int, int],
    epochs: int,
    sigma_end: float,
    seed: int = 0,
) -> Tensor:
    """
    Fits a batch self-organizing map with a rectangular grid to `data` (sample, dim) and
    returns the prototypes (grid[0] * grid[1], dim). The neighborhood radius decays
    exponentially from half the grid size to `sigma_end` over the epochs. Deterministic
    for a given seed (the prototypes are initialized with random samples).
    """
    data = data.to(torch.float64)
    sample_count = data.shape[0]
    neuron_count = grid[0] * grid[1]

    generator = torch.Generator().manual_seed(seed)
    initial = torch.randint(0, sample_count, (neuron_count,), generator=generator)
    prototypes = data[initial].clone()

    coordinates = torch.cartesian_prod(
        torch.arange(grid[0], dtype=torch.float64),
        torch.arange(grid[1], dtype=torch.float64),
    ).reshape(neuron_count, 2)
    grid_distances_sq = torch.cdist(coordinates, coordinates) ** 2

    sigma_start = max(max(grid) / 2, sigma_end)
    for epoch in range(epochs):
        fraction = epoch / max(epochs - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** fraction

        winners = torch.cdist(data, prototypes).argmin(dim=1)
        # influence[j, i]: neighborhood weight of neuron j for the winner of sample i.
        influence = torch.exp(-grid_distances_sq[:, winners] / (2 * sigma**2))
        weight_sums = influence.sum(dim=1, keepdim=True)
        updated = (influence @ data) / weight_sums.clamp_min(1e-12)
        # Neurons that no sample influences keep their prototype.
        prototypes = torch.where(weight_sums > 1e-12, updated, prototypes)

    return prototypes.to(torch.float32)


def neuron_hits(prototypes: Tensor, data: Tensor) -> Tensor:
    """Number of samples whose nearest prototype is each neuron."""
    winners = torch.cdist(data.to(torch.float32), prototypes).argmin(dim=1)
    return torch.bincount(winners, minlength=prototypes.shape[0])


def orthonormal_extension(
    basis: Tensor,
    candidates: Tensor,
    count: int,
    min_relative_norm: float = 0.05,
) -> Tensor:
    """
    Extends the orthonormal rows of `basis` by up to `count` directions from `candidates`
    (in order) using Gram-Schmidt. A candidate whose component orthogonal to the
    directions found so far is shorter than `min_relative_norm` of its length adds nothing
    new and is skipped. Returns (count, dim); rows that could not be filled are zero.
    """
    directions: list[Tensor] = []
    current = basis.to(torch.float64)

    for candidate in candidates.to(torch.float64):
        if len(directions) == count:
            break
        norm = candidate.norm()
        if norm == 0:
            continue
        residual = candidate - current.T @ (current @ candidate)
        if residual.norm() < min_relative_norm * norm:
            continue
        direction = residual / residual.norm()
        directions.append(direction)
        current = torch.cat([current, direction[None]])

    result = torch.zeros(count, basis.shape[1], dtype=torch.float64)
    if directions:
        result[: len(directions)] = torch.stack(directions)
    return result.to(torch.float32)


def compute_extra_directions(
    bad_residuals: Tensor,
    good_means: Tensor,
    main_directions: Tensor,
    count: int,
    grid: tuple[int, int],
    epochs: int,
    sigma_end: float,
    orthogonalize_good: bool,
    seed: int = 0,
) -> Tensor:
    """
    Additional refusal directions per residual row (embeddings and layers).

    - bad_residuals: (prompt, row, hidden) residuals of the harmful prompts.
    - good_means: (row, hidden) mean residuals of the harmless prompts.
    - main_directions: (row, hidden) unit mean-difference directions (already orthogonalized
      against the good direction if that setting is on).

    For each row, a SOM is fitted to the harmful residuals, and the differences between its
    prototypes (most hits first) and the harmless mean are made orthogonal to the good
    direction (if `orthogonalize_good`), to the main direction and to each other.
    Returns (row, count, hidden); missing directions are zero.
    """
    row_count, hidden = good_means.shape
    extras = torch.zeros(row_count, count, hidden, dtype=torch.float32)
    if count == 0:
        return extras

    for row in range(row_count):
        data = bad_residuals[:, row, :].to(torch.float32)
        prototypes = train_som(data, grid, epochs, sigma_end, seed=seed + row)
        hits = neuron_hits(prototypes, data)
        order = torch.argsort(hits, descending=True, stable=True)
        order = order[hits[order] > 0]

        candidates = prototypes[order] - good_means[row]
        basis = main_directions[row][None].to(torch.float32)
        if orthogonalize_good:
            good_direction = F.normalize(good_means[row], p=2, dim=0)
            # The main direction is orthogonal to the good direction already, so both
            # can be part of the basis that the candidates are made orthogonal to.
            basis = torch.cat([basis, good_direction[None]])
        extras[row] = orthonormal_extension(basis, candidates, count)

    return extras


def interpolate_extras(extras: Tensor, direction_index: float, main: Tensor) -> Tensor:
    """
    Extra directions for the global direction scope: those of the residual row nearest to
    `direction_index` (the extras of different rows are not aligned, so they cannot be
    interpolated like the main direction), made orthogonal to the interpolated main
    direction `main` again. `extras` has shape (row, count, hidden).
    """
    row = min(math.floor(direction_index + 1.5), extras.shape[0] - 1)
    candidates = extras[row]
    present = candidates.norm(dim=1) > 0
    result = torch.zeros_like(candidates)
    extension = orthonormal_extension(
        main[None].to(torch.float32).cpu(),
        candidates[present].cpu(),
        int(present.sum()),
    )
    result[: extension.shape[0]] = extension
    return result


def paired_differences(bad_residuals: Tensor, good_residuals: Tensor) -> Tensor:
    """
    The per-pair difference of the residuals, (pair, row, hidden), for prompt sets whose
    entries are counterfactual twins at the same index (the harness paired training set and
    train/good_train are built that way: same language, same technical terms, same opening
    words, same final punctuation). Everything but the refusal cancels inside a row, which
    the mean of the two sets cannot do.

    The MEAN of these differences equals the difference of the means, so the classic
    direction does not change; what changes is everything computed from the spread of the
    differences (whitening, singular directions).
    """
    if bad_residuals.shape != good_residuals.shape:
        raise ValueError(
            "Paired statistics need the same number of bad and good prompts, in the same "
            f"order, but their residuals have shapes {tuple(bad_residuals.shape)} and "
            f"{tuple(good_residuals.shape)}"
        )
    return bad_residuals - good_residuals


def whiten(direction: Tensor, differences: Tensor, shrinkage: float = 0.1) -> Tensor:
    """
    The unit direction `Sigma^-1/2 d` of one residual row, where Sigma is the covariance of
    `differences` (pair, hidden) plus a ridge, i.e. the mean-difference direction corrected
    for how much the differences actually vary along each axis (LDA / "whitened diff").

    With fewer pairs than dimensions Sigma is singular, so a ridge of `shrinkage` times its
    mean eigenvalue is added. The result is computed from the thin SVD of `differences`
    (pairs x hidden), never from a hidden x hidden matrix: inside the span of the pairs the
    component is scaled by (s^2 / n + lambda)^-1/2, outside it by lambda^-1/2.
    """
    pairs, hidden = differences.shape
    if pairs < 2:
        return F.normalize(direction, p=2, dim=0)

    data = differences.to(torch.float64)
    data = data - data.mean(dim=0, keepdim=True)
    values = torch.linalg.svdvals(data)
    variances = values**2 / max(pairs - 1, 1)
    # The ridge is relative to the mean eigenvalue over the whole space, so it does not
    # depend on how the scale of the residuals is measured.
    ridge = shrinkage * float(variances.sum()) / hidden
    if ridge <= 0:
        return F.normalize(direction, p=2, dim=0)

    _, _, basis = torch.linalg.svd(data, full_matrices=False)
    vector = direction.to(torch.float64)
    inside = basis @ vector
    scales = (variances[: basis.shape[0]] + ridge).rsqrt() - ridge**-0.5
    whitened = basis.T @ (scales * inside) + vector * ridge**-0.5
    return F.normalize(whitened.to(direction.dtype), p=2, dim=0)


def paired_svd_directions(differences: Tensor, count: int) -> Tensor:
    """
    The top `count` right singular vectors of the centered paired differences of one
    residual row, (count, hidden): the directions along which the pairs differ most beyond
    their mean. Rows that do not exist (fewer pairs than `count`) are zero.

    These replace the self-organizing map as the source of the additional directions when
    the prompt sets are paired: a cluster of the SOM is a group of harmful prompts, while a
    singular vector of the differences is a direction of refusal that the pairing has
    already cleaned of language and topic.
    """
    pairs, hidden = differences.shape
    result = torch.zeros(count, hidden, dtype=differences.dtype)
    if pairs < 2 or count == 0:
        return result

    data = differences.to(torch.float64)
    data = data - data.mean(dim=0, keepdim=True)
    _, _, basis = torch.linalg.svd(data, full_matrices=False)
    available = min(count, basis.shape[0])
    result[:available] = basis[:available].to(differences.dtype)
    return result
