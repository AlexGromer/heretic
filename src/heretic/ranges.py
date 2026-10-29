# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Helpers for the configurable search ranges of the modifiers (fork addition, v3).

import math


def check_range(
    name: str | None,
    value: tuple[float, float],
    *,
    lower_limit: float | None = 0.0,
    upper_limit: float | None = None,
    strictly_positive: bool = False,
) -> tuple[float, float]:
    """
    Validates a closed search interval (low, high) and returns it.

    - Both bounds must be finite numbers, and low <= high (low == high pins the parameter).
    - `lower_limit` / `upper_limit` bound the whole interval. The default lower limit of 0
      rejects negative values, which would make no sense for weights and fractions.
      Pass `lower_limit=None` where a negative lower bound is meaningful.
    - `strictly_positive` additionally requires low > 0 (needed for log-uniform sampling).

    Raises ValueError with a message that names the setting, so that a pydantic
    validator surfaces it directly to the user.
    """
    low, high = value
    name = name or "range"

    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"{name}: bounds must be finite numbers, got {list(value)}")

    if low > high:
        raise ValueError(
            f"{name}: lower bound {low} is greater than upper bound {high}"
        )

    if lower_limit is not None and low < lower_limit:
        raise ValueError(
            f"{name}: lower bound {low} is below the allowed minimum {lower_limit}"
        )

    if upper_limit is not None and high > upper_limit:
        raise ValueError(
            f"{name}: upper bound {high} is above the allowed maximum {upper_limit}"
        )

    if strictly_positive and low <= 0:
        raise ValueError(f"{name}: lower bound must be greater than 0, got {low}")

    return value


def scaled_floor(fraction: float, count: int) -> int:
    """
    floor(fraction * count) that is robust against float representation errors
    (0.1 * 30 == 3.0000000000000004, 0.29 * 100 == 28.999999999999996).
    """
    return math.floor(fraction * count + 1e-9)
