"""Dimensionless norm-change metrics for block-boundary activations."""
from __future__ import annotations

import math


def norm_change_metrics(in_norm: float, out_norm: float) -> dict[str, float]:
    """Return output/input, relative change, and log ratio for positive norms."""
    if not math.isfinite(in_norm) or not math.isfinite(out_norm):
        raise ValueError("Input and output norms must be finite")
    if in_norm <= 0 or out_norm <= 0:
        raise ValueError("Input and output norms must be positive")

    ratio = out_norm / in_norm
    return {
        "output_input_norm_ratio": ratio,
        "relative_norm_change": ratio - 1.0,
        "log_norm_ratio": math.log(out_norm) - math.log(in_norm),
    }
