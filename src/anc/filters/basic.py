from __future__ import annotations

import numpy as np

from anc.signals.signal import Signal


def first_order_lowpass(
    signal: Signal,
    alpha: float,
) -> Signal:
    """
    First-order recursive low-pass filter.

    y[n] = (1 - alpha)x[n] + alpha y[n-1]
    """

    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1.")

    x = signal.samples
    y = np.zeros_like(x)

    if len(x) == 0:
        return Signal(
            samples=y,
            sampling_rate=signal.sampling_rate,
            metadata={
                **signal.metadata,
                "filter_type": "first_order_lowpass",
                "alpha": alpha,
            },
        )

    y[0] = (1.0 - alpha) * x[0]

    for n in range(1, len(x)):
        y[n] = (
            (1.0 - alpha) * x[n]
            + alpha * y[n - 1]
        )

    return Signal(
        samples=y,
        sampling_rate=signal.sampling_rate,
        metadata={
            **signal.metadata,
            "filter_type": "first_order_lowpass",
            "alpha": alpha,
        },
    )


def fir_filter(
    signal: Signal,
    coefficients: np.ndarray | list[float],
) -> Signal:
    """
    Apply a causal finite impulse response filter.

    y[n] = sum_{k=0}^{L-1} h[k] x[n-k]

    Samples before n=0 are assumed to be zero.
    Output length is identical to input length.
    """

    h = np.asarray(coefficients, dtype=np.float64)

    if h.ndim != 1:
        raise ValueError(
            "FIR coefficients must be one-dimensional."
        )

    if len(h) == 0:
        raise ValueError(
            "FIR coefficient array cannot be empty."
        )

    x = signal.samples
    y = np.zeros_like(x, dtype=np.float64)

    for n in range(len(x)):
        accumulator = 0.0

        for k in range(len(h)):
            input_index = n - k

            if input_index >= 0:
                accumulator += (
                    h[k] * x[input_index]
                )

        y[n] = accumulator

    return Signal(
        samples=y,
        sampling_rate=signal.sampling_rate,
        metadata={
            **signal.metadata,
            "filter_type": "fir",
            "fir_coefficients": h.tolist(),
            "fir_length": len(h),
        },
    )