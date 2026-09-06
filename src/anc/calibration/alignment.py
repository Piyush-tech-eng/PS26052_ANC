"""Delay estimation and explicit signal alignment.

Module 6.02 introduces a reusable timing-calibration layer for ANC
recordings and measurements.

The core convention is:

    positive lag

means that the target signal occurs later than the reference signal.

For example:

    target[n] ≈ reference[n - lag]

when lag > 0.

The module intentionally keeps timing calibration separate from the ANC
controller.  The controller should receive already synchronized signals
rather than silently applying unknown timing corrections internally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_vector(
    values: np.ndarray | list[float],
    *,
    name: str,
) -> np.ndarray:
    """Validate and return a finite one-dimensional float64 vector."""

    vector = np.asarray(
        values,
        dtype=np.float64,
    )

    if vector.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if len(vector) == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return vector.copy()


def _nonnegative_integer(
    value: int,
    *,
    name: str,
) -> int:
    """Validate a non-negative integer."""

    if (
        not isinstance(
            value,
            (int, np.integer),
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{name} must not be negative."
        )

    return int(value)


@dataclass(frozen=True)
class DelayEstimate:
    """Estimated integer timing offset between two signals.

    Convention:

        lag_samples > 0

    means the target occurs later than the reference.
    """

    lag_samples: int

    peak_correlation: float

    normalized_peak_correlation: float

    reference_length: int

    target_length: int

    max_delay_samples: int | None


@dataclass(frozen=True)
class AlignmentResult:
    """Aligned overlapping portions of two signals."""

    reference: np.ndarray

    target: np.ndarray

    delay: DelayEstimate

    reference_start: int

    target_start: int

    overlap_samples: int


def _overlap_for_lag(
    reference: np.ndarray,
    target: np.ndarray,
    lag_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return overlapping segments for one lag.

    Positive lag means target is delayed relative to reference:

        target[n] ≈ reference[n - lag]

    Therefore, when lag > 0:

        reference[0:]
        target[lag:]

    are compared.
    """

    reference_length = len(reference)
    target_length = len(target)

    if lag_samples >= 0:

        reference_start = 0
        target_start = lag_samples

    else:

        reference_start = -lag_samples
        target_start = 0

    overlap_length = min(
        reference_length
        - reference_start,
        target_length
        - target_start,
    )

    if overlap_length <= 0:
        return (
            np.empty(
                0,
                dtype=np.float64,
            ),
            np.empty(
                0,
                dtype=np.float64,
            ),
        )

    return (
        reference[
            reference_start:
            reference_start
            + overlap_length
        ],
        target[
            target_start:
            target_start
            + overlap_length
        ],
    )


def _normalized_correlation(
    reference: np.ndarray,
    target: np.ndarray,
) -> float:
    """Return normalized zero-mean correlation."""

    if len(reference) != len(target):
        raise ValueError(
            "Signals must have equal lengths."
        )

    if len(reference) == 0:
        raise ValueError(
            "Signals must not be empty."
        )

    reference_centered = (
        reference
        - np.mean(reference)
    )

    target_centered = (
        target
        - np.mean(target)
    )

    denominator = (
        np.linalg.norm(
            reference_centered
        )
        * np.linalg.norm(
            target_centered
        )
    )

    if denominator <= np.finfo(
        np.float64
    ).eps:
        return 0.0

    return float(
        np.dot(
            reference_centered,
            target_centered,
        )
        / denominator
    )


def estimate_delay(
    reference: np.ndarray | list[float],
    target: np.ndarray | list[float],
    *,
    max_delay_samples: int | None = None,
    minimum_overlap_samples: int = 32,
    allow_polarity_inversion: bool = True,
) -> DelayEstimate:
    """Estimate integer delay using normalized cross-correlation.

    Parameters
    ----------
    reference:
        Timing reference signal.

    target:
        Signal whose timing is estimated relative to ``reference``.

    max_delay_samples:
        Optional maximum absolute lag to search.

    minimum_overlap_samples:
        Minimum number of overlapping samples required for a candidate lag.

    allow_polarity_inversion:
        If True, delay selection maximizes absolute normalized correlation.
        This allows calibration to succeed when an acoustic/electrical path
        inverts signal polarity.

    Returns
    -------
    DelayEstimate

    Convention
    ----------
    A positive estimated lag means the target occurs later than the
    reference.
    """

    reference_vector = _finite_vector(
        reference,
        name="reference",
    )

    target_vector = _finite_vector(
        target,
        name="target",
    )

    minimum_overlap = _nonnegative_integer(
        minimum_overlap_samples,
        name="minimum_overlap_samples",
    )

    if minimum_overlap == 0:
        raise ValueError(
            "minimum_overlap_samples must be positive."
        )

    if max_delay_samples is not None:

        maximum_delay = (
            _nonnegative_integer(
                max_delay_samples,
                name="max_delay_samples",
            )
        )

    else:

        maximum_delay = max(
            len(reference_vector),
            len(target_vector),
        ) - 1

    best_lag: int | None = None

    best_score = -np.inf

    best_correlation = 0.0

    best_peak = 0.0

    for lag in range(
        -maximum_delay,
        maximum_delay + 1,
    ):

        reference_overlap, target_overlap = (
            _overlap_for_lag(
                reference_vector,
                target_vector,
                lag,
            )
        )

        if (
            len(reference_overlap)
            < minimum_overlap
        ):
            continue

        correlation = (
            _normalized_correlation(
                reference_overlap,
                target_overlap,
            )
        )

        score = (
            abs(correlation)
            if allow_polarity_inversion
            else correlation
        )

        peak = float(
            np.dot(
                reference_overlap,
                target_overlap,
            )
        )

        if score > best_score:

            best_lag = lag

            best_score = score

            best_correlation = (
                correlation
            )

            best_peak = peak

    if best_lag is None:
        raise ValueError(
            "No valid lag produced the required "
            "minimum overlap."
        )

    return DelayEstimate(
        lag_samples=int(
            best_lag
        ),
        peak_correlation=float(
            best_peak
        ),
        normalized_peak_correlation=float(
            best_correlation
        ),
        reference_length=len(
            reference_vector
        ),
        target_length=len(
            target_vector
        ),
        max_delay_samples=(
            max_delay_samples
        ),
    )


def align_signals(
    reference: np.ndarray | list[float],
    target: np.ndarray | list[float],
    *,
    delay: DelayEstimate | None = None,
    max_delay_samples: int | None = None,
    minimum_overlap_samples: int = 32,
    allow_polarity_inversion: bool = True,
) -> AlignmentResult:
    """Estimate timing and return explicitly aligned overlapping signals.

    If ``delay`` is supplied, no new delay estimation is performed.

    The returned arrays always have identical lengths and represent the
    maximum overlapping region consistent with the estimated lag.
    """

    reference_vector = _finite_vector(
        reference,
        name="reference",
    )

    target_vector = _finite_vector(
        target,
        name="target",
    )

    if delay is None:

        delay = estimate_delay(
            reference_vector,
            target_vector,
            max_delay_samples=(
                max_delay_samples
            ),
            minimum_overlap_samples=(
                minimum_overlap_samples
            ),
            allow_polarity_inversion=(
                allow_polarity_inversion
            ),
        )

    elif not isinstance(
        delay,
        DelayEstimate,
    ):
        raise TypeError(
            "delay must be a DelayEstimate "
            "or None."
        )

    lag = delay.lag_samples

    if lag >= 0:

        reference_start = 0
        target_start = lag

    else:

        reference_start = -lag
        target_start = 0

    reference_aligned, target_aligned = (
        _overlap_for_lag(
            reference_vector,
            target_vector,
            lag,
        )
    )

    if (
        len(reference_aligned)
        < minimum_overlap_samples
    ):
        raise ValueError(
            "Aligned overlap is shorter than "
            "minimum_overlap_samples."
        )

    return AlignmentResult(
        reference=reference_aligned.copy(),
        target=target_aligned.copy(),
        delay=delay,
        reference_start=int(
            reference_start
        ),
        target_start=int(
            target_start
        ),
        overlap_samples=int(
            len(reference_aligned)
        ),
    )


def apply_known_delay(
    signal: np.ndarray | list[float],
    *,
    delay_samples: int,
) -> np.ndarray:
    """Apply an integer delay without changing signal length.

    Positive delay inserts leading zeros and truncates the end.

    Negative delay advances the signal and inserts trailing zeros.

    This helper is useful for controlled calibration experiments.
    """

    vector = _finite_vector(
        signal,
        name="signal",
    )

    if (
        not isinstance(
            delay_samples,
            (int, np.integer),
        )
        or isinstance(
            delay_samples,
            bool,
        )
    ):
        raise TypeError(
            "delay_samples must be an integer."
        )

    delay = int(
        delay_samples
    )

    output = np.zeros_like(
        vector
    )

    if delay >= 0:

        if delay < len(vector):

            output[delay:] = (
                vector[
                    :len(vector)
                    - delay
                ]
            )

    else:

        advance = -delay

        if advance < len(vector):

            output[
                :len(vector)
                - advance
            ] = vector[
                advance:
            ]

    return output