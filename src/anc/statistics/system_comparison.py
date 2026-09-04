from __future__ import annotations

import numpy as np


def align_impulse_responses(
    true_impulse_response: np.ndarray,
    estimated_impulse_response: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align two causal FIR impulse responses by zero-padding
    the shorter response to the length of the longer one.
    """

    true_h = np.asarray(
        true_impulse_response,
        dtype=np.float64,
    )

    estimated_h = np.asarray(
        estimated_impulse_response,
        dtype=np.float64,
    )

    if true_h.ndim != 1:
        raise ValueError(
            "True impulse response must be one-dimensional."
        )

    if estimated_h.ndim != 1:
        raise ValueError(
            "Estimated impulse response must be one-dimensional."
        )

    if len(true_h) == 0:
        raise ValueError(
            "True impulse response must not be empty."
        )

    if len(estimated_h) == 0:
        raise ValueError(
            "Estimated impulse response must not be empty."
        )

    if not np.isfinite(true_h).all():
        raise ValueError(
            "True impulse response contains NaN or Inf."
        )

    if not np.isfinite(estimated_h).all():
        raise ValueError(
            "Estimated impulse response contains NaN or Inf."
        )

    length = max(
        len(true_h),
        len(estimated_h),
    )

    aligned_true = np.zeros(
        length,
        dtype=np.float64,
    )

    aligned_estimated = np.zeros(
        length,
        dtype=np.float64,
    )

    aligned_true[:len(true_h)] = true_h

    aligned_estimated[:len(estimated_h)] = (
        estimated_h
    )

    return (
        aligned_true,
        aligned_estimated,
    )


def coefficient_error(
    true_impulse_response: np.ndarray,
    estimated_impulse_response: np.ndarray,
) -> np.ndarray:
    """
    Compute:

        error = h_true - h_estimated

    after causal zero-padding alignment.
    """

    true_h, estimated_h = (
        align_impulse_responses(
            true_impulse_response,
            estimated_impulse_response,
        )
    )

    return true_h - estimated_h


def coefficient_mse(
    true_impulse_response: np.ndarray,
    estimated_impulse_response: np.ndarray,
) -> float:
    """
    Compute mean squared coefficient error.
    """

    error = coefficient_error(
        true_impulse_response,
        estimated_impulse_response,
    )

    return float(
        np.mean(error ** 2)
    )


def coefficient_rmse(
    true_impulse_response: np.ndarray,
    estimated_impulse_response: np.ndarray,
) -> float:
    """
    Compute root mean squared coefficient error.
    """

    return float(
        np.sqrt(
            coefficient_mse(
                true_impulse_response,
                estimated_impulse_response,
            )
        )
    )


def relative_coefficient_error(
    true_impulse_response: np.ndarray,
    estimated_impulse_response: np.ndarray,
) -> float:
    """
    Compute:

        ||h_true - h_estimated||_2
        -------------------------
               ||h_true||_2
    """

    true_h, estimated_h = (
        align_impulse_responses(
            true_impulse_response,
            estimated_impulse_response,
        )
    )

    numerator = np.linalg.norm(
        true_h - estimated_h
    )

    denominator = (
        np.linalg.norm(true_h)
        + np.finfo(np.float64).eps
    )

    return float(
        numerator / denominator
    )


def coefficient_correlation(
    true_impulse_response: np.ndarray,
    estimated_impulse_response: np.ndarray,
) -> float:
    """
    Compute zero-mean normalized correlation between
    true and estimated impulse responses.
    """

    true_h, estimated_h = (
        align_impulse_responses(
            true_impulse_response,
            estimated_impulse_response,
        )
    )

    true_centered = (
        true_h - np.mean(true_h)
    )

    estimated_centered = (
        estimated_h - np.mean(estimated_h)
    )

    denominator = (
        np.linalg.norm(true_centered)
        * np.linalg.norm(
            estimated_centered
        )
    )

    if denominator == 0.0:
        raise ValueError(
            "Coefficient correlation is undefined when "
            "either impulse response has zero variance."
        )

    return float(
        np.dot(
            true_centered,
            estimated_centered,
        )
        / denominator
    )