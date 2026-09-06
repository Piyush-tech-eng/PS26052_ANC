"""Reusable, source-agnostic metrics for the Module 7 ANC benchmark."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _vector(values: np.ndarray | list[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return array


def compute_signal_power(values: np.ndarray | list[float]) -> float:
    """Return mean-square signal power."""

    array = _vector(values, "values")
    return float(np.mean(array**2))


def compute_noise_reduction_db(
    baseline: np.ndarray | list[float], residual: np.ndarray | list[float]
) -> float:
    """Return attenuation relative to a no-control baseline in dB."""

    baseline_power = compute_signal_power(baseline)
    residual_power = compute_signal_power(residual)
    floor = np.finfo(np.float64).eps
    return float(10.0 * np.log10((baseline_power + floor) / (residual_power + floor)))


def compute_convergence_profile(
    residual: np.ndarray | list[float], window_size: int = 256
) -> np.ndarray:
    """Return a causal moving mean-square residual profile."""

    array = _vector(residual, "residual")
    if not isinstance(window_size, (int, np.integer)) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    squared = array**2
    cumulative = np.cumsum(squared, dtype=np.float64)
    output = np.empty_like(squared)
    for index in range(len(array)):
        start = max(0, index - int(window_size) + 1)
        total = cumulative[index] - (cumulative[start - 1] if start else 0.0)
        output[index] = total / (index - start + 1)
    return output


def compute_segment_mse(
    estimate: np.ndarray | list[float], target: np.ndarray | list[float], *, segments: int = 4
) -> dict[str, float | list[float]]:
    """Return full-run and equal-duration segment MSE/RMSE."""

    estimate_array = _vector(estimate, "estimate")
    target_array = _vector(target, "target")
    if len(estimate_array) != len(target_array):
        raise ValueError("estimate and target must have equal lengths.")
    if not isinstance(segments, (int, np.integer)) or segments <= 0:
        raise ValueError("segments must be a positive integer.")
    error_squared = (estimate_array - target_array) ** 2
    chunks = np.array_split(error_squared, int(segments))
    segment_mse = [float(np.mean(chunk)) for chunk in chunks if len(chunk)]
    mse = float(np.mean(error_squared))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "segment_mse": segment_mse}


def compute_band_attenuation(
    baseline: np.ndarray | list[float], residual: np.ndarray | list[float], sampling_rate_hz: int,
    *, bands_hz: tuple[tuple[float, float], ...] = ((0.0, 500.0), (500.0, 2_000.0), (2_000.0, 4_000.0)),
) -> dict[str, float]:
    """Return FFT-power attenuation for each requested frequency band."""

    baseline_array = _vector(baseline, "baseline")
    residual_array = _vector(residual, "residual")
    if len(baseline_array) != len(residual_array):
        raise ValueError("baseline and residual must have equal lengths.")
    if not isinstance(sampling_rate_hz, (int, np.integer)) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be a positive integer.")
    frequencies = np.fft.rfftfreq(len(baseline_array), d=1.0 / sampling_rate_hz)
    baseline_power = np.abs(np.fft.rfft(baseline_array)) ** 2
    residual_power = np.abs(np.fft.rfft(residual_array)) ** 2
    floor = np.finfo(np.float64).eps
    output: dict[str, float] = {}
    for lower, upper in bands_hz:
        if lower < 0 or upper <= lower:
            raise ValueError("Frequency bands must have 0 <= lower < upper.")
        mask = (frequencies >= lower) & (frequencies < upper)
        if not mask.any():
            continue
        output[f"{lower:g}-{upper:g}Hz"] = float(
            10.0 * np.log10((np.mean(baseline_power[mask]) + floor) / (np.mean(residual_power[mask]) + floor))
        )
    return output


def detect_divergence(
    residual: np.ndarray | list[float], coefficient_history: np.ndarray | None = None,
    *, coefficient_norm_limit: float = 1e6
) -> dict[str, float | bool]:
    """Report finite-value and coefficient-growth stability indicators."""

    residual_array = np.asarray(residual, dtype=np.float64)
    finite_residual = residual_array.ndim == 1 and residual_array.size > 0 and np.isfinite(residual_array).all()
    maximum_norm = 0.0
    finite_coefficients = True
    if coefficient_history is not None:
        history = np.asarray(coefficient_history, dtype=np.float64)
        finite_coefficients = history.ndim == 2 and history.size > 0 and np.isfinite(history).all()
        if finite_coefficients:
            maximum_norm = float(np.max(np.linalg.norm(history, axis=1)))
    divergent = not finite_residual or not finite_coefficients or maximum_norm > coefficient_norm_limit
    return {
        "divergent": bool(divergent), "residual_finite": bool(finite_residual),
        "coefficients_finite": bool(finite_coefficients), "max_coefficient_norm": maximum_norm,
    }


def evaluate_scenario(
    residual: np.ndarray | list[float], *, sampling_rate_hz: int, baseline_residual: np.ndarray | list[float] | None = None,
    target: np.ndarray | list[float] | None = None, coefficient_history: np.ndarray | None = None,
    convergence_window: int = 256,
) -> dict[str, object]:
    """Compute the standard flat-ish metric record for one controller run."""

    residual_array = _vector(residual, "residual")
    profile = compute_convergence_profile(residual_array, convergence_window)
    metrics: dict[str, object] = {
        "initial_residual_power": float(profile[0]),
        "final_residual_power": float(np.mean(residual_array[-min(1000, len(residual_array)): ]**2)),
        "residual_power": compute_signal_power(residual_array),
        "convergence_profile": profile,
        "stability": detect_divergence(residual_array, coefficient_history),
    }
    if baseline_residual is not None:
        baseline_array = _vector(baseline_residual, "baseline_residual")
        if len(baseline_array) != len(residual_array):
            raise ValueError("baseline_residual must match residual length.")
        metrics["attenuation_db_vs_no_control"] = compute_noise_reduction_db(baseline_array, residual_array)
        metrics["band_attenuation_db"] = compute_band_attenuation(baseline_array, residual_array, sampling_rate_hz)
    else:
        metrics["attenuation_db_vs_no_control"] = None
        metrics["band_attenuation_db"] = {}
    if target is None:
        metrics["target_available"] = False
        metrics["mse"] = None
        metrics["rmse"] = None
        metrics["segment_mse"] = None
    else:
        mse = compute_segment_mse(residual_array, target)
        metrics.update(mse)
        metrics["target_available"] = True
    return metrics
