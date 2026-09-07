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


def compute_si_snr(
    estimate: np.ndarray | list[float], target: np.ndarray | list[float],
) -> float:
    """Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB.

    SI-SNR is the standard metric for speech separation/enhancement and
    directly optimizes the SNR metric the PS26052 problem statement targets.

    SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)

    where s_target = (<s_hat, s> / ||s||^2) * s  and  e_noise = s_hat - s_target.
    """
    estimate_array = _vector(estimate, "estimate")
    target_array = _vector(target, "target")
    if len(estimate_array) != len(target_array):
        raise ValueError("estimate and target must have equal lengths.")

    # Zero-mean normalization
    target_zm = target_array - np.mean(target_array)
    estimate_zm = estimate_array - np.mean(estimate_array)

    floor = np.finfo(np.float64).eps
    target_energy = float(np.dot(target_zm, target_zm))
    if target_energy < floor:
        return 0.0

    # s_target = (<s_hat, s> / ||s||^2) * s
    projection = float(np.dot(estimate_zm, target_zm))
    s_target = (projection / target_energy) * target_zm

    # e_noise = s_hat - s_target
    e_noise = estimate_zm - s_target

    s_target_energy = float(np.dot(s_target, s_target))
    e_noise_energy = float(np.dot(e_noise, e_noise))

    return float(10.0 * np.log10((s_target_energy + floor) / (e_noise_energy + floor)))


def compute_stoi(
    estimate: np.ndarray | list[float],
    target: np.ndarray | list[float],
    sampling_rate_hz: int,
    *,
    frame_length_ms: float = 25.625,
    frame_shift_ms: float = 10.0,
) -> float:
    """Compute an approximation of Short-Time Objective Intelligibility (STOI).

    This is a simplified STOI computation based on short-time temporal
    envelope correlation. For production evaluation, use the ``pystoi``
    package; this built-in version follows the same "pure function on
    numpy arrays" pattern as the other metrics in this module and requires
    no external dependencies.

    Returns a value in [0, 1] where 1 indicates perfect intelligibility.
    """
    estimate_array = _vector(estimate, "estimate")
    target_array = _vector(target, "target")
    if len(estimate_array) != len(target_array):
        raise ValueError("estimate and target must have equal lengths.")
    if not isinstance(sampling_rate_hz, (int, np.integer)) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be a positive integer.")

    frame_length = max(1, int(frame_length_ms * sampling_rate_hz / 1000.0))
    frame_shift = max(1, int(frame_shift_ms * sampling_rate_hz / 1000.0))

    num_frames = max(0, (len(target_array) - frame_length) // frame_shift + 1)
    if num_frames < 1:
        return 0.0

    correlations = []
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        t_frame = target_array[start:end]
        e_frame = estimate_array[start:end]

        t_mean = np.mean(t_frame)
        e_mean = np.mean(e_frame)
        t_centered = t_frame - t_mean
        e_centered = e_frame - e_mean

        t_std = np.std(t_frame)
        e_std = np.std(e_frame)

        floor = np.finfo(np.float64).eps
        if t_std < floor or e_std < floor:
            continue

        correlation = float(np.mean(t_centered * e_centered) / (t_std * e_std))
        correlations.append(max(-1.0, min(1.0, correlation)))

    if not correlations:
        return 0.0

    return float(np.mean(correlations))


def compute_pesq_approx(
    estimate: np.ndarray | list[float],
    target: np.ndarray | list[float],
    sampling_rate_hz: int,
) -> float:
    """Compute a simplified PESQ-like perceptual quality estimate.

    This is NOT a standards-compliant ITU-T P.862 implementation. It provides
    a lightweight approximation based on segmental SNR and spectral distortion
    that correlates with perceptual quality. For standards-compliant PESQ,
    use the ``pesq`` Python package.

    Returns a value in approximately [1.0, 4.5] (MOS-LQO scale).
    """
    estimate_array = _vector(estimate, "estimate")
    target_array = _vector(target, "target")
    if len(estimate_array) != len(target_array):
        raise ValueError("estimate and target must have equal lengths.")
    if not isinstance(sampling_rate_hz, (int, np.integer)) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be a positive integer.")

    # Segmental SNR (frame-by-frame SNR averaged over frames)
    frame_length = max(1, int(0.032 * sampling_rate_hz))  # 32ms frames
    frame_shift = max(1, int(0.016 * sampling_rate_hz))   # 16ms shift
    num_frames = max(0, (len(target_array) - frame_length) // frame_shift + 1)

    floor = np.finfo(np.float64).eps
    seg_snrs = []
    spectral_dists = []

    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        t_frame = target_array[start:end]
        e_frame = estimate_array[start:end]

        # Frame-level SNR (clamped to [-10, 35] dB as in standard seg-SNR)
        signal_power = float(np.mean(t_frame ** 2))
        noise_power = float(np.mean((t_frame - e_frame) ** 2))
        if signal_power > floor:
            if noise_power > floor:
                snr = 10.0 * np.log10(signal_power / noise_power)
                seg_snrs.append(max(-10.0, min(35.0, snr)))
            else:
                # Near-perfect reconstruction: assign maximum clamped SNR
                seg_snrs.append(35.0)

        # Log-spectral distortion
        t_spec = np.abs(np.fft.rfft(t_frame * np.hanning(frame_length)))
        e_spec = np.abs(np.fft.rfft(e_frame * np.hanning(frame_length)))
        t_spec = np.maximum(t_spec, floor)
        e_spec = np.maximum(e_spec, floor)
        lsd = float(np.sqrt(np.mean((np.log10(t_spec) - np.log10(e_spec)) ** 2)))
        spectral_dists.append(lsd)

    if not seg_snrs:
        return 1.0

    avg_seg_snr = float(np.mean(seg_snrs))
    avg_lsd = float(np.mean(spectral_dists))

    # Map to MOS-LQO-like scale [1.0, 4.5]
    # Higher seg-SNR and lower spectral distortion → higher quality
    # This is a heuristic mapping, not ITU-T P.862
    quality = 1.0 + 3.5 * (1.0 / (1.0 + np.exp(-(avg_seg_snr - 10.0) / 8.0)))
    distortion_penalty = min(1.0, avg_lsd / 2.0)  # Penalize high spectral distortion
    quality = quality * (1.0 - 0.4 * distortion_penalty)

    return float(max(1.0, min(4.5, quality)))


def evaluate_scenario(
    residual: np.ndarray | list[float], *, sampling_rate_hz: int, baseline_residual: np.ndarray | list[float] | None = None,
    target: np.ndarray | list[float] | None = None, coefficient_history: np.ndarray | None = None,
    convergence_window: int = 256,
) -> dict[str, object]:
    """Compute the standard flat-ish metric record for one controller run.

    When ``target`` is provided, speech-perceptual metrics (SI-SNR, STOI,
    PESQ-approx) are also computed alongside classical MSE/RMSE metrics.
    """

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
        metrics["si_snr_db"] = None
        metrics["stoi"] = None
        metrics["pesq_approx"] = None
    else:
        target_array = _vector(target, "target")
        mse = compute_segment_mse(residual_array, target_array)
        metrics.update(mse)
        metrics["target_available"] = True
        metrics["si_snr_db"] = compute_si_snr(residual_array, target_array)
        metrics["stoi"] = compute_stoi(residual_array, target_array, sampling_rate_hz)
        metrics["pesq_approx"] = compute_pesq_approx(residual_array, target_array, sampling_rate_hz)
    return metrics

