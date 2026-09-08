"""Training loss functions for speech enhancement models.

These losses are only exercised if Phase 6b (optional training pipeline) is
activated.  They are kept in the main ``ai`` package so that experiment
scripts can import them without importing the full training infrastructure.

Loss functions
--------------
* **SI-SNR loss** — primary objective, directly optimizes the SI-SNR metric
* **L1 / L2 losses** — simple time-domain reconstruction losses
* **Multi-resolution STFT loss** — perceptual loss based on spectral
  convergence + log-magnitude across multiple FFT sizes
* **Weighted combination** — configurable weighted sum of the above
"""

from __future__ import annotations

import numpy as np


def si_snr_loss(estimate: np.ndarray, target: np.ndarray) -> float:
    """Negative SI-SNR (lower is better, suitable for minimization).

    Parameters
    ----------
    estimate, target : np.ndarray
        1-D float64 mono signals of equal length.

    Returns
    -------
    float
        ``-SI-SNR`` in dB.  Minimizing this maximizes SI-SNR.
    """
    estimate = np.asarray(estimate, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if len(estimate) != len(target):
        raise ValueError("estimate and target must have equal lengths.")

    # Zero-mean
    target_zm = target - np.mean(target)
    estimate_zm = estimate - np.mean(estimate)

    floor = np.finfo(np.float64).eps
    target_energy = float(np.dot(target_zm, target_zm))
    if target_energy < floor:
        return 0.0

    projection = float(np.dot(estimate_zm, target_zm))
    s_target = (projection / target_energy) * target_zm
    e_noise = estimate_zm - s_target

    s_energy = float(np.dot(s_target, s_target))
    e_energy = float(np.dot(e_noise, e_noise))

    si_snr = 10.0 * np.log10((s_energy + floor) / (e_energy + floor))
    return float(-si_snr)


def l1_loss(estimate: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error (L1) between estimate and target."""
    estimate = np.asarray(estimate, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if len(estimate) != len(target):
        raise ValueError("estimate and target must have equal lengths.")
    return float(np.mean(np.abs(estimate - target)))


def l2_loss(estimate: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error (L2) between estimate and target."""
    estimate = np.asarray(estimate, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if len(estimate) != len(target):
        raise ValueError("estimate and target must have equal lengths.")
    return float(np.mean((estimate - target) ** 2))


def _stft_loss_single(
    estimate: np.ndarray,
    target: np.ndarray,
    fft_size: int,
    hop_size: int,
) -> tuple[float, float]:
    """Spectral convergence + log-magnitude loss for one FFT size."""
    window = np.hanning(fft_size).astype(np.float64)
    floor = np.finfo(np.float64).eps

    def _stft_mag(x: np.ndarray) -> np.ndarray:
        num_frames = max(0, 1 + (len(x) - fft_size) // hop_size)
        mags = np.zeros((num_frames, fft_size // 2 + 1))
        for i in range(num_frames):
            start = i * hop_size
            frame = x[start:start + fft_size] * window
            mags[i] = np.abs(np.fft.rfft(frame))
        return mags

    est_mag = _stft_mag(estimate)
    tgt_mag = _stft_mag(target)

    if est_mag.size == 0 or tgt_mag.size == 0:
        return 0.0, 0.0

    # Spectral convergence: Frobenius norm of difference / Frobenius norm of target
    diff_norm = np.sqrt(np.sum((est_mag - tgt_mag) ** 2))
    tgt_norm = np.sqrt(np.sum(tgt_mag ** 2))
    spectral_convergence = float(diff_norm / (tgt_norm + floor))

    # Log-magnitude loss
    log_mag_loss = float(np.mean(np.abs(
        np.log(np.maximum(est_mag, floor)) - np.log(np.maximum(tgt_mag, floor))
    )))

    return spectral_convergence, log_mag_loss


def multi_resolution_stft_loss(
    estimate: np.ndarray,
    target: np.ndarray,
    *,
    fft_sizes: tuple[int, ...] = (512, 1024, 2048),
    hop_ratios: tuple[float, ...] = (0.25, 0.25, 0.25),
) -> float:
    """Multi-resolution STFT perceptual loss.

    Computes spectral convergence + log-magnitude loss across multiple
    FFT sizes and averages them.  This encourages the model to preserve
    both fine and coarse spectral structure.

    Parameters
    ----------
    estimate, target : np.ndarray
        1-D float64 mono signals of equal length.
    fft_sizes : tuple of int
        FFT sizes to evaluate.
    hop_ratios : tuple of float
        Hop size as a fraction of FFT size for each resolution.
    """
    estimate = np.asarray(estimate, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if len(estimate) != len(target):
        raise ValueError("estimate and target must have equal lengths.")

    total_sc = 0.0
    total_lm = 0.0
    count = 0

    for fft_size, hop_ratio in zip(fft_sizes, hop_ratios):
        hop_size = max(1, int(fft_size * hop_ratio))
        if len(estimate) < fft_size:
            continue
        sc, lm = _stft_loss_single(estimate, target, fft_size, hop_size)
        total_sc += sc
        total_lm += lm
        count += 1

    if count == 0:
        return 0.0

    return float((total_sc + total_lm) / count)


def combined_loss(
    estimate: np.ndarray,
    target: np.ndarray,
    *,
    si_snr_weight: float = 1.0,
    l1_weight: float = 0.1,
    stft_weight: float = 0.5,
) -> dict[str, float]:
    """Weighted combination of SI-SNR, L1, and multi-resolution STFT losses.

    Returns a dict with individual components and the total, for logging.
    """
    loss_si_snr = si_snr_loss(estimate, target)
    loss_l1 = l1_loss(estimate, target)
    loss_stft = multi_resolution_stft_loss(estimate, target)

    total = (
        si_snr_weight * loss_si_snr
        + l1_weight * loss_l1
        + stft_weight * loss_stft
    )

    return {
        "si_snr_loss": loss_si_snr,
        "l1_loss": loss_l1,
        "stft_loss": loss_stft,
        "total_loss": total,
        "weights": {
            "si_snr": si_snr_weight,
            "l1": l1_weight,
            "stft": stft_weight,
        },
    }
