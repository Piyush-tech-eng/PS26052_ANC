"""Lightweight on-Pi FxNLMS adaptive filter for edge execution."""

from __future__ import annotations

import numpy as np


class LocalFxNLMS:
    """Fast, memory-efficient Filtered-x NLMS adaptive filter for Raspberry Pi 3.

    Parameters
    ----------
    filter_length : int
        Number of adaptive FIR filter taps (default 64).
    step_size : float
        Normalized adaptation step size mu (default 0.01).
    epsilon : float
        Regularization constant to prevent division by zero (default 1e-6).
    """

    def __init__(
        self,
        filter_length: int = 64,
        step_size: float = 0.01,
        epsilon: float = 1e-6,
    ) -> None:
        self.length = filter_length
        self.mu = step_size
        self.eps = epsilon

        self.w = np.zeros(filter_length, dtype=np.float64)
        self.x_buf = np.zeros(filter_length, dtype=np.float64)
        self.sec_path = np.zeros(filter_length, dtype=np.float64)
        self.sec_path[0] = 1.0  # Default direct acoustic path identity

    def set_secondary_path(self, impulse_response: np.ndarray) -> None:
        """Set estimated secondary path model."""
        ir = np.asarray(impulse_response, dtype=np.float64)
        if len(ir) < self.length:
            self.sec_path = np.pad(ir, (0, self.length - len(ir)))
        else:
            self.sec_path = ir[:self.length].copy()

    def process_sample(self, ref_sample: float, err_sample: float) -> tuple[float, float]:
        """Process one sample of reference and error microphone signals.

        Returns
        -------
        tuple of (antinoise_sample, residual_error_sample)
        """
        # Shift reference delay line
        self.x_buf[1:] = self.x_buf[:-1]
        self.x_buf[0] = ref_sample

        # Antinoise prediction
        y = float(np.dot(self.w, self.x_buf))

        # Residual cancellation error
        e = err_sample - y

        # Filtered reference via secondary path
        x_filtered = float(np.dot(self.sec_path, self.x_buf))

        # Normalization energy
        norm = np.dot(self.x_buf, self.x_buf) + self.eps

        # Update weights: w[n+1] = w[n] + (mu / norm) * e[n] * x_filt[n]
        self.w += (self.mu / norm) * e * self.x_buf

        return y, e

    def process_frame(
        self,
        ref_frame: np.ndarray,
        err_frame: np.ndarray,
    ) -> np.ndarray:
        """Process a full frame sample-by-sample."""
        ref = np.asarray(ref_frame, dtype=np.float64).ravel()
        err = np.asarray(err_frame, dtype=np.float64).ravel()
        out = np.zeros_like(err)

        for i in range(len(err)):
            _, out[i] = self.process_sample(ref[i], err[i])

        return out
