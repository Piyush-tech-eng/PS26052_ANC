"""Stateful, frame-by-frame FxNLMS adaptive filter for streaming ANC.

This is the streaming counterpart of the offline ANC engine in
``anc.replay.engine``.  The key difference is that this class **persists
filter coefficients and delay-line state across calls** instead of
processing a whole pre-loaded array in one shot.

The math is identical to the existing replay engine (FxNLMS update):

    e[n] = d[n] + S(z) * y[n]           (error signal)
    x_f[n] = S_hat(z) * x[n]            (filtered reference)
    w[n+1] = w[n] - mu * e[n] * x_f[n] / (eps + ||x_f||^2)

Usage in the hybrid engine:

    anc = FrameANC(filter_length=64, step_size=0.01, ...)
    for ref_frame, err_frame in stream:
        residual = anc.process_frame(ref_frame, err_frame)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrameANCConfig:
    """Configuration for the frame-by-frame ANC processor."""

    filter_length: int = 64
    step_size: float = 0.01
    epsilon: float = 1e-6
    algorithm: str = "fxnlms"  # "fxlms", "fxnlms", "lms", "nlms", "none"

    def __post_init__(self) -> None:
        if self.filter_length <= 0:
            raise ValueError("filter_length must be positive.")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive.")
        if self.algorithm not in {"fxlms", "fxnlms", "lms", "nlms", "none"}:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")


class FrameANC:
    """Stateful, frame-by-frame adaptive ANC filter.

    Parameters
    ----------
    config : FrameANCConfig
        Filter configuration.
    secondary_path_true : np.ndarray
        True secondary path impulse response (for cancellation output).
    secondary_path_model : np.ndarray
        Modelled secondary path impulse response (for filtered-x update).
    """

    def __init__(
        self,
        config: FrameANCConfig,
        secondary_path_true: np.ndarray,
        secondary_path_model: np.ndarray,
    ) -> None:
        self._config = config
        self._secondary_true = np.asarray(
            secondary_path_true, dtype=np.float64
        ).copy()
        self._secondary_model = np.asarray(
            secondary_path_model, dtype=np.float64
        ).copy()

        if self._secondary_true.ndim != 1 or len(self._secondary_true) == 0:
            raise ValueError("secondary_path_true must be a non-empty 1-D array.")
        if self._secondary_model.ndim != 1 or len(self._secondary_model) == 0:
            raise ValueError("secondary_path_model must be a non-empty 1-D array.")

        # Filter coefficients
        self._coefficients = np.zeros(config.filter_length, dtype=np.float64)

        # Delay lines
        self._controller_state = np.zeros(config.filter_length, dtype=np.float64)
        self._secondary_state = np.zeros(
            len(self._secondary_true), dtype=np.float64
        )
        self._model_state = np.zeros(
            len(self._secondary_model), dtype=np.float64
        )
        self._filtered_ref_state = np.zeros(
            config.filter_length, dtype=np.float64
        )

    @property
    def coefficients(self) -> np.ndarray:
        """Current filter coefficients (read-only copy)."""
        return self._coefficients.copy()

    def process_sample(
        self, reference: float, measured: float
    ) -> float:
        """Process one sample through the ANC filter.

        Parameters
        ----------
        reference : float
            Reference microphone sample x[n].
        measured : float
            Error/measured microphone sample d[n].

        Returns
        -------
        float
            Residual error e[n] = d[n] + S(z)*y[n].
        """
        cfg = self._config

        # Update controller delay line
        if cfg.filter_length > 1:
            self._controller_state[1:] = self._controller_state[:-1]
        self._controller_state[0] = reference

        # Controller output: y[n] = w^T * x[n]
        y_n = float(np.dot(self._coefficients, self._controller_state))

        # True secondary path: y_s[n] = S(z) * y[n]
        if len(self._secondary_state) > 1:
            self._secondary_state[1:] = self._secondary_state[:-1]
        self._secondary_state[0] = y_n
        y_s = float(np.dot(self._secondary_true, self._secondary_state))

        # Error: e[n] = d[n] + y_s[n]
        error = measured + y_s

        # Secondary path model filtering: x_f[n] = S_hat(z) * x[n]
        if len(self._model_state) > 1:
            self._model_state[1:] = self._model_state[:-1]
        self._model_state[0] = reference
        x_filtered = float(np.dot(self._secondary_model, self._model_state))

        # Filtered reference delay line
        if cfg.filter_length > 1:
            self._filtered_ref_state[1:] = self._filtered_ref_state[:-1]
        self._filtered_ref_state[0] = x_filtered

        # Adaptation
        if cfg.algorithm == "none":
            pass
        elif cfg.algorithm == "lms":
            self._coefficients -= (
                cfg.step_size * error * self._controller_state
            )
        elif cfg.algorithm == "nlms":
            norm = cfg.epsilon + float(
                np.dot(self._controller_state, self._controller_state)
            )
            self._coefficients -= (
                (cfg.step_size / norm) * error * self._controller_state
            )
        elif cfg.algorithm == "fxlms":
            self._coefficients -= (
                cfg.step_size * error * self._filtered_ref_state
            )
        elif cfg.algorithm == "fxnlms":
            norm = cfg.epsilon + float(
                np.dot(self._filtered_ref_state, self._filtered_ref_state)
            )
            self._coefficients -= (
                (cfg.step_size / norm) * error * self._filtered_ref_state
            )

        return error

    def process_frame(
        self,
        reference_frame: np.ndarray,
        measured_frame: np.ndarray,
    ) -> np.ndarray:
        """Process a frame of samples through the ANC filter.

        Parameters
        ----------
        reference_frame : np.ndarray
            Reference microphone frame.
        measured_frame : np.ndarray
            Error/measured microphone frame.

        Returns
        -------
        np.ndarray
            Residual error frame.
        """
        ref = np.asarray(reference_frame, dtype=np.float64).ravel()
        meas = np.asarray(measured_frame, dtype=np.float64).ravel()
        if len(ref) != len(meas):
            raise ValueError("reference and measured frames must have equal lengths.")

        residual = np.empty(len(ref), dtype=np.float64)
        for i in range(len(ref)):
            residual[i] = self.process_sample(float(ref[i]), float(meas[i]))

        return residual

    def reset(self) -> None:
        """Reset all filter state (coefficients and delay lines)."""
        self._coefficients[:] = 0.0
        self._controller_state[:] = 0.0
        self._secondary_state[:] = 0.0
        self._model_state[:] = 0.0
        self._filtered_ref_state[:] = 0.0
