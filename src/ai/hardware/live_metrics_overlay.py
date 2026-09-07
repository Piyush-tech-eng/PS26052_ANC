"""Rolling live metrics overlay for the real-time demo.

Computes and displays rolling SNR estimate on the live stream during the
demo — high judge-visibility, low implementation cost given Phase 4's
metrics already exist.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from anc.evaluation.metrics import compute_si_snr, compute_signal_power


class LiveMetricsOverlay:
    """Rolling real-time metrics display for the demo.

    Parameters
    ----------
    window_seconds : float
        Duration of the rolling window for metric computation.
    sample_rate : int
        Audio sample rate.
    update_interval : float
        Minimum seconds between display updates.
    """

    def __init__(
        self,
        window_seconds: float = 2.0,
        sample_rate: int = 16_000,
        update_interval: float = 0.5,
    ) -> None:
        self._window_samples = int(window_seconds * sample_rate)
        self._sample_rate = sample_rate
        self._update_interval = update_interval

        # Rolling buffers
        self._input_buffer = deque(maxlen=self._window_samples)
        self._output_buffer = deque(maxlen=self._window_samples)

        # Metrics history
        self._snr_history: deque[float] = deque(maxlen=100)
        self._si_snr_history: deque[float] = deque(maxlen=100)
        self._power_reduction_history: deque[float] = deque(maxlen=100)

        self._last_update = 0.0
        self._callback = None

    def set_display_callback(self, callback) -> None:
        """Set a callback function for display updates.

        The callback receives a dict with current metrics.
        """
        self._callback = callback

    def feed(
        self,
        input_audio: np.ndarray,
        output_audio: np.ndarray,
    ) -> dict[str, float] | None:
        """Feed input/output audio pair and optionally compute metrics.

        Parameters
        ----------
        input_audio : np.ndarray
            Noisy/raw input chunk.
        output_audio : np.ndarray
            Enhanced output chunk (same length).

        Returns
        -------
        dict or None
            Metrics dict if enough data accumulated, else None.
        """
        inp = np.asarray(input_audio, dtype=np.float64).ravel()
        out = np.asarray(output_audio, dtype=np.float64).ravel()

        self._input_buffer.extend(inp)
        self._output_buffer.extend(out)

        now = time.time()
        if now - self._last_update < self._update_interval:
            return None

        if len(self._input_buffer) < self._sample_rate // 4:
            return None  # Need at least 250ms

        self._last_update = now
        return self._compute_metrics()

    def _compute_metrics(self) -> dict[str, float]:
        """Compute current rolling metrics."""
        inp = np.array(self._input_buffer, dtype=np.float64)
        out = np.array(self._output_buffer, dtype=np.float64)

        floor = np.finfo(np.float64).eps

        # Input/output power
        input_power = compute_signal_power(inp)
        output_power = compute_signal_power(out)

        # Power reduction in dB
        power_reduction_db = float(
            10.0 * np.log10((input_power + floor) / (output_power + floor))
        )

        # Simple SNR estimate (assuming noise = input - output)
        noise_estimate = inp[:len(out)] - out[:len(inp)]
        noise_power = float(np.mean(noise_estimate ** 2))
        snr_db = float(
            10.0 * np.log10((output_power + floor) / (noise_power + floor))
        )

        metrics = {
            "input_power_db": float(10.0 * np.log10(input_power + floor)),
            "output_power_db": float(10.0 * np.log10(output_power + floor)),
            "power_reduction_db": power_reduction_db,
            "estimated_snr_db": snr_db,
            "buffer_samples": len(self._input_buffer),
        }

        # Store history
        self._snr_history.append(snr_db)
        self._power_reduction_history.append(power_reduction_db)

        # Fire callback if set
        if self._callback is not None:
            self._callback(metrics)

        return metrics

    def format_display(self, metrics: dict[str, float] | None = None) -> str:
        """Format metrics for terminal display.

        Parameters
        ----------
        metrics : dict, optional
            Metrics dict from ``feed()``.  If None, uses latest computed.

        Returns
        -------
        str
            Formatted display string.
        """
        if metrics is None:
            if not self._snr_history:
                return "⏳ Collecting audio..."
            metrics = {
                "estimated_snr_db": self._snr_history[-1],
                "power_reduction_db": self._power_reduction_history[-1],
                "input_power_db": 0.0,
                "output_power_db": 0.0,
            }

        snr = metrics.get("estimated_snr_db", 0.0)
        reduction = metrics.get("power_reduction_db", 0.0)

        # Visual bar for SNR
        bar_width = 30
        snr_clamped = max(0, min(30, snr))
        filled = int(snr_clamped / 30 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        lines = [
            "┌─────────────────────────────────────────┐",
            "│         LIVE ANC+AI METRICS              │",
            "├─────────────────────────────────────────┤",
            f"│  SNR:  {snr:+6.1f} dB  [{bar}] │",
            f"│  Noise Reduction: {reduction:+6.1f} dB              │",
            "└─────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all buffers and history."""
        self._input_buffer.clear()
        self._output_buffer.clear()
        self._snr_history.clear()
        self._si_snr_history.clear()
        self._power_reduction_history.clear()
