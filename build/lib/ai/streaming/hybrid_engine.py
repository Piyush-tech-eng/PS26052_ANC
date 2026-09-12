"""Hybrid ANC + AI streaming engine with per-stage latency instrumentation.

Wires the complete cascade:

1. Chunked audio in (reference + measured microphone frames)
2. ``FrameANC`` — classical FxNLMS adaptive cancellation
3. ``OverlapAddProcessor`` wrapping any ``EnhancementModel`` — AI enhancement
4. Enhanced audio out

The engine exposes explicit per-stage timing (ANC, AI inference, total) so
the hardware integration layer can display latency on the demo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ai.models.base import EnhancementModel
from ai.streaming.frame_anc import FrameANC, FrameANCConfig
from ai.streaming.overlap_add import OverlapAddProcessor


@dataclass
class StageTimings:
    """Per-stage latency measurements for one processing call."""
    anc_seconds: float = 0.0
    ai_seconds: float = 0.0
    total_seconds: float = 0.0
    frame_samples: int = 0
    sample_rate: int = 16_000

    @property
    def anc_ms(self) -> float:
        return self.anc_seconds * 1000.0

    @property
    def ai_ms(self) -> float:
        return self.ai_seconds * 1000.0

    @property
    def total_ms(self) -> float:
        return self.total_seconds * 1000.0

    @property
    def realtime_ratio(self) -> float:
        """Processing time / audio duration. < 1.0 means real-time capable."""
        audio_duration = self.frame_samples / self.sample_rate if self.sample_rate > 0 else 1.0
        return self.total_seconds / audio_duration if audio_duration > 0 else float("inf")


class HybridEngine:
    """Cascaded ANC → AI enhancement engine for real-time processing.

    Parameters
    ----------
    model : EnhancementModel
        AI speech enhancement backend.
    anc_config : FrameANCConfig, optional
        ANC filter configuration. If None, ANC stage is skipped
        (AI-only mode).
    secondary_path_true : np.ndarray, optional
        True secondary path for ANC. Required if anc_config is provided.
    secondary_path_model : np.ndarray, optional
        Modelled secondary path for ANC. Required if anc_config is provided.
    sample_rate : int
        Audio sample rate.
    """

    def __init__(
        self,
        model: EnhancementModel,
        anc_config: FrameANCConfig | None = None,
        secondary_path_true: np.ndarray | None = None,
        secondary_path_model: np.ndarray | None = None,
        sample_rate: int = 16_000,
    ) -> None:
        if not isinstance(model, EnhancementModel):
            raise TypeError("model must be an EnhancementModel instance.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        self._model = model
        self._sample_rate = sample_rate
        self._overlap_add = OverlapAddProcessor(model)

        # ANC stage (optional)
        self._anc: FrameANC | None = None
        if anc_config is not None:
            if secondary_path_true is None or secondary_path_model is None:
                raise ValueError(
                    "secondary_path_true and secondary_path_model are required "
                    "when anc_config is provided."
                )
            self._anc = FrameANC(
                anc_config,
                secondary_path_true,
                secondary_path_model,
            )

        # Rolling timing history
        self._timing_history: list[StageTimings] = []
        self._max_history = 100

    def negotiate_chunk_size(
        self,
        current_chunk_size: int,
        *,
        min_samples: int = 160,
        max_samples: int = 1600,
        target_headroom: float = 0.5,
        step_samples: int = 80,
    ) -> int:
        """Dynamically adapt chunk size based on measured processing headroom (Phase D.2).

        Trades latency vs throughput:
        - If processing ratio > 0.80 (approaching underrun), increases chunk size
          to reduce per-frame overhead.
        - If processing ratio < 0.35 (abundant headroom), decreases chunk size
          to reduce latency.
        - Otherwise maintains current chunk size.

        Parameters
        ----------
        current_chunk_size : int
            Current frame size in samples.
        min_samples : int
            Lower bound on chunk size (latency floor).
        max_samples : int
            Upper bound on chunk size (latency ceiling).
        target_headroom : float
            Desired processing-time-to-audio-duration ratio (default 0.5 = 50% load).
        step_samples : int
            Granularity for adjustments.

        Returns
        -------
        int
            Recommended new chunk size in samples.
        """
        if not self._timing_history:
            return current_chunk_size

        recent_ratios = [
            t.realtime_ratio
            for t in self._timing_history[-10:]
            if t.realtime_ratio < float("inf")
        ]
        if not recent_ratios:
            return current_chunk_size

        avg_ratio = sum(recent_ratios) / len(recent_ratios)

        # High CPU pressure: back off by increasing chunk size
        if avg_ratio > 0.80:
            return min(max_samples, current_chunk_size + step_samples)
        # Plentiful headroom: lower latency by decreasing chunk size
        elif avg_ratio < 0.35:
            return max(min_samples, current_chunk_size - step_samples)
        return current_chunk_size

    @property
    def has_anc(self) -> bool:
        """Whether the classical ANC stage is active."""
        return self._anc is not None

    @property
    def model_name(self) -> str:
        return self._model.name

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def convergence_indicator(self) -> float:
        """Estimated filter convergence, 0.0 (diverged) to 1.0 (converged).

        Uses the ratio of recent error energy reduction as a proxy for
        convergence.  Returns 0.0 if ANC is not active.
        """
        if self._anc is None:
            return 0.0
        # Use filter coefficient norm relative to a moving average as
        # a simple convergence heuristic: once the norm stabilises the
        # filter has converged.
        coeff_norm = float(np.linalg.norm(self._anc.coefficients))
        if not hasattr(self, '_prev_coeff_norms'):
            self._prev_coeff_norms: list[float] = []
        self._prev_coeff_norms.append(coeff_norm)
        if len(self._prev_coeff_norms) > 50:
            self._prev_coeff_norms.pop(0)
        if len(self._prev_coeff_norms) < 5:
            return 0.0
        recent = self._prev_coeff_norms[-10:]
        if max(recent) < 1e-10:
            return 0.0
        variation = (max(recent) - min(recent)) / (max(recent) + 1e-10)
        return float(np.clip(1.0 - variation, 0.0, 1.0))

    @property
    def estimated_attenuation_db(self) -> float:
        """Estimated noise attenuation in dB from recent processing.

        Compares input RMS to output RMS from the most recent frames.
        Returns 0.0 if no data available.
        """
        if not self._timing_history:
            return 0.0
        if not hasattr(self, '_recent_input_rms'):
            return 0.0
        if not hasattr(self, '_recent_output_rms'):
            return 0.0
        if self._recent_input_rms < 1e-10:
            return 0.0
        ratio = self._recent_output_rms / (self._recent_input_rms + 1e-10)
        return float(-20.0 * np.log10(max(ratio, 1e-10)))

    @property
    def latest_timing(self) -> StageTimings | None:
        """Most recent processing timing, or None if no frames processed."""
        return self._timing_history[-1] if self._timing_history else None

    @property
    def average_timing(self) -> StageTimings:
        """Rolling average of recent timings."""
        if not self._timing_history:
            return StageTimings()
        n = len(self._timing_history)
        return StageTimings(
            anc_seconds=sum(t.anc_seconds for t in self._timing_history) / n,
            ai_seconds=sum(t.ai_seconds for t in self._timing_history) / n,
            total_seconds=sum(t.total_seconds for t in self._timing_history) / n,
            frame_samples=self._timing_history[-1].frame_samples,
            sample_rate=self._sample_rate,
        )

    def process_frame(
        self,
        measured: np.ndarray,
        reference: np.ndarray | None = None,
    ) -> tuple[np.ndarray, StageTimings]:
        """Process one audio frame through the hybrid pipeline.

        Parameters
        ----------
        measured : np.ndarray
            Error/measured microphone frame (primary input).
        reference : np.ndarray, optional
            Reference microphone frame. Required if ANC is active.

        Returns
        -------
        tuple of (np.ndarray, StageTimings)
            Enhanced audio frame and per-stage timing.
        """
        measured = np.asarray(measured, dtype=np.float64).ravel()
        self._recent_input_rms = float(np.sqrt(np.mean(measured ** 2)))
        t_total_start = time.perf_counter()

        # Stage 1: Classical ANC (if active and reference channel provided)
        t_anc_start = time.perf_counter()
        if self._anc is not None and reference is not None:
            reference = np.asarray(reference, dtype=np.float64).ravel()
            if len(reference) != len(measured):
                raise ValueError("reference and measured must have equal lengths.")
            anc_output = self._anc.process_frame(reference, measured)
        else:
            anc_output = measured.copy()
        t_anc_end = time.perf_counter()

        # Stage 2: AI Enhancement
        t_ai_start = time.perf_counter()
        enhanced = self._overlap_add.process(anc_output)
        t_ai_end = time.perf_counter()

        t_total_end = time.perf_counter()

        timing = StageTimings(
            anc_seconds=t_anc_end - t_anc_start,
            ai_seconds=t_ai_end - t_ai_start,
            total_seconds=t_total_end - t_total_start,
            frame_samples=len(measured),
            sample_rate=self._sample_rate,
        )

        # Store timing
        self._timing_history.append(timing)
        if len(self._timing_history) > self._max_history:
            self._timing_history.pop(0)

        self._recent_output_rms = float(np.sqrt(np.mean(enhanced ** 2)))
        return enhanced, timing

    def process_batch(
        self,
        measured: np.ndarray,
        reference: np.ndarray | None = None,
    ) -> tuple[np.ndarray, StageTimings]:
        """Process a full audio signal (non-streaming, for evaluation).

        Equivalent to ``process_frame`` but semantically for offline use.
        """
        return self.process_frame(measured, reference)

    def reset(self) -> None:
        """Reset all internal state for a new stream."""
        if self._anc is not None:
            self._anc.reset()
        self._overlap_add.reset()
        self._timing_history.clear()
