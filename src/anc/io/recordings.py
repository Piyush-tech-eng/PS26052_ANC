"""Recorded and measured signal input utilities.

Module 6.03 provides a common representation for signals entering the
offline ANC replay pipeline.

The ANC controller itself should not need to know whether a signal came
from:

- a NumPy artifact,
- a WAV recording,
- an in-memory measurement,
- or a future live-audio adapter.

This module converts those sources into validated SignalRecording
objects with explicit sampling-rate and channel metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.io import wavfile


SignalRole = Literal[
    "reference",
    "measured",
]


def _validate_sampling_rate(
    sampling_rate_hz: int,
) -> int:
    """Validate a positive integer sampling rate."""

    if (
        not isinstance(
            sampling_rate_hz,
            (int, np.integer),
        )
        or isinstance(
            sampling_rate_hz,
            bool,
        )
    ):
        raise TypeError(
            "sampling_rate_hz must be an integer."
        )

    if sampling_rate_hz <= 0:
        raise ValueError(
            "sampling_rate_hz must be positive."
        )

    return int(
        sampling_rate_hz
    )


def _to_float64(
    samples: np.ndarray,
) -> np.ndarray:
    """Convert supported audio arrays to normalized float64."""

    array = np.asarray(
        samples
    )

    if array.size == 0:
        raise ValueError(
            "samples must not be empty."
        )

    if np.issubdtype(
        array.dtype,
        np.floating,
    ):
        output = array.astype(
            np.float64,
            copy=True,
        )

    elif np.issubdtype(
        array.dtype,
        np.signedinteger,
    ):
        info = np.iinfo(
            array.dtype
        )

        scale = float(
            max(
                abs(
                    info.min
                ),
                info.max,
            )
        )

        output = (
            array.astype(
                np.float64
            )
            / scale
        )

    elif np.issubdtype(
        array.dtype,
        np.unsignedinteger,
    ):
        info = np.iinfo(
            array.dtype
        )

        midpoint = (
            float(
                info.max
            )
            / 2.0
        )

        output = (
            array.astype(
                np.float64
            )
            - midpoint
        ) / midpoint

    else:
        raise TypeError(
            "samples must have an integer "
            "or floating dtype."
        )

    if not np.isfinite(
        output
    ).all():
        raise ValueError(
            "samples contain NaN or Inf."
        )

    return output


def _select_channel(
    samples: np.ndarray,
    *,
    channel: int | None,
) -> tuple[np.ndarray, int, int]:
    """Select one channel and return signal metadata.

    Returns:

        selected_samples,
        selected_channel,
        total_channels
    """

    array = np.asarray(
        samples
    )

    if array.ndim == 1:

        if channel not in (
            None,
            0,
        ):
            raise ValueError(
                "Mono recording only has channel 0."
            )

        return (
            array.copy(),
            0,
            1,
        )

    if array.ndim != 2:
        raise ValueError(
            "Recording samples must be "
            "one-dimensional or two-dimensional."
        )

    total_channels = int(
        array.shape[1]
    )

    if total_channels == 0:
        raise ValueError(
            "Recording must contain at least "
            "one channel."
        )

    selected_channel = (
        0
        if channel is None
        else channel
    )

    if (
        not isinstance(
            selected_channel,
            (int, np.integer),
        )
        or isinstance(
            selected_channel,
            bool,
        )
    ):
        raise TypeError(
            "channel must be an integer or None."
        )

    selected_channel = int(
        selected_channel
    )

    if not (
        0
        <= selected_channel
        < total_channels
    ):
        raise ValueError(
            "channel is outside the available "
            "channel range."
        )

    return (
        array[
            :,
            selected_channel
        ].copy(),
        selected_channel,
        total_channels,
    )


@dataclass(frozen=True)
class SignalRecording:
    """Validated mono signal plus explicit recording metadata."""

    samples: np.ndarray

    sampling_rate_hz: int

    role: SignalRole

    source: str

    selected_channel: int

    total_channels: int

    def __post_init__(
        self,
    ) -> None:

        samples = np.asarray(
            self.samples,
            dtype=np.float64,
        )

        if samples.ndim != 1:
            raise ValueError(
                "SignalRecording.samples must "
                "be one-dimensional."
            )

        if len(samples) == 0:
            raise ValueError(
                "SignalRecording.samples must "
                "not be empty."
            )

        if not np.isfinite(
            samples
        ).all():
            raise ValueError(
                "SignalRecording.samples contain "
                "NaN or Inf."
            )

        _validate_sampling_rate(
            self.sampling_rate_hz
        )

        if self.role not in (
            "reference",
            "measured",
        ):
            raise ValueError(
                "role must be 'reference' "
                "or 'measured'."
            )

        if self.total_channels <= 0:
            raise ValueError(
                "total_channels must be positive."
            )

        if not (
            0
            <= self.selected_channel
            < self.total_channels
        ):
            raise ValueError(
                "selected_channel is invalid."
            )

        object.__setattr__(
            self,
            "samples",
            samples.copy(),
        )

    @property
    def num_samples(
        self,
    ) -> int:
        return int(
            len(
                self.samples
            )
        )

    @property
    def duration_seconds(
        self,
    ) -> float:
        return (
            self.num_samples
            / self.sampling_rate_hz
        )


@dataclass(frozen=True)
class ANCReplayInputs:
    """Validated pair of signals for offline ANC replay."""

    reference: SignalRecording

    measured: SignalRecording

    alignment_applied: bool

    delay_samples: int | None

    overlap_samples: int

    def __post_init__(
        self,
    ) -> None:

        if (
            self.reference.sampling_rate_hz
            != self.measured.sampling_rate_hz
        ):
            raise ValueError(
                "reference and measured recordings "
                "must have the same sampling rate."
            )

        if (
            self.reference.num_samples
            != self.measured.num_samples
        ):
            raise ValueError(
                "reference and measured recordings "
                "must have equal lengths."
            )

        if self.overlap_samples <= 0:
            raise ValueError(
                "overlap_samples must be positive."
            )


def recording_from_array(
    samples: np.ndarray,
    *,
    sampling_rate_hz: int,
    role: SignalRole,
    source: str = "in_memory",
    channel: int | None = None,
) -> SignalRecording:
    """Create a validated recording from an in-memory array."""

    selected, selected_channel, total_channels = (
        _select_channel(
            np.asarray(
                samples
            ),
            channel=channel,
        )
    )

    normalized = _to_float64(
        selected
    )

    return SignalRecording(
        samples=normalized,
        sampling_rate_hz=(
            _validate_sampling_rate(
                sampling_rate_hz
            )
        ),
        role=role,
        source=source,
        selected_channel=(
            selected_channel
        ),
        total_channels=(
            total_channels
        ),
    )


def load_wav_recording(
    path: str | Path,
    *,
    role: SignalRole,
    channel: int | None = None,
) -> SignalRecording:
    """Load a WAV recording."""

    file_path = Path(
        path
    )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"WAV file does not exist: "
            f"{file_path}"
        )

    sampling_rate_hz, samples = (
        wavfile.read(
            file_path
        )
    )

    return recording_from_array(
        samples,
        sampling_rate_hz=(
            int(
                sampling_rate_hz
            )
        ),
        role=role,
        source=str(
            file_path
        ),
        channel=channel,
    )


def load_npy_recording(
    path: str | Path,
    *,
    sampling_rate_hz: int,
    role: SignalRole,
    channel: int | None = None,
) -> SignalRecording:
    """Load a NumPy recording.

    A .npy file does not intrinsically store sampling-rate metadata, so
    sampling_rate_hz must be supplied explicitly.
    """

    file_path = Path(
        path
    )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"NumPy file does not exist: "
            f"{file_path}"
        )

    samples = np.load(
        file_path,
        allow_pickle=False,
    )

    return recording_from_array(
        samples,
        sampling_rate_hz=(
            sampling_rate_hz
        ),
        role=role,
        source=str(
            file_path
        ),
        channel=channel,
    )