"""Common acoustic recording and ANC scenario data contracts.

Module 6 introduces a source-independent boundary between acoustic data
acquisition/loading and ANC processing.

Synthetic signals and recorded signals are represented using the same
AcousticRecording contract.  ANCScenario then combines compatible reference
and error-domain signals with explicit provenance and optional path metadata.

The controller itself must not depend on whether the original source was
synthetic or recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import wave

import numpy as np

from anc.secondary_path import SecondaryPathModel


_SOURCE_TYPES = frozenset({"synthetic", "recorded"})


def _finite_vector(
    values: np.ndarray | list[float],
    *,
    name: str,
) -> np.ndarray:
    """Validate and return a finite one-dimensional float64 vector."""

    vector = np.asarray(values, dtype=np.float64)

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


def _positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    """Validate a positive integer."""

    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, bool)
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return int(value)


def _json_safe_metadata(
    metadata: dict[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    """Return a shallow metadata copy.

    Full JSON serialization is intentionally deferred to artifact-writing
    code.  This contract preserves metadata without mutating caller-owned
    dictionaries.
    """

    if not isinstance(metadata, dict):
        raise TypeError(
            f"{name} must be a dictionary."
        )

    return dict(metadata)


@dataclass(frozen=True)
class AcousticRecording:
    """One source-independent acoustic signal.

    Internally, samples are always represented as a finite one-dimensional
    float64 vector.

    ``source_type`` records whether the signal originated from a controlled
    synthetic generator or from a recorded/measured source.
    """

    samples: np.ndarray
    sampling_rate_hz: int
    source_type: str
    provenance: dict[str, Any] = field(
        default_factory=dict,
    )
    channel_metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "samples",
            _finite_vector(
                self.samples,
                name="samples",
            ),
        )

        _positive_integer(
            self.sampling_rate_hz,
            name="sampling_rate_hz",
        )

        if self.source_type not in _SOURCE_TYPES:
            supported = ", ".join(
                sorted(_SOURCE_TYPES),
            )

            raise ValueError(
                "source_type must be one of: "
                f"{supported}."
            )

        object.__setattr__(
            self,
            "provenance",
            _json_safe_metadata(
                self.provenance,
                name="provenance",
            ),
        )

        object.__setattr__(
            self,
            "channel_metadata",
            _json_safe_metadata(
                self.channel_metadata,
                name="channel_metadata",
            ),
        )

    @property
    def num_samples(self) -> int:
        """Number of samples."""

        return int(len(self.samples))

    @property
    def duration_seconds(self) -> float:
        """Signal duration in seconds."""

        return float(
            self.num_samples
            / self.sampling_rate_hz
        )


@dataclass(frozen=True)
class ANCScenario:
    """Complete input contract for one Module 6 ANC scenario.

    ``reference`` is required.

    ``error_input`` represents the available disturbance/error-domain signal
    supplied to the replay layer.

    A true secondary path is optional because real recorded measurements may
    not have access to hidden physical ground truth.

    The identified secondary-path model is required because Module 6 should
    consume the portable handoff produced by Module 5.
    """

    scenario_id: str

    reference: AcousticRecording

    error_input: AcousticRecording

    secondary_path_model: SecondaryPathModel

    true_secondary_path: np.ndarray | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.scenario_id,
            str,
        ):
            raise TypeError(
                "scenario_id must be a string."
            )

        if not self.scenario_id.strip():
            raise ValueError(
                "scenario_id must not be empty."
            )

        if not isinstance(
            self.reference,
            AcousticRecording,
        ):
            raise TypeError(
                "reference must be an AcousticRecording."
            )

        if not isinstance(
            self.error_input,
            AcousticRecording,
        ):
            raise TypeError(
                "error_input must be an AcousticRecording."
            )

        if (
            self.reference.sampling_rate_hz
            != self.error_input.sampling_rate_hz
        ):
            raise ValueError(
                "reference and error_input must have "
                "the same sampling rate."
            )

        if (
            len(self.reference.samples)
            != len(self.error_input.samples)
        ):
            raise ValueError(
                "reference and error_input must have "
                "the same number of samples."
            )

        if not isinstance(
            self.secondary_path_model,
            SecondaryPathModel,
        ):
            raise TypeError(
                "secondary_path_model must be a "
                "SecondaryPathModel."
            )

        if (
            self.secondary_path_model.sampling_rate_hz
            != self.reference.sampling_rate_hz
        ):
            raise ValueError(
                "secondary_path_model sampling rate "
                "must match the scenario sampling rate."
            )

        if self.true_secondary_path is not None:
            object.__setattr__(
                self,
                "true_secondary_path",
                _finite_vector(
                    self.true_secondary_path,
                    name="true_secondary_path",
                ),
            )

        object.__setattr__(
            self,
            "metadata",
            _json_safe_metadata(
                self.metadata,
                name="metadata",
            ),
        )

    @property
    def sampling_rate_hz(self) -> int:
        """Common scenario sampling rate."""

        return self.reference.sampling_rate_hz

    @property
    def num_samples(self) -> int:
        """Common scenario signal length."""

        return self.reference.num_samples

    @property
    def duration_seconds(self) -> float:
        """Common scenario duration."""

        return self.reference.duration_seconds

    @property
    def source_types(self) -> dict[str, str]:
        """Return provenance-friendly source types."""

        return {
            "reference": self.reference.source_type,
            "error_input": self.error_input.source_type,
        }


def recording_from_array(
    samples: np.ndarray | list[float],
    *,
    sampling_rate_hz: int,
    source_type: str,
    provenance: dict[str, Any] | None = None,
    channel_metadata: dict[str, Any] | None = None,
) -> AcousticRecording:
    """Create an AcousticRecording from an in-memory mono array."""

    return AcousticRecording(
        samples=np.asarray(
            samples,
            dtype=np.float64,
        ),
        sampling_rate_hz=sampling_rate_hz,
        source_type=source_type,
        provenance=(
            provenance
            if provenance is not None
            else {}
        ),
        channel_metadata=(
            channel_metadata
            if channel_metadata is not None
            else {
                "original_channels": 1,
                "selected_channel": 0,
            }
        ),
    )


def load_npy_recording(
    path: str | Path,
    *,
    sampling_rate_hz: int,
    source_type: str = "recorded",
    provenance: dict[str, Any] | None = None,
) -> AcousticRecording:
    """Load a mono one-dimensional recording from a .npy file."""

    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(
            f"Recording not found: {source}"
        )

    if source.suffix.lower() != ".npy":
        raise ValueError(
            "load_npy_recording requires a .npy file."
        )

    samples = np.load(
        source,
        allow_pickle=False,
    )

    metadata = {
        "loader": "npy",
        "path": str(source),
    }

    if provenance is not None:
        metadata.update(provenance)

    return recording_from_array(
        samples,
        sampling_rate_hz=sampling_rate_hz,
        source_type=source_type,
        provenance=metadata,
    )


def _decode_pcm_24bit(
    raw_bytes: bytes,
) -> np.ndarray:
    """Decode little-endian signed 24-bit PCM."""

    raw = np.frombuffer(
        raw_bytes,
        dtype=np.uint8,
    )

    if len(raw) % 3 != 0:
        raise ValueError(
            "Invalid 24-bit PCM byte count."
        )

    triples = raw.reshape(
        -1,
        3,
    ).astype(np.int32)

    values = (
        triples[:, 0]
        | (triples[:, 1] << 8)
        | (triples[:, 2] << 16)
    )

    negative = values & 0x800000

    values = values - (
        negative != 0
    ) * 0x1000000

    return values.astype(
        np.float64,
    ) / float(2**23)


def _decode_pcm_samples(
    raw_bytes: bytes,
    *,
    sample_width_bytes: int,
) -> np.ndarray:
    """Decode standard PCM WAV sample widths into float64."""

    if sample_width_bytes == 1:
        values = np.frombuffer(
            raw_bytes,
            dtype=np.uint8,
        ).astype(np.float64)

        return (
            values - 128.0
        ) / 128.0

    if sample_width_bytes == 2:
        values = np.frombuffer(
            raw_bytes,
            dtype="<i2",
        ).astype(np.float64)

        return values / float(
            2**15,
        )

    if sample_width_bytes == 3:
        return _decode_pcm_24bit(
            raw_bytes,
        )

    if sample_width_bytes == 4:
        values = np.frombuffer(
            raw_bytes,
            dtype="<i4",
        ).astype(np.float64)

        return values / float(
            2**31,
        )

    raise ValueError(
        "Unsupported WAV PCM sample width: "
        f"{sample_width_bytes} bytes."
    )


def load_wav_recording(
    path: str | Path,
    *,
    channel: int = 0,
    source_type: str = "recorded",
    provenance: dict[str, Any] | None = None,
) -> AcousticRecording:
    """Load one channel from a standard PCM WAV recording.

    The loader intentionally does not silently resample or average channels.
    Channel selection is explicit and recorded in metadata.
    """

    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(
            f"Recording not found: {source}"
        )

    if source.suffix.lower() != ".wav":
        raise ValueError(
            "load_wav_recording requires a .wav file."
        )

    if (
        not isinstance(
            channel,
            (int, np.integer),
        )
        or isinstance(
            channel,
            bool,
        )
    ):
        raise TypeError(
            "channel must be an integer."
        )

    with wave.open(
        str(source),
        "rb",
    ) as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sampling_rate_hz = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        raw_bytes = wav_file.readframes(
            num_frames,
        )

    if channel < 0 or channel >= num_channels:
        raise ValueError(
            "channel is outside the available "
            f"range [0, {num_channels - 1}]."
        )

    decoded = _decode_pcm_samples(
        raw_bytes,
        sample_width_bytes=sample_width,
    )

    expected_samples = (
        num_frames * num_channels
    )

    if len(decoded) != expected_samples:
        raise ValueError(
            "Decoded WAV sample count does not "
            "match WAV metadata."
        )

    channels = decoded.reshape(
        num_frames,
        num_channels,
    )

    selected = channels[:, channel]

    metadata = {
        "loader": "wave",
        "path": str(source),
    }

    if provenance is not None:
        metadata.update(provenance)

    return AcousticRecording(
        samples=selected,
        sampling_rate_hz=sampling_rate_hz,
        source_type=source_type,
        provenance=metadata,
        channel_metadata={
            "original_channels": num_channels,
            "selected_channel": int(channel),
            "sample_width_bytes": sample_width,
        },
    )