"""Basic Module 2 input utilities.

M2.01 consumes the canonical Module 1 handoff without recreating or
altering its signals.  Statistical estimators are deliberately deferred to
the later Module 2 stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Module1Handoff:
    """Validated inputs passed from Module 1 to Module 2.

    Attributes
    ----------
    reference:
        The Module 1 reference signal ``x[n]``.
    desired:
        The Module 1 desired signal ``d[n]``.
    sampling_rate_hz:
        Sampling rate declared by the canonical handoff metadata.
    metadata:
        Complete, unmodified metadata read from ``metadata.json``.
    source_directory:
        Directory containing the canonical Module 1 artifacts.
    """

    reference: np.ndarray
    desired: np.ndarray
    sampling_rate_hz: int
    metadata: dict[str, Any]
    source_directory: Path

    @property
    def num_samples(self) -> int:
        """Number of samples in each validated input signal."""

        return len(self.reference)

    @property
    def duration_seconds(self) -> float:
        """Duration implied by the validated signal length and sample rate."""

        return self.num_samples / self.sampling_rate_hz


def load_module1_handoff(
    handoff_directory: str | Path,
) -> Module1Handoff:
    """Load and validate the frozen canonical Module 1 handoff.

    The loader reads only ``reference.npy``, ``desired.npy``, and
    ``metadata.json``.  The known FIR remains a separate validation input for
    M2.09, so it cannot leak into the batch Wiener estimate in this stage.

    Parameters
    ----------
    handoff_directory:
        Directory containing the canonical Module 1 handoff artifacts.

    Returns
    -------
    Module1Handoff
        Validated reference and desired signals plus their metadata.

    Raises
    ------
    FileNotFoundError
        If a required handoff artifact is absent.
    ValueError
        If an artifact is malformed or violates the Module 1-to-Module 2
        input contract.
    """

    source_directory = Path(handoff_directory)

    if not source_directory.is_dir():
        raise FileNotFoundError(
            "Module 1 handoff directory does not exist: "
            f"{source_directory}"
        )

    metadata = _load_metadata(
        _required_file(source_directory, "metadata.json")
    )

    reference = _load_signal_array(
        _required_file(source_directory, "reference.npy"),
        signal_name="reference",
    )

    desired = _load_signal_array(
        _required_file(source_directory, "desired.npy"),
        signal_name="desired",
    )

    if len(reference) != len(desired):
        raise ValueError(
            "Module 1 reference and desired signals must have equal "
            "lengths."
        )

    sampling_rate_hz = _sampling_rate_from_metadata(metadata)
    _validate_metadata_sample_count(
        metadata,
        actual_num_samples=len(reference),
    )

    return Module1Handoff(
        reference=reference,
        desired=desired,
        sampling_rate_hz=sampling_rate_hz,
        metadata=metadata,
        source_directory=source_directory,
    )


def build_m2_input_summary(
    handoff: Module1Handoff,
) -> dict[str, Any]:
    """Create the compact, JSON-serializable M2.01 input summary."""

    return {
        "module": "M2.01",
        "source_dataset": handoff.metadata.get("dataset"),
        "sampling_rate_hz": handoff.sampling_rate_hz,
        "num_samples": handoff.num_samples,
        "duration_seconds": handoff.duration_seconds,
        "reference": {
            "symbol": "x[n]",
            "file": "reference.npy",
            "shape": list(handoff.reference.shape),
            "dtype": str(handoff.reference.dtype),
        },
        "desired": {
            "symbol": "d[n]",
            "file": "desired.npy",
            "shape": list(handoff.desired.shape),
            "dtype": str(handoff.desired.dtype),
        },
        "validation": {
            "reference_and_desired_same_length": True,
            "reference_samples_finite": True,
            "desired_samples_finite": True,
            "sampling_rate_valid": True,
            "metadata_num_samples_matches": True,
        },
    }


def save_m2_input_summary(
    handoff: Module1Handoff,
    output_path: str | Path,
) -> Path:
    """Write a compact M2.01 input summary outside the frozen handoff."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as file:
        json.dump(
            build_m2_input_summary(handoff),
            file,
            indent=2,
        )
        file.write("\n")

    return destination


def _required_file(
    directory: Path,
    filename: str,
) -> Path:
    """Return a required artifact path or raise a clear contract error."""

    path = directory / filename

    if not path.is_file():
        raise FileNotFoundError(
            "Module 1 handoff is missing required artifact: "
            f"{path}"
        )

    return path


def _load_metadata(
    metadata_path: Path,
) -> dict[str, Any]:
    """Load the JSON metadata object without modifying it."""

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Module 1 metadata.json is not valid JSON."
        ) from error

    if not isinstance(metadata, dict):
        raise ValueError(
            "Module 1 metadata.json must contain a JSON object."
        )

    return metadata


def _load_signal_array(
    signal_path: Path,
    *,
    signal_name: str,
) -> np.ndarray:
    """Load a finite, real-valued, one-dimensional signal array."""

    try:
        samples = np.load(signal_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Could not load Module 1 {signal_name} signal: "
            f"{signal_path}"
        ) from error

    if samples.ndim != 1:
        raise ValueError(
            f"Module 1 {signal_name} signal must be one-dimensional."
        )

    if len(samples) == 0:
        raise ValueError(
            f"Module 1 {signal_name} signal cannot be empty."
        )

    if (
        not np.issubdtype(samples.dtype, np.number)
        or np.iscomplexobj(samples)
    ):
        raise ValueError(
            f"Module 1 {signal_name} signal must contain real numbers."
        )

    samples = np.asarray(samples, dtype=np.float64)

    if not np.isfinite(samples).all():
        raise ValueError(
            f"Module 1 {signal_name} signal contains NaN or Inf."
        )

    return samples


def _sampling_rate_from_metadata(
    metadata: dict[str, Any],
) -> int:
    """Return the positive integer sampling rate stored in metadata."""

    if "sampling_rate_hz" not in metadata:
        raise ValueError(
            "Module 1 metadata is missing sampling_rate_hz."
        )

    value = metadata["sampling_rate_hz"]

    if isinstance(value, bool):
        raise ValueError(
            "Module 1 sampling_rate_hz must be a positive integer."
        )

    try:
        sampling_rate = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Module 1 sampling_rate_hz must be a positive integer."
        ) from error

    if (
        not np.isfinite(sampling_rate)
        or sampling_rate <= 0
        or not sampling_rate.is_integer()
    ):
        raise ValueError(
            "Module 1 sampling_rate_hz must be a positive integer."
        )

    return int(sampling_rate)


def _validate_metadata_sample_count(
    metadata: dict[str, Any],
    *,
    actual_num_samples: int,
) -> None:
    """Ensure the canonical metadata sample count agrees with the arrays."""

    if "num_samples" not in metadata:
        raise ValueError(
            "Module 1 metadata is missing num_samples."
        )

    value = metadata["num_samples"]

    if isinstance(value, bool):
        raise ValueError(
            "Module 1 metadata num_samples must be an integer."
        )

    try:
        num_samples = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Module 1 metadata num_samples must be an integer."
        ) from error

    if (
        not np.isfinite(num_samples)
        or not num_samples.is_integer()
        or int(num_samples) != actual_num_samples
    ):
        raise ValueError(
            "Module 1 metadata num_samples does not match the loaded "
            "signals."
        )

def mean(signal: np.ndarray) -> float:
    """Return the arithmetic mean of a real-valued signal."""
    samples = np.asarray(signal, dtype=np.float64)

    if samples.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(samples) == 0:
        raise ValueError("Signal must not be empty.")

    if not np.isfinite(samples).all():
        raise ValueError("Signal contains NaN or Inf.")

    return float(np.mean(samples))


def average_power(signal: np.ndarray) -> float:
    """Return finite-signal average power."""

    samples = np.asarray(signal, dtype=np.float64)

    if samples.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(samples) == 0:
        raise ValueError("Signal must not be empty.")

    if not np.isfinite(samples).all():
        raise ValueError("Signal contains NaN or Inf.")

    return float(np.mean(samples ** 2))


def rms(signal: np.ndarray) -> float:
    """Return the root-mean-square value."""

    return float(np.sqrt(average_power(signal)))


def variance(signal: np.ndarray) -> float:
    """Return the population variance of a real-valued signal."""

    samples = np.asarray(signal, dtype=np.float64)

    if samples.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(samples) == 0:
        raise ValueError("Signal must not be empty.")

    if not np.isfinite(samples).all():
        raise ValueError("Signal contains NaN or Inf.")

    return float(np.var(samples))


def summarize_signal_statistics(
    signal: np.ndarray,
) -> dict[str, float]:
    """
    Compute the complete Module 2.02 statistical summary.
    """

    mu = mean(signal)
    power = average_power(signal)
    rms_value = rms(signal)
    var = variance(signal)

    return {
        "mean": mu,
        "variance": var,
        "rms": rms_value,
        "average_power": power,
    }