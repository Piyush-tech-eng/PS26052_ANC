"""Secondary-path identification and portable model serialization.

Identification deliberately receives only an excitation and the measured
response.  A true path may be passed to the separate validation function, but
never to the estimator itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from anc.adaptive import LMSFilter, NLMSFilter, moving_average


def _finite_vector(values: np.ndarray | list[float], *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(vector) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return vector.copy()


def _positive_integer(value: int, *, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


@dataclass(frozen=True)
class IdentificationConfig:
    """Configuration for a measured-input/output secondary-path estimate."""

    filter_length: int
    step_size: float
    algorithm: str = "nlms"
    epsilon: float = 1e-8
    sampling_rate_hz: int = 8_000
    learning_window: int = 256
    experiment_seed: int | None = None

    def __post_init__(self) -> None:
        _positive_integer(self.filter_length, name="filter_length")
        _positive_integer(self.sampling_rate_hz, name="sampling_rate_hz")
        _positive_integer(self.learning_window, name="learning_window")
        if self.algorithm not in {"lms", "nlms"}:
            raise ValueError("algorithm must be 'lms' or 'nlms'.")
        if not np.isfinite(float(self.step_size)) or self.step_size <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if not np.isfinite(float(self.epsilon)) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        if self.experiment_seed is not None and not isinstance(
            self.experiment_seed,
            (int, np.integer),
        ):
            raise TypeError("experiment_seed must be an integer or None.")


@dataclass(frozen=True)
class IdentificationResult:
    """Histories and final estimate from one secondary-path experiment."""

    excitation: np.ndarray
    measured_response: np.ndarray
    estimated_response: np.ndarray
    response_error: np.ndarray
    squared_error: np.ndarray
    mse_learning_curve: np.ndarray
    coefficient_history: np.ndarray
    secondary_path_estimate: np.ndarray
    config: IdentificationConfig


@dataclass(frozen=True)
class ValidationMetrics:
    """Numerical comparison of a hidden true path and an estimate."""

    impulse_response_error: np.ndarray
    impulse_response_rmse: float
    relative_impulse_error: float
    magnitude_response_rmse_db: float
    phase_response_rmse_radians: float


@dataclass(frozen=True)
class SecondaryPathModel:
    """Portable, versioned secondary-path model artifact.

    The `.npz` archive contains only numeric arrays and a JSON metadata string,
    and is always loaded with ``allow_pickle=False``.
    """

    impulse_response: np.ndarray
    sampling_rate_hz: int
    identification_config: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    format_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "impulse_response",
            _finite_vector(self.impulse_response, name="impulse_response"),
        )
        _positive_integer(self.sampling_rate_hz, name="sampling_rate_hz")
        _positive_integer(self.format_version, name="format_version")

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-compatible metadata describing this model artifact."""

        return {
            "format": "ps26052-anc-secondary-path-model",
            "format_version": self.format_version,
            "sampling_rate_hz": self.sampling_rate_hz,
            "filter_length": int(len(self.impulse_response)),
            "identification_config": self.identification_config,
            "validation_metrics": self.validation_metrics,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        """Save a stable, reloadable `.npz` model artifact."""

        destination = Path(path)
        if destination.suffix != ".npz":
            raise ValueError("SecondaryPathModel artifacts must use a .npz suffix.")
        destination.parent.mkdir(parents=True, exist_ok=True)

        metadata_json = json.dumps(
            self.to_metadata(),
            sort_keys=True,
            separators=(",", ":"),
        )
        np.savez_compressed(
            destination,
            impulse_response=self.impulse_response,
            metadata_json=np.asarray(metadata_json),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "SecondaryPathModel":
        """Load and validate a portable `.npz` model artifact."""

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Secondary-path model not found: {source}")

        with np.load(source, allow_pickle=False) as archive:
            required = {"impulse_response", "metadata_json"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(
                    "Secondary-path model is missing fields: "
                    f"{', '.join(sorted(missing))}."
                )
            impulse_response = archive["impulse_response"]
            metadata_json = str(archive["metadata_json"].item())

        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as error:
            raise ValueError("Secondary-path model metadata is not valid JSON.") from error

        if metadata.get("format") != "ps26052-anc-secondary-path-model":
            raise ValueError("Unsupported secondary-path model format.")

        return cls(
            impulse_response=impulse_response,
            sampling_rate_hz=metadata["sampling_rate_hz"],
            identification_config=metadata.get("identification_config", {}),
            validation_metrics=metadata.get("validation_metrics", {}),
            metadata=metadata.get("metadata", {}),
            format_version=metadata["format_version"],
        )


def identify_secondary_path(
    excitation: np.ndarray | list[float],
    measured_response: np.ndarray | list[float],
    config: IdentificationConfig,
) -> IdentificationResult:
    """Estimate an FIR secondary path from excitation and measured response.

    This routine has intentionally no ``true_path`` parameter, which prevents
    synthetic ground truth from becoming an estimator prior.
    """

    if not isinstance(config, IdentificationConfig):
        raise TypeError("config must be an IdentificationConfig.")

    probe = _finite_vector(excitation, name="excitation")
    measured = _finite_vector(measured_response, name="measured_response")
    if len(probe) != len(measured):
        raise ValueError("excitation and measured_response must have equal length.")

    if config.algorithm == "lms":
        result = LMSFilter(
            filter_length=config.filter_length,
            step_size=config.step_size,
        ).adapt(probe, measured)
        squared_error = result.error ** 2
    else:
        result = NLMSFilter(
            filter_length=config.filter_length,
            step_size=config.step_size,
            epsilon=config.epsilon,
        ).adapt(probe, measured)
        squared_error = result.squared_error

    return IdentificationResult(
        excitation=probe,
        measured_response=measured,
        estimated_response=result.output,
        response_error=result.error,
        squared_error=squared_error,
        mse_learning_curve=moving_average(squared_error, config.learning_window),
        coefficient_history=result.coefficient_history,
        secondary_path_estimate=result.final_coefficients,
        config=config,
    )


def validate_secondary_path_estimate(
    secondary_path_true: np.ndarray | list[float],
    secondary_path_estimate: np.ndarray | list[float],
    *,
    frequency_bins: int = 2_048,
) -> ValidationMetrics:
    """Compare true and estimated paths after identification has completed."""

    true_path = _finite_vector(secondary_path_true, name="secondary_path_true")
    estimate = _finite_vector(
        secondary_path_estimate,
        name="secondary_path_estimate",
    )
    _positive_integer(frequency_bins, name="frequency_bins")

    comparison_length = max(len(true_path), len(estimate))
    padded_true = np.pad(true_path, (0, comparison_length - len(true_path)))
    padded_estimate = np.pad(estimate, (0, comparison_length - len(estimate)))
    impulse_error = padded_estimate - padded_true

    true_response = np.fft.rfft(padded_true, n=frequency_bins)
    estimate_response = np.fft.rfft(padded_estimate, n=frequency_bins)
    floor = np.finfo(np.float64).tiny
    magnitude_difference_db = 20.0 * np.log10(
        np.maximum(np.abs(estimate_response), floor)
        / np.maximum(np.abs(true_response), floor)
    )
    phase_difference = np.angle(
        estimate_response * np.conj(true_response)
    )

    return ValidationMetrics(
        impulse_response_error=impulse_error,
        impulse_response_rmse=float(np.sqrt(np.mean(impulse_error ** 2))),
        relative_impulse_error=float(
            np.linalg.norm(impulse_error)
            / (np.linalg.norm(padded_true) + np.finfo(np.float64).eps)
        ),
        magnitude_response_rmse_db=float(
            np.sqrt(np.mean(magnitude_difference_db ** 2))
        ),
        phase_response_rmse_radians=float(
            np.sqrt(np.mean(phase_difference ** 2))
        ),
    )


def perturb_secondary_path_model(
    impulse_response: np.ndarray | list[float],
    *,
    gain: float = 1.0,
    delay_samples: int = 0,
    coefficient_perturbation: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Create a same-length gain/delay/coefficient-mismatched model path."""

    model = _finite_vector(impulse_response, name="impulse_response")
    model_gain = float(gain)
    if not np.isfinite(model_gain):
        raise ValueError("gain must be finite.")
    if not isinstance(delay_samples, (int, np.integer)) or isinstance(
        delay_samples,
        bool,
    ):
        raise TypeError("delay_samples must be an integer.")

    model *= model_gain
    if delay_samples > 0:
        model = np.concatenate(
            (np.zeros(int(delay_samples), dtype=np.float64), model)
        )[: len(model)]
    elif delay_samples < 0:
        shift = abs(int(delay_samples))
        model = np.concatenate(
            (model[shift:], np.zeros(min(shift, len(model)), dtype=np.float64))
        )[: len(model)]

    if coefficient_perturbation is not None:
        perturbation = _finite_vector(
            coefficient_perturbation,
            name="coefficient_perturbation",
        )
        if len(perturbation) != len(model):
            raise ValueError(
                "coefficient_perturbation length must match impulse_response."
            )
        model += perturbation

    return model


def model_from_identification(
    identification: IdentificationResult,
    validation: ValidationMetrics | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> SecondaryPathModel:
    """Build the hand-off artifact consumed by Module 4 and later modules.

    ``validation`` is optional so a future measured physical path can be
    exported even when a hidden ground-truth impulse response is unavailable.
    """

    if not isinstance(identification, IdentificationResult):
        raise TypeError("identification must be an IdentificationResult.")
    if validation is not None and not isinstance(validation, ValidationMetrics):
        raise TypeError("validation must be a ValidationMetrics or None.")

    validation_dict: dict[str, Any] = {}
    if validation is not None:
        validation_dict = asdict(validation)
        validation_dict["impulse_response_error"] = validation.impulse_response_error.tolist()
    return SecondaryPathModel(
        impulse_response=identification.secondary_path_estimate,
        sampling_rate_hz=identification.config.sampling_rate_hz,
        identification_config=asdict(identification.config),
        validation_metrics=validation_dict,
        metadata=metadata or {},
    )
