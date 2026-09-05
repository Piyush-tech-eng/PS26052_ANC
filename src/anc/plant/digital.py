"""Simulation primitives for a causal, digital feedforward ANC plant.

The plant uses one explicit sign convention throughout::

    d[n]    = P(z) * x[n]
    y[n]    = W(z) * x[n]
    y_s[n]  = S(z) * y[n]
    e[n]    = d[n] + y_s[n]

Consequently, the controller learns a *negative* secondary-path
contribution.  The true secondary path is used only to form the simulated
measurement.  FxLMS and FxNLMS receive the separately supplied model path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anc.adaptive import AdaptiveFIR


_ALGORITHMS = frozenset({"none", "lms", "nlms", "fxlms", "fxnlms"})


def _as_finite_vector(values: np.ndarray | list[float], *, name: str) -> np.ndarray:
    """Return a finite, non-empty one-dimensional float vector."""

    vector = np.asarray(values, dtype=np.float64)

    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(vector) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or Inf.")

    return vector.copy()


def _validate_positive_integer(value: int, *, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def apply_causal_fir(
    signal: np.ndarray | list[float],
    impulse_response: np.ndarray | list[float],
) -> np.ndarray:
    """Apply a zero-initialized causal FIR path without changing length."""

    x = _as_finite_vector(signal, name="signal")
    h = _as_finite_vector(impulse_response, name="impulse_response")

    return np.asarray(
        np.convolve(x, h, mode="full")[: len(x)],
        dtype=np.float64,
    )


def make_delayed_fir_path(
    coefficients: np.ndarray | list[float],
    *,
    delay_samples: int = 0,
    gain: float = 1.0,
) -> np.ndarray:
    """Create a causal FIR path with an explicit leading delay and gain."""

    base = _as_finite_vector(coefficients, name="coefficients")

    if not isinstance(delay_samples, (int, np.integer)) or isinstance(
        delay_samples,
        bool,
    ):
        raise TypeError("delay_samples must be an integer.")
    if delay_samples < 0:
        raise ValueError("delay_samples must not be negative.")

    path_gain = float(gain)
    if not np.isfinite(path_gain):
        raise ValueError("gain must be finite.")

    return np.concatenate(
        (
            np.zeros(int(delay_samples), dtype=np.float64),
            path_gain * base,
        )
    )


def _causal_mean_square(values: np.ndarray, window_size: int) -> np.ndarray:
    squared = values ** 2
    cumulative = np.cumsum(squared, dtype=np.float64)
    output = np.empty_like(squared)

    for index in range(len(squared)):
        start = max(0, index - window_size + 1)
        total = cumulative[index]
        if start:
            total -= cumulative[start - 1]
        output[index] = total / (index - start + 1)

    return output


@dataclass(frozen=True)
class ANCExperimentConfig:
    """Configuration for one reproducible ANC adaptation run."""

    filter_length: int
    step_size: float
    algorithm: str = "fxnlms"
    epsilon: float = 1e-8
    sampling_rate_hz: int = 8_000
    learning_window: int = 256

    def __post_init__(self) -> None:
        _validate_positive_integer(self.filter_length, name="filter_length")
        _validate_positive_integer(self.sampling_rate_hz, name="sampling_rate_hz")
        _validate_positive_integer(self.learning_window, name="learning_window")

        if self.algorithm not in _ALGORITHMS:
            supported = ", ".join(sorted(_ALGORITHMS))
            raise ValueError(
                f"algorithm must be one of: {supported}."
            )

        step_size = float(self.step_size)
        if not np.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be finite and positive.")

        epsilon = float(self.epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive.")


@dataclass(frozen=True)
class ANCExperimentResult:
    """Complete history produced by one digital ANC plant run."""

    reference: np.ndarray
    primary_disturbance: np.ndarray
    controller_output: np.ndarray
    secondary_path_output: np.ndarray
    error: np.ndarray
    filtered_reference: np.ndarray
    coefficient_history: np.ndarray
    final_coefficients: np.ndarray
    primary_path: np.ndarray
    secondary_path_true: np.ndarray
    secondary_path_model_used: np.ndarray
    error_power: np.ndarray
    initial_error_power: float
    final_error_power: float

    @property
    def residual_power_ratio(self) -> float:
        """Final-to-initial error power ratio for this adaptation run."""

        return float(
            self.final_error_power
            / (self.initial_error_power + np.finfo(np.float64).eps)
        )


def run_anc_experiment(
    reference: np.ndarray | list[float],
    primary_path: np.ndarray | list[float],
    secondary_path_true: np.ndarray | list[float],
    secondary_path_model: np.ndarray | list[float],
    config: ANCExperimentConfig,
    *,
    initial_coefficients: np.ndarray | list[float] | None = None,
) -> ANCExperimentResult:
    """Run a causal no-control, direct-LMS, or filtered-x ANC experiment.

    ``secondary_path_true`` is only used to create the measured contribution
    at the error sensor.  The adaptation direction for filtered-x algorithms
    is calculated exclusively from ``secondary_path_model``.
    """

    if not isinstance(config, ANCExperimentConfig):
        raise TypeError("config must be an ANCExperimentConfig.")

    x = _as_finite_vector(reference, name="reference")
    primary = _as_finite_vector(primary_path, name="primary_path")
    secondary_true = _as_finite_vector(
        secondary_path_true,
        name="secondary_path_true",
    )
    secondary_model = _as_finite_vector(
        secondary_path_model,
        name="secondary_path_model",
    )

    if initial_coefficients is not None:
        initial = _as_finite_vector(
            initial_coefficients,
            name="initial_coefficients",
        )
        if len(initial) != config.filter_length:
            raise ValueError(
                "initial_coefficients length must match filter_length."
            )
    else:
        initial = None

    # Reuse the Module 3 tapped-delay-line controller mechanics.  The plant
    # owns only the ANC-specific error measurement and filtered-x direction.
    controller = AdaptiveFIR(
        config.filter_length,
        initial_coefficients=initial,
    )

    num_samples = len(x)
    primary_disturbance = apply_causal_fir(x, primary)
    controller_output = np.empty(num_samples, dtype=np.float64)
    secondary_output = np.empty(num_samples, dtype=np.float64)
    error = np.empty(num_samples, dtype=np.float64)
    filtered_reference = np.empty(num_samples, dtype=np.float64)
    coefficient_history = np.empty(
        (num_samples, config.filter_length),
        dtype=np.float64,
    )

    filtered_reference_state = np.zeros(config.filter_length, dtype=np.float64)
    secondary_state = np.zeros(len(secondary_true), dtype=np.float64)
    model_state = np.zeros(len(secondary_model), dtype=np.float64)

    for index, sample in enumerate(x):
        controller_output[index], controller_state = controller.process_sample(sample)

        if len(secondary_state) > 1:
            secondary_state[1:] = secondary_state[:-1]
        secondary_state[0] = controller_output[index]
        secondary_output[index] = float(np.dot(secondary_true, secondary_state))

        error[index] = primary_disturbance[index] + secondary_output[index]

        if len(model_state) > 1:
            model_state[1:] = model_state[:-1]
        model_state[0] = sample
        filtered_reference[index] = float(np.dot(secondary_model, model_state))

        if config.filter_length > 1:
            filtered_reference_state[1:] = filtered_reference_state[:-1]
        filtered_reference_state[0] = filtered_reference[index]

        if config.algorithm == "lms":
            direction = controller_state
            adaptation_gain = config.step_size
        elif config.algorithm == "nlms":
            direction = controller_state
            adaptation_gain = config.step_size / (
                config.epsilon + float(np.dot(direction, direction))
            )
        elif config.algorithm == "fxlms":
            direction = filtered_reference_state
            adaptation_gain = config.step_size
        elif config.algorithm == "fxnlms":
            direction = filtered_reference_state
            adaptation_gain = config.step_size / (
                config.epsilon + float(np.dot(direction, direction))
            )
        else:
            direction = None
            adaptation_gain = 0.0

        if direction is not None:
            controller.coefficients -= adaptation_gain * error[index] * direction

        coefficient_history[index] = controller.coefficients

    error_power = _causal_mean_square(error, config.learning_window)
    comparison_window = min(1_000, max(1, num_samples // 4))

    return ANCExperimentResult(
        reference=x,
        primary_disturbance=primary_disturbance,
        controller_output=controller_output,
        secondary_path_output=secondary_output,
        error=error,
        filtered_reference=filtered_reference,
        coefficient_history=coefficient_history,
        final_coefficients=controller.coefficients.copy(),
        primary_path=primary,
        secondary_path_true=secondary_true,
        secondary_path_model_used=secondary_model,
        error_power=error_power,
        initial_error_power=float(np.mean(error[:comparison_window] ** 2)),
        final_error_power=float(np.mean(error[-comparison_window:] ** 2)),
    )
