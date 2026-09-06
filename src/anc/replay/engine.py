"""End-to-end ANC replay using prepared recorded signals.

Module 6.04 connects the standardized replay inputs produced by Module
6.03 to the existing digital ANC plant implemented in Modules 1-5.

The replay engine deliberately does not reimplement LMS, NLMS, FxLMS,
FxNLMS, or secondary-path processing. Those responsibilities remain in
anc.plant.digital.

A replay recording provides two signals:

    reference
        Signal used by the ANC controller.

    measured
        Recorded/measured disturbance at the error location.

The existing ANC plant normally generates its disturbance internally as:

    d[n] = P(z) * x[n]

For recorded replay, the measured signal is instead treated as the
disturbance history against which the controller is evaluated.

The controller output still propagates through the true secondary path:

    y_s[n] = S(z) * y[n]

and the replay residual is:

    e[n] = d_measured[n] + y_s[n]

The adaptation direction is computed using the supplied secondary-path
model, consistent with the existing ANC implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anc.io import ANCReplayInputs
from anc.plant.digital import (
    ANCExperimentConfig,
)


def _finite_vector(
    values: np.ndarray | list[float],
    *,
    name: str,
) -> np.ndarray:
    """Validate and return a finite one-dimensional float64 vector."""

    vector = np.asarray(
        values,
        dtype=np.float64,
    )

    if vector.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if len(vector) == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.isfinite(
        vector
    ).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return vector.copy()


def _causal_mean_square(
    values: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """Compute causal moving mean-square history."""

    squared = values ** 2

    cumulative = np.cumsum(
        squared,
        dtype=np.float64,
    )

    output = np.empty_like(
        squared
    )

    for index in range(
        len(values)
    ):
        start = max(
            0,
            index
            - window_size
            + 1,
        )

        total = cumulative[
            index
        ]

        if start > 0:
            total -= cumulative[
                start - 1
            ]

        output[index] = (
            total
            / (
                index
                - start
                + 1
            )
        )

    return output


@dataclass(frozen=True)
class ANCReplayResult:
    """Complete result of one recorded-signal ANC replay."""

    reference: np.ndarray

    measured_disturbance: np.ndarray

    controller_output: np.ndarray

    secondary_path_output: np.ndarray

    residual: np.ndarray

    filtered_reference: np.ndarray

    coefficient_history: np.ndarray

    final_coefficients: np.ndarray

    secondary_path_true: np.ndarray

    secondary_path_model_used: np.ndarray

    residual_power: np.ndarray

    initial_residual_power: float

    final_residual_power: float

    sampling_rate_hz: int

    alignment_applied: bool

    delay_samples: int | None

    @property
    def residual_power_ratio(
        self,
    ) -> float:
        """Return final-to-initial residual power ratio."""

        return float(
            self.final_residual_power
            / (
                self.initial_residual_power
                + np.finfo(
                    np.float64
                ).eps
            )
        )

    @property
    def attenuation_db(
        self,
    ) -> float:
        """Return initial-to-final residual attenuation in dB."""

        return float(
            10.0
            * np.log10(
                (
                    self.initial_residual_power
                    + np.finfo(
                        np.float64
                    ).eps
                )
                / (
                    self.final_residual_power
                    + np.finfo(
                        np.float64
                    ).eps
                )
            )
        )


def run_anc_replay(
    replay_inputs: ANCReplayInputs,
    secondary_path_true: np.ndarray | list[float],
    secondary_path_model: np.ndarray | list[float],
    config: ANCExperimentConfig,
    *,
    initial_coefficients: (
        np.ndarray
        | list[float]
        | None
    ) = None,
) -> ANCReplayResult:
    """Run ANC adaptation using prepared recorded replay signals.

    Parameters
    ----------
    replay_inputs:
        Equal-length reference and measured recordings prepared by
        Module 6.03.

    secondary_path_true:
        Physical/simulated secondary path used to propagate controller
        output to the error location.

    secondary_path_model:
        Model used for filtered-x adaptation.

    config:
        Existing ANC algorithm configuration.

    initial_coefficients:
        Optional initial controller coefficients.

    Notes
    -----
    Unlike ``run_anc_experiment()``, this replay function does not
    generate the disturbance through a primary path. The measured
    recording itself is the disturbance:

        d[n] = measured_disturbance[n]
    """

    if not isinstance(
        replay_inputs,
        ANCReplayInputs,
    ):
        raise TypeError(
            "replay_inputs must be an "
            "ANCReplayInputs instance."
        )

    if not isinstance(
        config,
        ANCExperimentConfig,
    ):
        raise TypeError(
            "config must be an "
            "ANCExperimentConfig."
        )

    sampling_rate_hz = (
        replay_inputs.reference
        .sampling_rate_hz
    )

    if (
        config.sampling_rate_hz
        != sampling_rate_hz
    ):
        raise ValueError(
            "config sampling_rate_hz must match "
            "the replay input sampling rate."
        )

    reference = _finite_vector(
        replay_inputs.reference.samples,
        name="reference",
    )

    measured = _finite_vector(
        replay_inputs.measured.samples,
        name="measured",
    )

    if len(reference) != len(
        measured
    ):
        raise ValueError(
            "reference and measured replay "
            "signals must have equal lengths."
        )

    secondary_true = _finite_vector(
        secondary_path_true,
        name="secondary_path_true",
    )

    secondary_model = _finite_vector(
        secondary_path_model,
        name="secondary_path_model",
    )

    if initial_coefficients is None:

        coefficients = np.zeros(
            config.filter_length,
            dtype=np.float64,
        )

    else:

        coefficients = _finite_vector(
            initial_coefficients,
            name=(
                "initial_coefficients"
            ),
        )

        if len(coefficients) != (
            config.filter_length
        ):
            raise ValueError(
                "initial_coefficients length must "
                "match config.filter_length."
            )

    num_samples = len(
        reference
    )

    # -------------------------------------------------
    # Controller state
    # -------------------------------------------------

    controller_state = np.zeros(
        config.filter_length,
        dtype=np.float64,
    )

    # -------------------------------------------------
    # Secondary-path state
    # -------------------------------------------------

    secondary_state = np.zeros(
        len(
            secondary_true
        ),
        dtype=np.float64,
    )

    # -------------------------------------------------
    # Secondary-path model state
    # -------------------------------------------------

    model_state = np.zeros(
        len(
            secondary_model
        ),
        dtype=np.float64,
    )

    filtered_reference_state = np.zeros(
        config.filter_length,
        dtype=np.float64,
    )

    # -------------------------------------------------
    # Histories
    # -------------------------------------------------

    controller_output = np.empty(
        num_samples,
        dtype=np.float64,
    )

    secondary_output = np.empty(
        num_samples,
        dtype=np.float64,
    )

    residual = np.empty(
        num_samples,
        dtype=np.float64,
    )

    filtered_reference = np.empty(
        num_samples,
        dtype=np.float64,
    )

    coefficient_history = np.empty(
        (
            num_samples,
            config.filter_length,
        ),
        dtype=np.float64,
    )

    # -------------------------------------------------
    # Sample-by-sample ANC replay
    # -------------------------------------------------

    for index, sample in enumerate(
        reference
    ):

        # Causal controller input vector:
        #
        # [x[n], x[n-1], ...]
        #
        if (
            config.filter_length
            > 1
        ):
            controller_state[1:] = (
                controller_state[:-1]
            )

        controller_state[0] = sample

        # Controller output:
        #
        # y[n] = w[n]^T x[n]
        #
        controller_output[index] = float(
            np.dot(
                coefficients,
                controller_state,
            )
        )

        # True secondary path:
        #
        # y_s[n] = S(z) * y[n]
        #
        if len(
            secondary_state
        ) > 1:
            secondary_state[1:] = (
                secondary_state[:-1]
            )

        secondary_state[0] = (
            controller_output[index]
        )

        secondary_output[index] = float(
            np.dot(
                secondary_true,
                secondary_state,
            )
        )

        # Replay residual:
        #
        # e[n] = measured[n] + y_s[n]
        #
        residual[index] = (
            measured[index]
            + secondary_output[index]
        )

        # Secondary-path model filtering:
        #
        # x_f[n] = S_hat(z) * x[n]
        #
        if len(
            model_state
        ) > 1:
            model_state[1:] = (
                model_state[:-1]
            )

        model_state[0] = sample

        filtered_reference[index] = float(
            np.dot(
                secondary_model,
                model_state,
            )
        )

        # Build causal filtered-x vector.
        if (
            config.filter_length
            > 1
        ):
            filtered_reference_state[
                1:
            ] = (
                filtered_reference_state[
                    :-1
                ]
            )

        filtered_reference_state[0] = (
            filtered_reference[index]
        )

        # -------------------------------------------------
        # Adaptation direction
        # -------------------------------------------------

        if (
            config.algorithm
            == "none"
        ):

            direction = None

            adaptation_gain = 0.0

        elif (
            config.algorithm
            == "lms"
        ):

            direction = (
                controller_state
            )

            adaptation_gain = (
                config.step_size
            )

        elif (
            config.algorithm
            == "nlms"
        ):

            direction = (
                controller_state
            )

            adaptation_gain = (
                config.step_size
                / (
                    config.epsilon
                    + float(
                        np.dot(
                            direction,
                            direction,
                        )
                    )
                )
            )

        elif (
            config.algorithm
            == "fxlms"
        ):

            direction = (
                filtered_reference_state
            )

            adaptation_gain = (
                config.step_size
            )

        elif (
            config.algorithm
            == "fxnlms"
        ):

            direction = (
                filtered_reference_state
            )

            adaptation_gain = (
                config.step_size
                / (
                    config.epsilon
                    + float(
                        np.dot(
                            direction,
                            direction,
                        )
                    )
                )
            )

        else:

            raise RuntimeError(
                "Unsupported algorithm reached "
                "replay loop."
            )

        # Existing ANC sign convention:
        #
        # w[n + 1]
        # =
        # w[n]
        # -
        # mu * e[n] * direction
        #
        if direction is not None:

            coefficients -= (
                adaptation_gain
                * residual[index]
                * direction
            )

        coefficient_history[index] = (
            coefficients
        )

    # -------------------------------------------------
    # Power analysis
    # -------------------------------------------------

    residual_power = (
        _causal_mean_square(
            residual,
            config.learning_window,
        )
    )

    comparison_window = min(
        1_000,
        max(
            1,
            num_samples
            // 4,
        ),
    )

    initial_residual_power = float(
        np.mean(
            residual[
                :comparison_window
            ]
            ** 2
        )
    )

    final_residual_power = float(
        np.mean(
            residual[
                -comparison_window:
            ]
            ** 2
        )
    )

    return ANCReplayResult(
        reference=reference.copy(),
        measured_disturbance=(
            measured.copy()
        ),
        controller_output=(
            controller_output
        ),
        secondary_path_output=(
            secondary_output
        ),
        residual=residual,
        filtered_reference=(
            filtered_reference
        ),
        coefficient_history=(
            coefficient_history
        ),
        final_coefficients=(
            coefficients.copy()
        ),
        secondary_path_true=(
            secondary_true.copy()
        ),
        secondary_path_model_used=(
            secondary_model.copy()
        ),
        residual_power=residual_power,
        initial_residual_power=(
            initial_residual_power
        ),
        final_residual_power=(
            final_residual_power
        ),
        sampling_rate_hz=(
            sampling_rate_hz
        ),
        alignment_applied=(
            replay_inputs.alignment_applied
        ),
        delay_samples=(
            replay_inputs.delay_samples
        ),
    )