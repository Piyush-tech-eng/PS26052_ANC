from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anc.adaptive.fir import AdaptiveFIR


@dataclass(frozen=True)
class NLMSResult:
    """
    Results produced by the NLMS adaptive filter.
    """

    output: np.ndarray
    error: np.ndarray
    squared_error: np.ndarray
    coefficient_history: np.ndarray
    final_coefficients: np.ndarray


class NLMSFilter(AdaptiveFIR):
    """
    Normalized Least Mean Squares adaptive FIR filter.

    The coefficient update is:

        w[n + 1]
        =
        w[n]
        +
        mu / (epsilon + ||x[n]||^2)
        *
        e[n]
        *
        x[n]

    where:

        x[n] = current input vector
        e[n] = desired[n] - output[n]
        mu   = normalized step size
    """

    def __init__(
        self,
        *,
        filter_length: int,
        step_size: float,
        epsilon: float = 1e-8,
        initial_coefficients: np.ndarray | None = None,
    ) -> None:

        super().__init__(
            filter_length,
            initial_coefficients=initial_coefficients,
        )

        if not np.isfinite(
            step_size
        ):
            raise ValueError(
                "step_size must be finite."
            )

        if step_size <= 0:
            raise ValueError(
                "step_size must be positive."
            )

        if not np.isfinite(
            epsilon
        ):
            raise ValueError(
                "epsilon must be finite."
            )

        if epsilon <= 0:
            raise ValueError(
                "epsilon must be positive."
            )

        self.step_size = float(
            step_size
        )

        self.epsilon = float(
            epsilon
        )

    def adapt(
        self,
        reference: np.ndarray,
        desired: np.ndarray,
    ) -> NLMSResult:
        """
        Adapt the filter using reference and desired
        signals.

        Parameters
        ----------
        reference:
            Input/reference signal.

        desired:
            Desired signal.

        Returns
        -------
        NLMSResult
            Output, error, squared error, coefficient
            history, and final coefficients.
        """

        reference = np.asarray(
            reference,
            dtype=np.float64,
        )

        desired = np.asarray(
            desired,
            dtype=np.float64,
        )

        if reference.ndim != 1:
            raise ValueError(
                "reference must be one-dimensional."
            )

        if desired.ndim != 1:
            raise ValueError(
                "desired must be one-dimensional."
            )

        if len(reference) == 0:
            raise ValueError(
                "reference must not be empty."
            )

        if len(desired) == 0:
            raise ValueError(
                "desired must not be empty."
            )

        if len(reference) != len(
            desired
        ):
            raise ValueError(
                "reference and desired must have "
                "the same length."
            )

        if not np.isfinite(
            reference
        ).all():
            raise ValueError(
                "reference contains NaN or Inf."
            )

        if not np.isfinite(
            desired
        ).all():
            raise ValueError(
                "desired contains NaN or Inf."
            )

        num_samples = len(
            reference
        )

        output = np.zeros(
            num_samples,
            dtype=np.float64,
        )

        error = np.zeros(
            num_samples,
            dtype=np.float64,
        )

        squared_error = np.zeros(
            num_samples,
            dtype=np.float64,
        )

        coefficient_history = np.zeros(
            (
                num_samples,
                self.filter_length,
            ),
            dtype=np.float64,
        )

        for index in range(
            num_samples
        ):

            # The causal tapped delay line and FIR prediction are shared
            # with LMS and filtered-x controllers through AdaptiveFIR.
            output[index], input_vector = self.process_sample(
                reference[index]
            )

            # Instantaneous error.
            error[index] = (
                desired[index]
                - output[index]
            )

            squared_error[index] = (
                error[index] ** 2
            )

            # Input-vector energy:
            #
            # ||x[n]||^2
            #
            input_energy = float(
                np.dot(
                    input_vector,
                    input_vector,
                )
            )

            # Normalized LMS update.
            normalized_step = (
                self.step_size
                / (
                    self.epsilon
                    + input_energy
                )
            )

            self.coefficients += (
                normalized_step
                * error[index]
                * input_vector
            )

            coefficient_history[
                index
            ] = self.coefficients

        return NLMSResult(
            output=output,
            error=error,
            squared_error=squared_error,
            coefficient_history=(
                coefficient_history
            ),
            final_coefficients=(
                self.coefficients.copy()
            ),
        )
