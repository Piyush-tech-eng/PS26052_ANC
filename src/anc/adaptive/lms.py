from __future__ import annotations

import numpy as np

from anc.adaptive.fir import (
    AdaptiveFIR,
    AdaptiveFilterResult,
)


class LMSFilter(AdaptiveFIR):
    """
    Least Mean Squares (LMS) adaptive FIR filter.

    For each sample:

        y[n] = w[n]^T x[n]

        e[n] = d[n] - y[n]

        w[n + 1] = w[n] + mu * e[n] * x[n]

    where:

        x[n] is the causal tapped-delay-line input vector,
        y[n] is the filter output,
        d[n] is the desired signal,
        e[n] is the instantaneous error,
        mu is the LMS step size.
    """

    def __init__(
        self,
        filter_length: int,
        step_size: float,
        *,
        initial_coefficients: np.ndarray | None = None,
    ) -> None:

        super().__init__(
            filter_length,
            initial_coefficients=initial_coefficients,
        )

        mu = float(step_size)

        if not np.isfinite(mu):
            raise ValueError(
                "step_size must be finite."
            )

        if mu <= 0.0:
            raise ValueError(
                "step_size must be positive."
            )

        self.step_size = mu

    def adapt_sample(
        self,
        reference_sample: float,
        desired_sample: float,
    ) -> tuple[float, float, np.ndarray]:
        """
        Process one sample and perform one LMS update.

        Returns
        -------
        output:
            y[n], calculated using w[n].

        error:
            e[n] = d[n] - y[n].

        updated_coefficients:
            w[n + 1].
        """

        desired = float(desired_sample)

        if not np.isfinite(desired):
            raise ValueError(
                "desired_sample must be finite."
            )

        output, input_vector = (
            self.process_sample(
                reference_sample
            )
        )

        error = desired - output

        self.coefficients += (
            self.step_size
            * error
            * input_vector
        )

        return (
            float(output),
            float(error),
            self.coefficients.copy(),
        )

    def adapt(
        self,
        reference: np.ndarray,
        desired: np.ndarray,
    ) -> AdaptiveFilterResult:
        """
        Adapt the LMS filter over complete signals.

        Parameters
        ----------
        reference:
            Reference/input signal x[n].

        desired:
            Desired signal d[n].

        Returns
        -------
        AdaptiveFilterResult
            Contains output history, error history,
            coefficient history, and final coefficients.
        """

        x = np.asarray(
            reference,
            dtype=np.float64,
        )

        d = np.asarray(
            desired,
            dtype=np.float64,
        )

        if x.ndim != 1:
            raise ValueError(
                "reference must be one-dimensional."
            )

        if d.ndim != 1:
            raise ValueError(
                "desired must be one-dimensional."
            )

        if len(x) == 0:
            raise ValueError(
                "reference must not be empty."
            )

        if len(d) == 0:
            raise ValueError(
                "desired must not be empty."
            )

        if len(x) != len(d):
            raise ValueError(
                "reference and desired must "
                "have the same length."
            )

        if not np.isfinite(x).all():
            raise ValueError(
                "reference contains NaN or Inf."
            )

        if not np.isfinite(d).all():
            raise ValueError(
                "desired contains NaN or Inf."
            )

        num_samples = len(x)

        output_history = np.empty(
            num_samples,
            dtype=np.float64,
        )

        error_history = np.empty(
            num_samples,
            dtype=np.float64,
        )

        coefficient_history = np.empty(
            (
                num_samples,
                self.filter_length,
            ),
            dtype=np.float64,
        )

        for index in range(num_samples):

            (
                output,
                error,
                coefficients,
            ) = self.adapt_sample(
                x[index],
                d[index],
            )

            output_history[index] = output

            error_history[index] = error

            coefficient_history[index] = (
                coefficients
            )

        return AdaptiveFilterResult(
            output=output_history,
            error=error_history,
            coefficient_history=coefficient_history,
            final_coefficients=(
                self.coefficients.copy()
            ),
        )