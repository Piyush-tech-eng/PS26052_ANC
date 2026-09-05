from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveFilterResult:
    """
    Result produced by an adaptive filtering run.
    """

    output: np.ndarray
    error: np.ndarray
    coefficient_history: np.ndarray
    final_coefficients: np.ndarray


class AdaptiveFIR:
    """
    Causal tapped-delay-line adaptive FIR filter.

    The input vector at sample n is:

        x_n = [
            x[n],
            x[n-1],
            ...
            x[n-M+1]
        ]

    with zero initial conditions.

    The output is:

        y[n] = w[n]^T x_n

    This class provides the FIR state and history mechanics.
    The adaptation rule itself is supplied by subclasses or
    higher-level algorithms such as LMS and NLMS.
    """

    def __init__(
        self,
        filter_length: int,
        *,
        initial_coefficients: np.ndarray | None = None,
    ) -> None:

        if not isinstance(
            filter_length,
            int,
        ):
            raise TypeError(
                "filter_length must be an integer."
            )

        if filter_length <= 0:
            raise ValueError(
                "filter_length must be positive."
            )

        self.filter_length = filter_length

        if initial_coefficients is None:

            coefficients = np.zeros(
                filter_length,
                dtype=np.float64,
            )

        else:

            coefficients = np.asarray(
                initial_coefficients,
                dtype=np.float64,
            )

            if coefficients.ndim != 1:
                raise ValueError(
                    "initial_coefficients must be "
                    "one-dimensional."
                )

            if len(coefficients) != filter_length:
                raise ValueError(
                    "initial_coefficients length must "
                    "match filter_length."
                )

            if not np.isfinite(
                coefficients
            ).all():
                raise ValueError(
                    "initial_coefficients contains "
                    "NaN or Inf."
                )

            coefficients = coefficients.copy()

        self.coefficients = coefficients

        self.input_state = np.zeros(
            filter_length,
            dtype=np.float64,
        )

    def reset(
        self,
        *,
        coefficients: np.ndarray | None = None,
    ) -> None:
        """
        Reset the delay line and coefficients.

        If coefficients are not supplied, coefficients
        are reset to zero.
        """

        self.input_state.fill(
            0.0
        )

        if coefficients is None:

            self.coefficients.fill(
                0.0
            )

            return

        values = np.asarray(
            coefficients,
            dtype=np.float64,
        )

        if values.ndim != 1:
            raise ValueError(
                "coefficients must be one-dimensional."
            )

        if len(values) != self.filter_length:
            raise ValueError(
                "coefficients length must match "
                "filter_length."
            )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                "coefficients contains NaN or Inf."
            )

        self.coefficients = (
            values.copy()
        )

    def input_vector(
        self,
        sample: float,
    ) -> np.ndarray:
        """
        Insert one new input sample into the causal
        tapped delay line.

        Returns:

            [x[n], x[n-1], ..., x[n-M+1]]
        """

        value = float(sample)

        if not np.isfinite(
            value
        ):
            raise ValueError(
                "Input sample must be finite."
            )

        if self.filter_length > 1:

            self.input_state[1:] = (
                self.input_state[:-1]
            )

        self.input_state[0] = value

        return self.input_state.copy()

    def predict_from_vector(
        self,
        input_vector: np.ndarray,
    ) -> float:
        """
        Compute:

            y[n] = w[n]^T x[n]
        """

        vector = np.asarray(
            input_vector,
            dtype=np.float64,
        )

        if vector.ndim != 1:
            raise ValueError(
                "input_vector must be "
                "one-dimensional."
            )

        if len(vector) != self.filter_length:
            raise ValueError(
                "input_vector length must match "
                "filter_length."
            )

        if not np.isfinite(
            vector
        ).all():
            raise ValueError(
                "input_vector contains NaN or Inf."
            )

        return float(
            np.dot(
                self.coefficients,
                vector,
            )
        )

    def process_sample(
        self,
        sample: float,
    ) -> tuple[float, np.ndarray]:
        """
        Process one sample without adapting coefficients.

        Returns:

            output,
            input_vector
        """

        vector = self.input_vector(
            sample
        )

        output = (
            self.predict_from_vector(
                vector
            )
        )

        return output, vector