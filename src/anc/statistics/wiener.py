from __future__ import annotations

import numpy as np


def build_correlation_matrix(
    autocorrelation_values: np.ndarray,
    filter_length: int,
) -> np.ndarray:
    """
    Construct the Wiener input correlation matrix R.

    Given autocorrelation values:

        Rxx[0], Rxx[1], ..., Rxx[M-1]

    construct the Toeplitz matrix:

        R[i, j] = Rxx[|i - j|]

    Parameters
    ----------
    autocorrelation_values:
        One-dimensional array containing non-negative-lag
        autocorrelation values beginning at lag zero.

    filter_length:
        Desired Wiener filter length M.

    Returns
    -------
    R:
        M x M symmetric Toeplitz correlation matrix.
    """

    rxx = np.asarray(
        autocorrelation_values,
        dtype=np.float64,
    )

    if rxx.ndim != 1:
        raise ValueError(
            "Autocorrelation values must be "
            "one-dimensional."
        )

    if len(rxx) == 0:
        raise ValueError(
            "Autocorrelation values must not be empty."
        )

    if not np.isfinite(rxx).all():
        raise ValueError(
            "Autocorrelation values contain NaN or Inf."
        )

    if not isinstance(
        filter_length,
        (int, np.integer),
    ):
        raise ValueError(
            "filter_length must be an integer."
        )

    if filter_length <= 0:
        raise ValueError(
            "filter_length must be positive."
        )

    if len(rxx) < filter_length:
        raise ValueError(
            "Not enough autocorrelation values "
            "for the requested filter length."
        )

    indices = np.arange(
        filter_length
    )

    lag_matrix = np.abs(
        indices[:, None]
        - indices[None, :]
    )

    return rxx[lag_matrix]

def build_cross_correlation_vector(
    cross_correlation_values: np.ndarray,
    filter_length: int,
) -> np.ndarray:
    """
    Construct the Wiener cross-correlation vector p.

    Given:

        Rxd[0], Rxd[1], ..., Rxd[M-1]

    construct:

        p = [
            Rxd[0],
            Rxd[1],
            ...
            Rxd[M-1]
        ]^T
    """

    rxd = np.asarray(
        cross_correlation_values,
        dtype=np.float64,
    )

    if rxd.ndim != 1:
        raise ValueError(
            "Cross-correlation values must be "
            "one-dimensional."
        )

    if len(rxd) == 0:
        raise ValueError(
            "Cross-correlation values must not be empty."
        )

    if not np.isfinite(rxd).all():
        raise ValueError(
            "Cross-correlation values contain NaN or Inf."
        )

    if not isinstance(
        filter_length,
        (int, np.integer),
    ):
        raise ValueError(
            "filter_length must be an integer."
        )

    if filter_length <= 0:
        raise ValueError(
            "filter_length must be positive."
        )

    if len(rxd) < filter_length:
        raise ValueError(
            "Not enough cross-correlation values "
            "for the requested filter length."
        )

    return rxd[:filter_length].copy()

def solve_wiener_hopf(
    correlation_matrix: np.ndarray,
    cross_correlation_vector: np.ndarray,
) -> np.ndarray:
    """
    Solve the Wiener-Hopf equation:

        R @ w_opt = p

    without explicitly computing R^{-1}.

    Parameters
    ----------
    correlation_matrix:
        Square Wiener input correlation matrix R.

    cross_correlation_vector:
        Wiener cross-correlation vector p.

    Returns
    -------
    w_opt:
        Optimal Wiener filter coefficient vector.
    """

    R = np.asarray(
        correlation_matrix,
        dtype=np.float64,
    )

    p = np.asarray(
        cross_correlation_vector,
        dtype=np.float64,
    )

    # -----------------------------------------
    # Validate R
    # -----------------------------------------

    if R.ndim != 2:
        raise ValueError(
            "Correlation matrix must be two-dimensional."
        )

    num_rows, num_columns = R.shape

    if num_rows != num_columns:
        raise ValueError(
            "Correlation matrix must be square."
        )

    if num_rows == 0:
        raise ValueError(
            "Correlation matrix must not be empty."
        )

    if not np.isfinite(R).all():
        raise ValueError(
            "Correlation matrix contains NaN or Inf."
        )

    # -----------------------------------------
    # Validate p
    # -----------------------------------------

    if p.ndim != 1:
        raise ValueError(
            "Cross-correlation vector must be one-dimensional."
        )

    if len(p) != num_rows:
        raise ValueError(
            "Cross-correlation vector length must match "
            "the correlation matrix dimension."
        )

    if not np.isfinite(p).all():
        raise ValueError(
            "Cross-correlation vector contains NaN or Inf."
        )

    # -----------------------------------------
    # Solve:
    #
    #     R @ w_opt = p
    #
    # Do NOT explicitly compute inv(R).
    # -----------------------------------------

    try:
        w_opt = np.linalg.solve(
            R,
            p,
        )

    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Failed to solve the Wiener-Hopf equation. "
            "The correlation matrix may be singular."
        ) from error

    return np.asarray(
        w_opt,
        dtype=np.float64,
    )