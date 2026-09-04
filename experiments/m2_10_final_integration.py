from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EPSILON = np.finfo(np.float64).eps


# =========================================================
# Validation / loading
# =========================================================


def load_vector(
    path: Path,
    *,
    name: str,
) -> np.ndarray:
    """Load and validate a finite one-dimensional array."""

    if not path.is_file():
        raise FileNotFoundError(
            f"{name} was not found: {path}"
        )

    try:
        values = np.load(
            path,
            allow_pickle=False,
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Could not load {name}: {path}"
        ) from error

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return values


def load_metadata(
    path: Path,
) -> dict:
    """Load Module 1 handoff metadata."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Metadata file was not found: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            "metadata.json is not valid JSON."
        ) from error

    if not isinstance(metadata, dict):
        raise ValueError(
            "metadata.json must contain a JSON object."
        )

    return metadata


def get_sampling_rate(
    metadata: dict,
) -> float:
    """Extract sampling rate from Module 1 metadata."""

    possible_keys = (
        "sampling_rate_hz",
        "sampling_rate",
        "fs",
    )

    for key in possible_keys:

        if key in metadata:

            value = float(metadata[key])

            if (
                not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    "Sampling rate must be finite "
                    "and positive."
                )

            return value

    raise KeyError(
        "Could not find sampling rate in metadata.json. "
        "Expected one of: "
        f"{possible_keys}"
    )


# =========================================================
# Wiener mathematics
# =========================================================


def build_regression_matrix(
    reference: np.ndarray,
    filter_length: int,
) -> np.ndarray:
    """
    Construct the causal tapped-delay matrix.

    Row n is:

        [x[n], x[n-1], ..., x[n-M+1]]

    Missing past samples are zero-padded.
    """

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    if filter_length <= 0:
        raise ValueError(
            "filter_length must be positive."
        )

    num_samples = len(reference)

    X = np.zeros(
        (
            num_samples,
            filter_length,
        ),
        dtype=np.float64,
    )

    for coefficient_index in range(
        filter_length
    ):

        delay = coefficient_index

        if delay == 0:

            X[:, coefficient_index] = (
                reference
            )

        else:

            X[
                delay:,
                coefficient_index
            ] = reference[:-delay]

    return X


def biased_autocorrelation(
    signal: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """
    Compute biased autocorrelation for non-negative lags.

    Rxx[k] =
        (1 / N)
        sum_{n=k}^{N-1}
        x[n] x[n-k]
    """

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    num_samples = len(signal)

    if max_lag < 0:
        raise ValueError(
            "max_lag must be non-negative."
        )

    max_lag = min(
        max_lag,
        num_samples - 1,
    )

    values = np.empty(
        max_lag + 1,
        dtype=np.float64,
    )

    for lag in range(
        max_lag + 1
    ):

        values[lag] = (
            np.dot(
                signal[lag:],
                signal[:num_samples - lag],
            )
            / num_samples
        )

    return values


def biased_cross_correlation(
    reference: np.ndarray,
    desired: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """
    Compute the Wiener cross-correlation vector sequence.

    Rxd[k] =
        (1 / N)
        sum_{n=k}^{N-1}
        x[n-k] d[n]
    """

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    desired = np.asarray(
        desired,
        dtype=np.float64,
    )

    if len(reference) != len(desired):
        raise ValueError(
            "reference and desired "
            "must have equal lengths."
        )

    num_samples = len(reference)

    max_lag = min(
        max_lag,
        num_samples - 1,
    )

    values = np.empty(
        max_lag + 1,
        dtype=np.float64,
    )

    for lag in range(
        max_lag + 1
    ):

        values[lag] = (
            np.dot(
                reference[:num_samples - lag],
                desired[lag:],
            )
            / num_samples
        )

    return values


def build_wiener_structures(
    regression_matrix: np.ndarray,
    desired: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct:

        R = (X^T X) / N

        p = (X^T d) / N
    """

    num_samples = len(desired)

    R = (
        regression_matrix.T
        @ regression_matrix
    ) / num_samples

    p = (
        regression_matrix.T
        @ desired
    ) / num_samples

    return R, p


def solve_wiener(
    R: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """
    Solve:

        R w_opt = p

    using np.linalg.solve rather than explicitly
    computing R^{-1}.
    """

    try:

        w_opt = np.linalg.solve(
            R,
            p,
        )

    except np.linalg.LinAlgError as error:

        raise np.linalg.LinAlgError(
            "Wiener correlation matrix is singular "
            "or ill-conditioned for direct solution."
        ) from error

    if not np.isfinite(w_opt).all():
        raise ValueError(
            "Wiener solution contains NaN or Inf."
        )

    return np.asarray(
        w_opt,
        dtype=np.float64,
    )


# =========================================================
# Metrics
# =========================================================


def mse(
    values: np.ndarray,
) -> float:
    """Mean squared value."""

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return float(
        np.mean(values ** 2)
    )


def normalized_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Normalized zero-lag correlation."""

    first = np.asarray(
        first,
        dtype=np.float64,
    )

    second = np.asarray(
        second,
        dtype=np.float64,
    )

    denominator = (
        np.linalg.norm(first)
        * np.linalg.norm(second)
    )

    if denominator <= EPSILON:
        return 0.0

    return float(
        np.dot(first, second)
        / denominator
    )


def coefficient_metrics(
    true_coefficients: np.ndarray,
    estimated_coefficients: np.ndarray,
) -> dict[str, float]:
    """Compare true and estimated FIR coefficients."""

    comparison_length = max(
        len(true_coefficients),
        len(estimated_coefficients),
    )

    true_aligned = np.pad(
        true_coefficients,
        (
            0,
            comparison_length
            - len(true_coefficients),
        ),
    )

    estimated_aligned = np.pad(
        estimated_coefficients,
        (
            0,
            comparison_length
            - len(estimated_coefficients),
        ),
    )

    difference = (
        true_aligned
        - estimated_aligned
    )

    difference_mse = float(
        np.mean(
            difference ** 2
        )
    )

    difference_rmse = float(
        np.sqrt(
            difference_mse
        )
    )

    relative_l2_error = float(
        np.linalg.norm(
            difference
        )
        / (
            np.linalg.norm(
                true_aligned
            )
            + EPSILON
        )
    )

    correlation = normalized_correlation(
        true_aligned,
        estimated_aligned,
    )

    return {
        "comparison_length": int(
            comparison_length
        ),
        "coefficient_mse": (
            difference_mse
        ),
        "coefficient_rmse": (
            difference_rmse
        ),
        "relative_l2_error": (
            relative_l2_error
        ),
        "coefficient_correlation": (
            correlation
        ),
        "maximum_absolute_error": float(
            np.max(
                np.abs(
                    difference
                )
            )
        ),
    }


# =========================================================
# Plotting
# =========================================================


def save_signal_plot(
    signal: np.ndarray,
    path: Path,
    *,
    sampling_rate_hz: float,
    title: str,
    ylabel: str = "Amplitude",
) -> None:

    time = (
        np.arange(len(signal))
        / sampling_rate_hz
    )

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        time,
        signal,
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


def save_comparison_plot(
    first: np.ndarray,
    second: np.ndarray,
    path: Path,
    *,
    sampling_rate_hz: float,
    first_label: str,
    second_label: str,
    title: str,
) -> None:

    if len(first) != len(second):
        raise ValueError(
            "Signals must have equal length."
        )

    time = (
        np.arange(len(first))
        / sampling_rate_hz
    )

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        time,
        first,
        label=first_label,
    )

    plt.plot(
        time,
        second,
        label=second_label,
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        title
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


def save_sequence_plot(
    sequence: np.ndarray,
    path: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:

    index = np.arange(
        len(sequence)
    )

    plt.figure(
        figsize=(10, 4)
    )

    plt.plot(
        index,
        sequence,
        marker="o",
        markersize=3,
    )

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


# =========================================================
# Main experiment
# =========================================================


def main() -> None:

    # -------------------------------------------------
    # Project structure
    # -------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    handoff_directory = (
        project_root
        / "results"
        / "m1_handoff"
    )

    results_directory = (
        project_root
        / "results"
        / "m2_10_final_integration"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Load frozen Module 1 handoff
    # -------------------------------------------------

    metadata = load_metadata(
        handoff_directory
        / "metadata.json"
    )

    x = load_vector(
        handoff_directory
        / "reference.npy",
        name="Module 1 reference signal x[n]",
    )

    d = load_vector(
        handoff_directory
        / "desired.npy",
        name="Module 1 desired signal d[n]",
    )

    h_true = load_vector(
        handoff_directory
        / "system_impulse_response.npy",
        name=(
            "Module 1 ground-truth "
            "impulse response h[n]"
        ),
    )

    sampling_rate_hz = (
        get_sampling_rate(
            metadata
        )
    )

    if len(x) != len(d):
        raise ValueError(
            "Module 1 reference and desired "
            "signals must have equal lengths."
        )

    # -------------------------------------------------
    # Wiener filter configuration
    #
    # For this controlled FIR identification experiment,
    # use the true FIR length as the estimator order.
    #
    # M2 does NOT use true coefficients in the solution;
    # only their LENGTH is used as the filter order.
    # -------------------------------------------------

    filter_length = len(
        h_true
    )

    # -------------------------------------------------
    # M2.03 / M2.04
    #
    # Correlation analysis
    # -------------------------------------------------

    rxx = (
        biased_autocorrelation(
            x,
            max_lag=filter_length - 1,
        )
    )

    rxd = (
        biased_cross_correlation(
            x,
            d,
            max_lag=filter_length - 1,
        )
    )

    # -------------------------------------------------
    # M2.05
    #
    # Construct tapped-delay representation
    # and Wiener structures.
    # -------------------------------------------------

    X = build_regression_matrix(
        x,
        filter_length,
    )

    R, p = (
        build_wiener_structures(
            X,
            d,
        )
    )

    # -------------------------------------------------
    # Consistency checks
    #
    # Rxx / Rxd should match the first-column
    # correlation structures generated by X.
    # -------------------------------------------------

    R_first_column = (
        R[:, 0]
    )

    p_from_matrix = p.copy()

    correlation_structure_passed = bool(
        np.allclose(
            R_first_column,
            rxx,
            rtol=1e-10,
            atol=1e-12,
        )
    )

    cross_correlation_structure_passed = bool(
        np.allclose(
            p_from_matrix,
            rxd,
            rtol=1e-10,
            atol=1e-12,
        )
    )

    # -------------------------------------------------
    # M2.06
    #
    # Solve Wiener-Hopf equation.
    #
    # R w_opt = p
    # -------------------------------------------------

    w_opt = solve_wiener(
        R,
        p,
    )

    # -------------------------------------------------
    # M2.07
    #
    # Generate Wiener estimate and error.
    # -------------------------------------------------

    d_hat = (
        X
        @ w_opt
    )

    error = (
        d
        - d_hat
    )

    # -------------------------------------------------
    # M2.08
    #
    # Performance metrics.
    # -------------------------------------------------

    desired_power = mse(
        d
    )

    estimate_power = mse(
        d_hat
    )

    error_power = mse(
        error
    )

    relative_error_power = float(
        error_power
        / (
            desired_power
            + EPSILON
        )
    )

    explained_power_fraction = float(
        1.0
        - relative_error_power
    )

    desired_estimate_correlation = (
        normalized_correlation(
            d,
            d_hat,
        )
    )

    # -------------------------------------------------
    # Orthogonality principle
    #
    # X^T e / N should be approximately zero.
    # -------------------------------------------------

    orthogonality_vector = (
        X.T
        @ error
        / len(error)
    )

    maximum_orthogonality_residual = (
        float(
            np.max(
                np.abs(
                    orthogonality_vector
                )
            )
        )
    )

    orthogonality_l2_norm = float(
        np.linalg.norm(
            orthogonality_vector
        )
    )

    # -------------------------------------------------
    # Wiener equation residual
    # -------------------------------------------------

    wiener_equation_residual = (
        R
        @ w_opt
        - p
    )

    wiener_equation_residual_norm = (
        float(
            np.linalg.norm(
                wiener_equation_residual
            )
        )
    )

    relative_wiener_residual = (
        float(
            wiener_equation_residual_norm
            / (
                np.linalg.norm(p)
                + EPSILON
            )
        )
    )

    # -------------------------------------------------
    # M2.09
    #
    # Compare Wiener estimate with true FIR.
    # -------------------------------------------------

    system_metrics = (
        coefficient_metrics(
            h_true,
            w_opt,
        )
    )

    comparison_length = (
        system_metrics[
            "comparison_length"
        ]
    )

    true_aligned = np.pad(
        h_true,
        (
            0,
            comparison_length
            - len(h_true),
        ),
    )

    estimated_aligned = np.pad(
        w_opt,
        (
            0,
            comparison_length
            - len(w_opt),
        ),
    )

    coefficient_error = (
        true_aligned
        - estimated_aligned
    )

    # -------------------------------------------------
    # Acceptance checks
    # -------------------------------------------------

    finite_values_passed = bool(
        np.isfinite(x).all()
        and np.isfinite(d).all()
        and np.isfinite(h_true).all()
        and np.isfinite(rxx).all()
        and np.isfinite(rxd).all()
        and np.isfinite(R).all()
        and np.isfinite(p).all()
        and np.isfinite(w_opt).all()
        and np.isfinite(d_hat).all()
        and np.isfinite(error).all()
    )

    correlation_matrix_symmetric = bool(
        np.allclose(
            R,
            R.T,
            rtol=1e-10,
            atol=1e-12,
        )
    )

    error_identity_passed = bool(
        np.allclose(
            error,
            d - d_hat,
            rtol=1e-12,
            atol=1e-12,
        )
    )

    wiener_equation_passed = bool(
        np.allclose(
            R @ w_opt,
            p,
            rtol=1e-8,
            atol=1e-10,
        )
    )

    autocorrelation_power_passed = bool(
        np.isclose(
            rxx[0],
            np.mean(
                x ** 2
            ),
            rtol=1e-10,
            atol=1e-12,
        )
    )

    acceptance = {
        "all_values_finite": (
            finite_values_passed
        ),
        "correlation_matrix_symmetric": (
            correlation_matrix_symmetric
        ),
        "autocorrelation_zero_lag_matches_power": (
            autocorrelation_power_passed
        ),
        "correlation_structure_consistent": (
            correlation_structure_passed
        ),
        "cross_correlation_structure_consistent": (
            cross_correlation_structure_passed
        ),
        "wiener_equation_satisfied": (
            wiener_equation_passed
        ),
        "error_identity_satisfied": (
            error_identity_passed
        ),
    }

    overall_passed = all(
        acceptance.values()
    )

    # -------------------------------------------------
    # Save numerical outputs
    # -------------------------------------------------

    np.save(
        results_directory
        / "reference_x.npy",
        x,
    )

    np.save(
        results_directory
        / "desired_d.npy",
        d,
    )

    np.save(
        results_directory
        / "autocorrelation_rxx.npy",
        rxx,
    )

    # Compatibility filename
    np.save(
        results_directory
        / "rxx.npy",
        rxx,
    )

    np.save(
        results_directory
        / "cross_correlation_rxd.npy",
        rxd,
    )

    # Compatibility filename
    np.save(
        results_directory
        / "rxd.npy",
        rxd,
    )

    np.save(
        results_directory
        / "correlation_matrix_R.npy",
        R,
    )

    np.save(
        results_directory
        / "cross_correlation_vector_p.npy",
        p,
    )

    np.save(
        results_directory
        / "wiener_coefficients.npy",
        w_opt,
    )

    # Compatibility filename
    np.save(
        results_directory
        / "w_opt.npy",
        w_opt,
    )

    np.save(
        results_directory
        / "estimated_desired_d_hat.npy",
        d_hat,
    )

    np.save(
        results_directory
        / "error_e.npy",
        error,
    )

    np.save(
        results_directory
        / "true_impulse_response.npy",
        true_aligned,
    )

    np.save(
        results_directory
        / "estimated_impulse_response.npy",
        estimated_aligned,
    )

    np.save(
        results_directory
        / "coefficient_error.npy",
        coefficient_error,
    )

    np.save(
        results_directory
        / "orthogonality_vector.npy",
        orthogonality_vector,
    )

    # -------------------------------------------------
    # Save visualizations
    # -------------------------------------------------

    save_signal_plot(
        x,
        results_directory
        / "reference_waveform.png",
        sampling_rate_hz=sampling_rate_hz,
        title="Reference Signal x[n]",
    )

    save_signal_plot(
        d,
        results_directory
        / "desired_waveform.png",
        sampling_rate_hz=sampling_rate_hz,
        title="Desired Signal d[n]",
    )

    save_comparison_plot(
        d,
        d_hat,
        results_directory
        / "desired_vs_estimate.png",
        sampling_rate_hz=sampling_rate_hz,
        first_label="Desired d[n]",
        second_label="Wiener estimate d_hat[n]",
        title=(
            "Desired Signal vs "
            "Wiener Estimate"
        ),
    )

    save_signal_plot(
        error,
        results_directory
        / "error_waveform.png",
        sampling_rate_hz=sampling_rate_hz,
        title="Wiener Error e[n]",
    )

    save_sequence_plot(
        rxx,
        results_directory
        / "autocorrelation.png",
        xlabel="Lag k",
        ylabel="Rxx[k]",
        title=(
            "Reference Autocorrelation"
        ),
    )

    save_sequence_plot(
        rxd,
        results_directory
        / "cross_correlation.png",
        xlabel="Lag k",
        ylabel="Rxd[k]",
        title=(
            "Reference-Desired "
            "Cross-Correlation"
        ),
    )

    save_sequence_plot(
        w_opt,
        results_directory
        / "wiener_coefficients.png",
        xlabel="Coefficient index",
        ylabel="Coefficient value",
        title=(
            "Optimal Wiener "
            "Filter Coefficients"
        ),
    )

    save_comparison_plot(
        true_aligned,
        estimated_aligned,
        results_directory
        / "true_vs_estimated_system.png",
        sampling_rate_hz=1.0,
        first_label="True FIR h[n]",
        second_label="Wiener estimate w_opt[n]",
        title=(
            "True FIR System vs "
            "Wiener Estimate"
        ),
    )

    # -------------------------------------------------
    # Save final JSON summary
    # -------------------------------------------------

    summary = {
        "module": "M2.10",
        "status": (
            "PASS"
            if overall_passed
            else "FAIL"
        ),
        "purpose": (
            "Final reproducible end-to-end "
            "Wiener/MMSE integration experiment."
        ),
        "configuration": {
            "sampling_rate_hz": (
                sampling_rate_hz
            ),
            "num_samples": (
                len(x)
            ),
            "duration_seconds": (
                len(x)
                / sampling_rate_hz
            ),
            "filter_length": (
                filter_length
            ),
        },
        "mathematical_pipeline": [
            "x[n]",
            "d[n]",
            "Rxx[k]",
            "Rxd[k]",
            "R",
            "p",
            "w_opt",
            "d_hat[n]",
            "e[n]",
            "MSE",
        ],
        "signal_metrics": {
            "desired_power": (
                desired_power
            ),
            "estimate_power": (
                estimate_power
            ),
            "error_power_mse": (
                error_power
            ),
            "relative_error_power": (
                relative_error_power
            ),
            "explained_power_fraction": (
                explained_power_fraction
            ),
            "desired_estimate_correlation": (
                desired_estimate_correlation
            ),
        },
        "wiener_equation": {
            "residual_l2_norm": (
                wiener_equation_residual_norm
            ),
            "relative_residual": (
                relative_wiener_residual
            ),
        },
        "orthogonality": {
            "maximum_absolute_residual": (
                maximum_orthogonality_residual
            ),
            "l2_norm": (
                orthogonality_l2_norm
            ),
        },
        "system_comparison": (
            system_metrics
        ),
        "acceptance": (
            acceptance
        ),
        "outputs": [
            "reference_x.npy",
            "desired_d.npy",
            "autocorrelation_rxx.npy",
            "cross_correlation_rxd.npy",
            "correlation_matrix_R.npy",
            "cross_correlation_vector_p.npy",
            "wiener_coefficients.npy",
            "estimated_desired_d_hat.npy",
            "error_e.npy",
            "true_impulse_response.npy",
            "estimated_impulse_response.npy",
            "coefficient_error.npy",
            "orthogonality_vector.npy",
            "reference_waveform.png",
            "desired_waveform.png",
            "desired_vs_estimate.png",
            "error_waveform.png",
            "autocorrelation.png",
            "cross_correlation.png",
            "wiener_coefficients.png",
            "true_vs_estimated_system.png",
        ],
    }

    summary_path = (
        results_directory
        / "final_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

        file.write("\n")

    # -------------------------------------------------
    # Console report
    # -------------------------------------------------

    print()
    print(
        "M2.10 — Final Wiener/MMSE "
        "Integration"
    )
    print(
        "=" * 55
    )

    print(
        f"Status: "
        f"{summary['status']}"
    )

    print()

    print(
        "Configuration"
    )
    print(
        f"  Sampling rate: "
        f"{sampling_rate_hz:g} Hz"
    )
    print(
        f"  Samples: "
        f"{len(x)}"
    )
    print(
        f"  Filter length: "
        f"{filter_length}"
    )

    print()

    print(
        "Signal Metrics"
    )
    print(
        f"  Desired power: "
        f"{desired_power:.10g}"
    )
    print(
        f"  Estimate power: "
        f"{estimate_power:.10g}"
    )
    print(
        f"  Error MSE: "
        f"{error_power:.10g}"
    )
    print(
        f"  Relative error power: "
        f"{relative_error_power:.10g}"
    )
    print(
        f"  Explained power fraction: "
        f"{explained_power_fraction:.10g}"
    )
    print(
        f"  Desired/estimate correlation: "
        f"{desired_estimate_correlation:.10g}"
    )

    print()

    print(
        "Wiener Validation"
    )
    print(
        f"  Relative Wiener residual: "
        f"{relative_wiener_residual:.10g}"
    )
    print(
        f"  Maximum orthogonality residual: "
        f"{maximum_orthogonality_residual:.10g}"
    )

    print()

    print(
        "True System Comparison"
    )
    print(
        f"  Coefficient RMSE: "
        f"{system_metrics['coefficient_rmse']:.10g}"
    )
    print(
        f"  Relative L2 error: "
        f"{system_metrics['relative_l2_error']:.10g}"
    )
    print(
        f"  Coefficient correlation: "
        f"{system_metrics['coefficient_correlation']:.10g}"
    )

    print()

    print(
        "Acceptance Checks"
    )

    for name, passed in (
        acceptance.items()
    ):

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"  [{status}] "
            f"{name}"
        )

    print()

    print(
        "Results saved to:"
    )
    print(
        results_directory
    )

    if not overall_passed:

        raise RuntimeError(
            "M2.10 acceptance failed. "
            "Inspect final_summary.json."
        )


if __name__ == "__main__":
    main()