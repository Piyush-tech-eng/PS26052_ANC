from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anc.adaptive import LMSFilter


def _load_signal(
    path: Path,
    *,
    signal_name: str,
) -> np.ndarray:
    """
    Load and validate a one-dimensional
    real-valued signal.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"{signal_name} was not found: "
            f"{path}"
        )

    try:
        signal = np.load(
            path,
            allow_pickle=False,
        )
    except (
        OSError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Could not load {signal_name}: "
            f"{path}"
        ) from error

    signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    if signal.ndim != 1:
        raise ValueError(
            f"{signal_name} must be "
            "one-dimensional."
        )

    if len(signal) == 0:
        raise ValueError(
            f"{signal_name} must not be empty."
        )

    if not np.isfinite(
        signal
    ).all():
        raise ValueError(
            f"{signal_name} contains NaN or Inf."
        )

    return signal


def main() -> None:

    project_root = Path(
        __file__
    ).resolve().parents[1]

    handoff_directory = (
        project_root
        / "results"
        / "m1_handoff"
    )

    results_directory = (
        project_root
        / "results"
        / "m3_02_lms"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_path = (
        handoff_directory
        / "reference.npy"
    )

    desired_path = (
        handoff_directory
        / "desired.npy"
    )

    reference = _load_signal(
        reference_path,
        signal_name="Reference signal",
    )

    desired = _load_signal(
        desired_path,
        signal_name="Desired signal",
    )

    if len(reference) != len(
        desired
    ):
        raise ValueError(
            "Reference and desired signals "
            "must have the same length."
        )

    # -------------------------------------------------
    # LMS configuration
    # -------------------------------------------------

    filter_length = 16

    step_size = 0.001

    lms = LMSFilter(
        filter_length=filter_length,
        step_size=step_size,
    )

    # -------------------------------------------------
    # Adaptive learning
    # -------------------------------------------------

    result = lms.adapt(
        reference,
        desired,
    )

    output = result.output
    error = result.error

    coefficient_history = (
        result.coefficient_history
    )

    final_coefficients = (
        result.final_coefficients
    )

    # -------------------------------------------------
    # Performance diagnostics
    # -------------------------------------------------

    squared_error = (
        error ** 2
    )

    mse = float(
        np.mean(
            squared_error
        )
    )

    rmse = float(
        np.sqrt(
            mse
        )
    )

    initial_error_power = float(
        np.mean(
            squared_error[
                : min(
                    1000,
                    len(squared_error),
                )
            ]
        )
    )

    final_error_power = float(
        np.mean(
            squared_error[
                -min(
                    1000,
                    len(squared_error),
                ):
            ]
        )
    )

    # -------------------------------------------------
    # Save artifacts
    # -------------------------------------------------

    np.save(
        results_directory
        / "reference.npy",
        reference,
    )

    np.save(
        results_directory
        / "desired.npy",
        desired,
    )

    np.save(
        results_directory
        / "output.npy",
        output,
    )

    np.save(
        results_directory
        / "error.npy",
        error,
    )

    np.save(
        results_directory
        / "coefficient_history.npy",
        coefficient_history,
    )

    np.save(
        results_directory
        / "final_coefficients.npy",
        final_coefficients,
    )

    np.save(
        results_directory
        / "squared_error.npy",
        squared_error,
    )

    summary = {
        "module": "M3.02",
        "experiment": (
            "LMS adaptive filtering "
            "on Module 1 handoff"
        ),
        "num_samples": int(
            len(reference)
        ),
        "filter_length": int(
            filter_length
        ),
        "step_size": float(
            step_size
        ),
        "mse": mse,
        "rmse": rmse,
        "initial_error_power": (
            initial_error_power
        ),
        "final_error_power": (
            final_error_power
        ),
        "error_power_ratio": float(
            final_error_power
            / (
                initial_error_power
                + np.finfo(
                    np.float64
                ).eps
            )
        ),
        "final_coefficient_norm": float(
            np.linalg.norm(
                final_coefficients
            )
        ),
        "reference_path": str(
            reference_path
        ),
        "desired_path": str(
            desired_path
        ),
    }

    summary_path = (
        results_directory
        / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        "M3.02 LMS experiment completed."
    )

    print(
        f"Samples: "
        f"{len(reference)}"
    )

    print(
        f"Filter length: "
        f"{filter_length}"
    )

    print(
        f"Step size: "
        f"{step_size}"
    )

    print(
        f"MSE: "
        f"{mse:.10g}"
    )

    print(
        f"RMSE: "
        f"{rmse:.10g}"
    )

    print(
        f"Initial error power: "
        f"{initial_error_power:.10g}"
    )

    print(
        f"Final error power: "
        f"{final_error_power:.10g}"
    )

    print(
        f"Results saved to: "
        f"{results_directory}"
    )


if __name__ == "__main__":
    main()