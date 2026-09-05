from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anc.adaptive import AdaptiveFIR


def _load_reference_signal(
    path: Path,
) -> np.ndarray:
    """
    Load and validate the frozen Module 1
    reference signal.
    """

    if not path.is_file():
        raise FileNotFoundError(
            "Reference signal was not found: "
            f"{path}"
        )

    try:
        reference = np.load(
            path,
            allow_pickle=False,
        )
    except (
        OSError,
        ValueError,
    ) as error:
        raise ValueError(
            "Could not load reference signal: "
            f"{path}"
        ) from error

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    if reference.ndim != 1:
        raise ValueError(
            "Reference signal must be "
            "one-dimensional."
        )

    if len(reference) == 0:
        raise ValueError(
            "Reference signal must not be empty."
        )

    if not np.isfinite(
        reference
    ).all():
        raise ValueError(
            "Reference signal contains NaN or Inf."
        )

    return reference


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
        / "m3_01_adaptive_fir"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_path = (
        handoff_directory
        / "reference.npy"
    )

    reference = (
        _load_reference_signal(
            reference_path
        )
    )

    # -------------------------------------------------
    # M3.01 configuration
    # -------------------------------------------------

    filter_length = 16

    # Deterministic non-adaptive FIR.
    #
    # The coefficients are normalized so the experiment
    # has a meaningful and stable output.
    initial_coefficients = np.ones(
        filter_length,
        dtype=np.float64,
    )

    initial_coefficients /= (
        np.sum(
            initial_coefficients
        )
    )

    adaptive_filter = AdaptiveFIR(
        filter_length=filter_length,
        initial_coefficients=(
            initial_coefficients
        ),
    )

    num_samples = len(
        reference
    )

    output = np.empty(
        num_samples,
        dtype=np.float64,
    )

    input_vectors = np.empty(
        (
            num_samples,
            filter_length,
        ),
        dtype=np.float64,
    )

    # -------------------------------------------------
    # Process sample by sample
    # -------------------------------------------------

    for index, sample in enumerate(
        reference
    ):

        (
            current_output,
            input_vector,
        ) = adaptive_filter.process_sample(
            sample
        )

        output[index] = (
            current_output
        )

        input_vectors[index] = (
            input_vector
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
        / "output.npy",
        output,
    )

    np.save(
        results_directory
        / "input_vectors.npy",
        input_vectors,
    )

    np.save(
        results_directory
        / "initial_coefficients.npy",
        initial_coefficients,
    )

    np.save(
        results_directory
        / "final_coefficients.npy",
        adaptive_filter.coefficients,
    )

    summary = {
        "module": "M3.01",
        "experiment": (
            "Adaptive FIR core "
            "causal processing"
        ),
        "num_samples": int(
            num_samples
        ),
        "filter_length": int(
            filter_length
        ),
        "adaptation_enabled": False,
        "reference_path": str(
            reference_path
        ),
        "output_mean": float(
            np.mean(
                output
            )
        ),
        "output_std": float(
            np.std(
                output
            )
        ),
        "input_vector_shape": list(
            input_vectors.shape
        ),
        "final_coefficients_changed": bool(
            not np.allclose(
                initial_coefficients,
                adaptive_filter.coefficients,
            )
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
        "M3.01 Adaptive FIR experiment completed."
    )

    print(
        f"Reference samples: "
        f"{num_samples}"
    )

    print(
        f"Filter length: "
        f"{filter_length}"
    )

    print(
        "Adaptation enabled: False"
    )

    print(
        f"Results saved to: "
        f"{results_directory}"
    )


if __name__ == "__main__":
    main()