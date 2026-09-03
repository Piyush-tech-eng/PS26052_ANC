from pathlib import Path
import json
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_experiment(script_name: str) -> None:
    """
    Run an experiment as an end-to-end integration step.

    If the experiment exits successfully, the corresponding
    Module 1 pipeline stage is considered executable.
    """

    script_path = (
        PROJECT_ROOT
        / "experiments"
        / script_name
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"\nExperiment failed: {script_name}\n"
        f"\nSTDOUT:\n{result.stdout}"
        f"\nSTDERR:\n{result.stderr}"
    )


def test_module1_end_to_end_acceptance() -> None:
    """
    Final Module 1 acceptance test.

    Verifies:

    1. Impulse-response experiment runs.
    2. Low-pass FIR experiment runs.
    3. High-pass FIR experiment runs.
    4. Canonical Module 1 -> Module 2 handoff runs.
    5. Required artifacts are created.
    6. Saved signals are valid.
    7. The handoff relationship is mathematically correct.

        d[n] = x[n] * h[n]
    """

    # -------------------------------------------------
    # Run complete Module 1 FIR demonstration suite
    # -------------------------------------------------

    run_experiment(
        "m1_08_impulse_response.py"
    )

    run_experiment(
        "m1_08_lowpass.py"
    )

    run_experiment(
        "m1_08_highpass.py"
    )

    # -------------------------------------------------
    # Run official Module 1 -> Module 2 handoff
    # -------------------------------------------------

    run_experiment(
        "m1_09_handoff.py"
    )

    # -------------------------------------------------
    # Verify result directories
    # -------------------------------------------------

    results_root = (
        PROJECT_ROOT
        / "results"
    )

    impulse_dir = (
        results_root
        / "m1_08_impulse_response"
    )

    lowpass_dir = (
        results_root
        / "m1_08_lowpass"
    )

    highpass_dir = (
        results_root
        / "m1_08_highpass"
    )

    handoff_dir = (
        results_root
        / "m1_handoff"
    )

    assert impulse_dir.exists()
    assert lowpass_dir.exists()
    assert highpass_dir.exists()
    assert handoff_dir.exists()

    # -------------------------------------------------
    # Verify impulse-response artifacts
    # -------------------------------------------------

    impulse_input = np.load(
        impulse_dir
        / "input_impulse.npy"
    )

    impulse_response = np.load(
        impulse_dir
        / "impulse_response.npy"
    )

    impulse_output = np.load(
        impulse_dir
        / "output.npy"
    )

    assert np.isfinite(
        impulse_input
    ).all()

    assert np.isfinite(
        impulse_response
    ).all()

    assert np.isfinite(
        impulse_output
    ).all()

    np.testing.assert_allclose(
        impulse_output[
            :len(impulse_response)
        ],
        impulse_response,
        rtol=1e-12,
        atol=1e-12,
    )

    # -------------------------------------------------
    # Verify low-pass artifacts
    # -------------------------------------------------

    lowpass_input = np.load(
        lowpass_dir
        / "input.npy"
    )

    lowpass_coefficients = np.load(
        lowpass_dir
        / "coefficients.npy"
    )

    lowpass_output = np.load(
        lowpass_dir
        / "output.npy"
    )

    assert (
        len(lowpass_input)
        == len(lowpass_output)
    )

    assert np.isfinite(
        lowpass_input
    ).all()

    assert np.isfinite(
        lowpass_coefficients
    ).all()

    assert np.isfinite(
        lowpass_output
    ).all()

    # -------------------------------------------------
    # Verify low-pass metadata
    # -------------------------------------------------

    with open(
        lowpass_dir
        / "metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        lowpass_metadata = json.load(
            file
        )

    assert (
        lowpass_metadata[
            "low_frequency_gain"
        ]
        >
        lowpass_metadata[
            "high_frequency_gain"
        ]
    )

    # -------------------------------------------------
    # Verify high-pass artifacts
    # -------------------------------------------------

    highpass_input = np.load(
        highpass_dir
        / "input.npy"
    )

    highpass_coefficients = np.load(
        highpass_dir
        / "coefficients.npy"
    )

    highpass_output = np.load(
        highpass_dir
        / "output.npy"
    )

    assert (
        len(highpass_input)
        == len(highpass_output)
    )

    assert np.isfinite(
        highpass_input
    ).all()

    assert np.isfinite(
        highpass_coefficients
    ).all()

    assert np.isfinite(
        highpass_output
    ).all()

    # -------------------------------------------------
    # Verify high-pass metadata
    # -------------------------------------------------

    with open(
        highpass_dir
        / "metadata.json",
        "r",
        encoding="utf-8",
    ) as file:
        highpass_metadata = json.load(
            file
        )

    assert (
        highpass_metadata[
            "high_frequency_gain"
        ]
        >
        highpass_metadata[
            "low_frequency_gain"
        ]
    )

    # -------------------------------------------------
    # Verify canonical handoff files
    # -------------------------------------------------

    reference_path = (
        handoff_dir
        / "reference.npy"
    )

    desired_path = (
        handoff_dir
        / "desired.npy"
    )

    impulse_response_path = (
        handoff_dir
        / "system_impulse_response.npy"
    )

    metadata_path = (
        handoff_dir
        / "metadata.json"
    )

    assert reference_path.exists()
    assert desired_path.exists()
    assert impulse_response_path.exists()
    assert metadata_path.exists()

    # -------------------------------------------------
    # Load canonical Module 2 dataset
    # -------------------------------------------------

    reference = np.load(
        reference_path
    )

    desired = np.load(
        desired_path
    )

    system_impulse_response = np.load(
        impulse_response_path
    )

    # -------------------------------------------------
    # Basic signal validity
    # -------------------------------------------------

    assert (
        len(reference)
        == len(desired)
    )

    assert len(reference) > 0

    assert len(
        system_impulse_response
    ) > 0

    assert np.isfinite(
        reference
    ).all()

    assert np.isfinite(
        desired
    ).all()

    assert np.isfinite(
        system_impulse_response
    ).all()

    # -------------------------------------------------
    # Mathematical handoff verification
    #
    # d[n] = x[n] * h[n]
    # -------------------------------------------------

    expected_desired = (
        np.convolve(
            reference,
            system_impulse_response,
            mode="full",
        )
        [:len(reference)]
    )

    np.testing.assert_allclose(
        desired,
        expected_desired,
        rtol=1e-12,
        atol=1e-12,
    )

    # -------------------------------------------------
    # Verify handoff metadata
    # -------------------------------------------------

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        handoff_metadata = json.load(
            file
        )

    assert (
        handoff_metadata[
            "dataset"
        ]
        ==
        "module_1_to_module_2_handoff"
    )

    assert (
        handoff_metadata[
            "num_samples"
        ]
        ==
        len(reference)
    )

    assert (
        handoff_metadata[
            "module_2_contract"
        ][
            "reference_file"
        ]
        ==
        "reference.npy"
    )

    assert (
        handoff_metadata[
            "module_2_contract"
        ][
            "desired_file"
        ]
        ==
        "desired.npy"
    )

    # -------------------------------------------------
    # Final acceptance condition
    # -------------------------------------------------

    assert (
        handoff_metadata[
            "verification"
        ][
            "fir_convolution_verified"
        ]
        is True
    )