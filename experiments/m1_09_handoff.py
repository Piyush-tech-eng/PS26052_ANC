from pathlib import Path
import json

import numpy as np

from anc.filters.basic import fir_filter
from anc.signals.generators import generate_white_noise
from anc.signals.signal import Signal
from anc.visualization import (
    plot_spectrum,
    plot_waveform,
)


def main() -> None:
    # -------------------------------------------------
    # Experiment configuration
    # -------------------------------------------------

    sampling_rate = 16_000
    duration = 5.0

    random_seed = 42

    # -------------------------------------------------
    # Known FIR system
    #
    # This is the system that maps:
    #
    # x[n] -> d[n]
    # -------------------------------------------------

    system_impulse_response = np.array(
        [
            0.10,
            0.15,
            0.25,
            0.30,
            0.25,
            0.15,
            0.10,
        ],
        dtype=np.float64,
    )

    # -------------------------------------------------
    # Generate reference signal x[n]
    #
    # White noise provides broad spectral excitation,
    # making it suitable for the next Wiener /
    # correlation-based identification module.
    # -------------------------------------------------

    reference = generate_white_noise(
        sampling_rate=sampling_rate,
        duration=duration,
        std=1.0,
        seed=random_seed,
    )

    # -------------------------------------------------
    # Apply known FIR system
    #
    # d[n] = h[n] * x[n]
    # -------------------------------------------------

    desired = fir_filter(
        reference,
        system_impulse_response,
    )

    # -------------------------------------------------
    # Verify basic handoff consistency
    # -------------------------------------------------

    assert (
        reference.num_samples
        == desired.num_samples
    ), "Reference and desired signals must have equal length."

    assert (
        reference.sampling_rate
        == desired.sampling_rate
    ), "Reference and desired sampling rates must match."

    assert np.isfinite(
        reference.samples
    ).all(), "Reference contains NaN or Inf."

    assert np.isfinite(
        desired.samples
    ).all(), "Desired signal contains NaN or Inf."

    # -------------------------------------------------
    # Recompute expected output independently
    #
    # This verifies that the saved relationship is:
    #
    # d[n] = x[n] * h[n]
    # -------------------------------------------------

    expected_output = np.convolve(
        reference.samples,
        system_impulse_response,
        mode="full",
    )[:reference.num_samples]

    np.testing.assert_allclose(
        desired.samples,
        expected_output,
        rtol=1e-12,
        atol=1e-12,
    )

    # -------------------------------------------------
    # Create Signal object for h[n]
    # -------------------------------------------------

    impulse_response_signal = Signal(
        samples=system_impulse_response,
        sampling_rate=sampling_rate,
        metadata={
            "signal_type": "system_impulse_response",
            "system_type": "FIR",
            "filter_length": len(
                system_impulse_response
            ),
        },
    )

    # -------------------------------------------------
    # Result directory
    # -------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    results_dir = (
        project_root
        / "results"
        / "m1_handoff"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Save canonical Module 2 inputs
    # -------------------------------------------------

    np.save(
        results_dir / "reference.npy",
        reference.samples,
    )

    np.save(
        results_dir / "desired.npy",
        desired.samples,
    )

    np.save(
        results_dir
        / "system_impulse_response.npy",
        system_impulse_response,
    )

    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    metadata = {
        "dataset": "module_1_to_module_2_handoff",

        "sampling_rate_hz": sampling_rate,

        "duration_seconds": duration,

        "num_samples": (
            reference.num_samples
        ),

        "reference_signal": {
            "type": "white_noise",
            "random_seed": random_seed,
        },

        "system": {
            "type": "FIR",
            "impulse_response": (
                system_impulse_response.tolist()
            ),
            "filter_length": len(
                system_impulse_response
            ),
        },

        "desired_signal": {
            "definition": (
                "desired[n] = "
                "system_impulse_response "
                "* reference[n]"
            ),
        },

        "module_2_contract": {
            "reference_file": (
                "reference.npy"
            ),
            "desired_file": (
                "desired.npy"
            ),
            "ground_truth_system_file": (
                "system_impulse_response.npy"
            ),
        },

        "verification": {
            "reference_and_desired_same_length": True,
            "sampling_rates_match": True,
            "all_samples_finite": True,
            "fir_convolution_verified": True,
        },
    }

    with open(
        results_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    # -------------------------------------------------
    # Visualizations
    # -------------------------------------------------

    plot_waveform(
        reference,
        results_dir
        / "reference_waveform.png",
        title=(
            "M1 Handoff — "
            "Reference Signal x[n]"
        ),
    )

    plot_waveform(
        desired,
        results_dir
        / "desired_waveform.png",
        title=(
            "M1 Handoff — "
            "Desired Signal d[n]"
        ),
    )

    plot_waveform(
        impulse_response_signal,
        results_dir
        / "impulse_response.png",
        title=(
            "M1 Handoff — "
            "Known FIR Impulse Response h[n]"
        ),
    )

    plot_spectrum(
        reference,
        results_dir
        / "reference_spectrum.png",
        title=(
            "M1 Handoff — "
            "Reference Spectrum"
        ),
    )

    plot_spectrum(
        desired,
        results_dir
        / "desired_spectrum.png",
        title=(
            "M1 Handoff — "
            "Desired Spectrum"
        ),
    )

    # -------------------------------------------------
    # Console summary
    # -------------------------------------------------

    print(
        "M1.09 — Module 1 to Module 2 Handoff"
    )

    print(
        "--------------------------------------"
    )

    print(
        f"Sampling rate: {sampling_rate} Hz"
    )

    print(
        f"Duration: {duration} seconds"
    )

    print(
        f"Number of samples: "
        f"{reference.num_samples}"
    )

    print(
        "\nKnown FIR system h[n]:"
    )

    print(
        system_impulse_response
    )

    print(
        "\nVerified relationship:"
    )

    print(
        "desired[n] = "
        "system_impulse_response "
        "* reference[n]"
    )

    print(
        "\nCanonical artifacts:"
    )

    print(
        f"  {results_dir / 'reference.npy'}"
    )

    print(
        f"  {results_dir / 'desired.npy'}"
    )

    print(
        f"  {results_dir / 'system_impulse_response.npy'}"
    )

    print(
        f"\nResults saved to:\n{results_dir}"
    )


if __name__ == "__main__":
    main()