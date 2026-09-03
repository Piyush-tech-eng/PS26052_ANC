from pathlib import Path
import json

import numpy as np

from anc.filters.basic import fir_filter
from anc.signals.generators import generate_impulse
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

    impulse_duration = 0.02
    impulse_index = 0
    impulse_amplitude = 1.0

    # Arbitrary FIR impulse response / system
    coefficients = np.array(
        [
            0.10,
            0.20,
            0.35,
            0.20,
            0.10,
            0.05,
        ],
        dtype=np.float64,
    )

    # -------------------------------------------------
    # Generate impulse input
    # -------------------------------------------------

    impulse = generate_impulse(
        sampling_rate=sampling_rate,
        duration=impulse_duration,
        index=impulse_index,
        amplitude=impulse_amplitude,
    )

    # -------------------------------------------------
    # Apply FIR system
    # -------------------------------------------------

    output = fir_filter(
        impulse,
        coefficients,
    )

    # -------------------------------------------------
    # Represent h[n] as a Signal for visualization
    # -------------------------------------------------

    impulse_response = Signal(
        samples=coefficients,
        sampling_rate=sampling_rate,
        metadata={
            "signal_type": "fir_impulse_response",
            "filter_length": len(coefficients),
        },
    )

    # -------------------------------------------------
    # Verify impulse-response property
    #
    # delta[n] * h[n] = h[n]
    # -------------------------------------------------

    np.testing.assert_allclose(
        output.samples[:len(coefficients)],
        coefficients,
        rtol=1e-12,
        atol=1e-12,
    )

    # -------------------------------------------------
    # Result directory
    # -------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    results_dir = (
        project_root
        / "results"
        / "m1_08_impulse_response"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Save numerical artifacts
    # -------------------------------------------------

    np.save(
        results_dir / "input_impulse.npy",
        impulse.samples,
    )

    np.save(
        results_dir / "impulse_response.npy",
        coefficients,
    )

    np.save(
        results_dir / "output.npy",
        output.samples,
    )

    # -------------------------------------------------
    # Save metadata
    # -------------------------------------------------

    metadata = {
        "experiment": "m1_08_impulse_response",
        "sampling_rate": sampling_rate,
        "input_length": impulse.num_samples,
        "filter_length": len(coefficients),
        "impulse_index": impulse_index,
        "impulse_amplitude": impulse_amplitude,
        "verification": (
            "output[:filter_length] "
            "matches impulse_response"
        ),
        "input_metadata": impulse.metadata,
        "output_metadata": output.metadata,
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
    # Time-domain visualizations
    # -------------------------------------------------

    plot_waveform(
        impulse,
        results_dir / "input_impulse_waveform.png",
        title="M1.08 — Input Unit Impulse x[n]",
    )

    plot_waveform(
        impulse_response,
        results_dir / "impulse_response_waveform.png",
        title="M1.08 — FIR Impulse Response h[n]",
    )

    plot_waveform(
        output,
        results_dir / "output_waveform.png",
        title="M1.08 — Output y[n] = x[n] * h[n]",
    )

    # -------------------------------------------------
    # Frequency-domain visualizations
    # -------------------------------------------------

    plot_spectrum(
        impulse_response,
        results_dir / "impulse_response_spectrum.png",
        title="M1.08 — System Frequency Representation",
    )

    plot_spectrum(
        output,
        results_dir / "output_spectrum.png",
        title="M1.08 — Output Spectrum",
    )

    # -------------------------------------------------
    # Console summary
    # -------------------------------------------------

    print("M1.08 — Impulse Response Experiment")
    print("-----------------------------------")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Input samples: {impulse.num_samples}")
    print(f"FIR length:    {len(coefficients)}")

    print("\nImpulse response h[n]:")
    print(coefficients)

    print("\nFirst output samples:")
    print(output.samples[:len(coefficients)])

    print(
        "\nVerification passed:"
        " delta[n] * h[n] = h[n]"
    )

    print(f"\nResults saved to:\n{results_dir}")


if __name__ == "__main__":
    main()