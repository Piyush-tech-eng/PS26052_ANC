from pathlib import Path
import json

import numpy as np

from anc.signals import generate_multi_tone
from anc.visualization import (
    plot_waveform,
    plot_spectrum,
)


def main():

    # ---------------------------------------------
    # Experiment configuration
    # ---------------------------------------------

    sampling_rate = 16000
    duration = 2.0

    frequencies = [
        200.0,
        440.0,
        1000.0,
    ]

    amplitudes = [
        1.0,
        0.7,
        0.4,
    ]

    # ---------------------------------------------
    # Generate signal
    # ---------------------------------------------

    signal = generate_multi_tone(
        sampling_rate=sampling_rate,
        duration=duration,
        frequencies=frequencies,
        amplitudes=amplitudes,
    )

    # ---------------------------------------------
    # Result directory
    # ---------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    results_dir = (
        project_root
        / "results"
        / "m1_02_multi_tone"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------
    # Save signal
    # ---------------------------------------------

    np.save(
        results_dir / "signal.npy",
        signal.samples,
    )

    # ---------------------------------------------
    # Save metadata
    # ---------------------------------------------

    metadata = {
        "sampling_rate": signal.sampling_rate,
        "num_samples": signal.num_samples,
        "duration_seconds": signal.duration,
        **signal.metadata,
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

    # ---------------------------------------------
    # Visualization
    # ---------------------------------------------

    plot_waveform(
        signal,
        results_dir / "waveform.png",
        title="M1.02 — Multi-Tone Signal",
    )

    plot_spectrum(
        signal,
        results_dir / "spectrum.png",
        title="M1.02 — Multi-Tone Spectrum",
    )

    # ---------------------------------------------
    # Summary
    # ---------------------------------------------

    print("M1.02 — Multi-Tone Experiment")
    print("-------------------------------")

    print(
        f"Sampling rate: "
        f"{signal.sampling_rate} Hz"
    )

    print(
        f"Duration: "
        f"{signal.duration:.3f} seconds"
    )

    print(
        f"Number of samples: "
        f"{signal.num_samples}"
    )

    print(
        f"Frequencies: "
        f"{frequencies} Hz"
    )

    print()
    print("Results saved to:")
    print(results_dir)


if __name__ == "__main__":
    main()