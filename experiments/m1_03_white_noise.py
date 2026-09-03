from pathlib import Path
import json

import numpy as np

from anc.signals import generate_white_noise
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
    std = 1.0
    seed = 42

    # ---------------------------------------------
    # Generate white noise
    # ---------------------------------------------

    signal = generate_white_noise(
        sampling_rate=sampling_rate,
        duration=duration,
        std=std,
        seed=seed,
    )

    # ---------------------------------------------
    # Result directory
    # ---------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    results_dir = (
        project_root
        / "results"
        / "m1_03_white_noise"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------
    # Save raw signal
    # ---------------------------------------------

    np.save(
        results_dir / "signal.npy",
        signal.samples,
    )

    # ---------------------------------------------
    # Statistics
    # ---------------------------------------------

    measured_mean = float(
        np.mean(signal.samples)
    )

    measured_std = float(
        np.std(signal.samples)
    )

    # ---------------------------------------------
    # Save metadata
    # ---------------------------------------------

    metadata = {
        "sampling_rate": signal.sampling_rate,
        "num_samples": signal.num_samples,
        "duration_seconds": signal.duration,
        "measured_mean": measured_mean,
        "measured_std": measured_std,
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
    # Visualizations
    # ---------------------------------------------

    plot_waveform(
        signal,
        results_dir / "waveform.png",
        title="M1.03 — White Gaussian Noise",
    )

    plot_spectrum(
        signal,
        results_dir / "spectrum.png",
        title="M1.03 — White Noise Spectrum",
    )

    # ---------------------------------------------
    # Summary
    # ---------------------------------------------

    print("M1.03 — White Noise Experiment")
    print("-------------------------------")

    print(
        f"Samples: {signal.num_samples}"
    )

    print(
        f"Sampling rate: "
        f"{signal.sampling_rate} Hz"
    )

    print(
        f"Measured mean: "
        f"{measured_mean:.6f}"
    )

    print(
        f"Measured std: "
        f"{measured_std:.6f}"
    )

    print()
    print("Results saved to:")
    print(results_dir)


if __name__ == "__main__":
    main()