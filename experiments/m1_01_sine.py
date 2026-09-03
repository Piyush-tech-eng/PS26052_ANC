from pathlib import Path
import json

import numpy as np

from anc.signals.generators import generate_sine
from anc.visualization import (
    plot_waveform,
    plot_spectrum,
)


def main():
    # -------------------------------------------------
    # Experiment configuration
    # -------------------------------------------------

    sampling_rate = 16000
    duration = 2.0
    frequency = 440.0
    amplitude = 1.0

    # -------------------------------------------------
    # Generate signal
    # -------------------------------------------------

    signal = generate_sine(
        sampling_rate=sampling_rate,
        duration=duration,
        frequency=frequency,
        amplitude=amplitude,
    )

    # -------------------------------------------------
    # Result directory
    # -------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    results_dir = (
        project_root
        / "results"
        / "m1_01_sine"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Save raw signal
    # -------------------------------------------------

    np.save(
        results_dir / "signal.npy",
        signal.samples,
    )

    # -------------------------------------------------
    # Save metadata
    # -------------------------------------------------

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

    # -------------------------------------------------
    # Visualizations
    # -------------------------------------------------

    plot_waveform(
        signal,
        results_dir / "waveform.png",
        title="M1.01 — 440 Hz Sine Wave",
    )

    plot_spectrum(
        signal,
        results_dir / "spectrum.png",
        title="M1.01 — Frequency Spectrum",
    )

    # -------------------------------------------------
    # Console summary
    # -------------------------------------------------

    print("M1.01 — Single Tone Experiment")
    print("-------------------------------")

    print(f"Samples:       {signal.num_samples}")
    print(f"Sampling rate: {signal.sampling_rate} Hz")
    print(f"Duration:      {signal.duration:.3f} seconds")

    print()
    print("Results saved to:")
    print(results_dir)


if __name__ == "__main__":
    main()