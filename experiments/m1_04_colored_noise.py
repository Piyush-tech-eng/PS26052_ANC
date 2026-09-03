from pathlib import Path
import json

import numpy as np

from anc.signals import (
    generate_white_noise,
    generate_colored_noise,
)
from anc.visualization import (
    plot_waveform,
    plot_spectrum,
)


def save_signal_results(
    signal,
    results_dir: Path,
    prefix: str,
):
    """
    Save signal samples, metadata, waveform and spectrum.
    """

    np.save(
        results_dir / f"{prefix}_signal.npy",
        signal.samples,
    )

    metadata = {
        "sampling_rate": signal.sampling_rate,
        "num_samples": signal.num_samples,
        "duration_seconds": signal.duration,
        **signal.metadata,
    }

    with open(
        results_dir / f"{prefix}_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    plot_waveform(
        signal,
        results_dir / f"{prefix}_waveform.png",
        title=f"M1.04 — {prefix.title()} Waveform",
    )

    plot_spectrum(
        signal,
        results_dir / f"{prefix}_spectrum.png",
        title=f"M1.04 — {prefix.title()} Spectrum",
    )


def main():

    # ---------------------------------------------
    # Configuration
    # ---------------------------------------------

    sampling_rate = 16000
    duration = 2.0

    std = 1.0
    alpha = 0.95
    seed = 42

    # ---------------------------------------------
    # Generate signals
    # ---------------------------------------------

    white_noise = generate_white_noise(
        sampling_rate=sampling_rate,
        duration=duration,
        std=std,
        seed=seed,
    )

    colored_noise = generate_colored_noise(
        sampling_rate=sampling_rate,
        duration=duration,
        std=std,
        alpha=alpha,
        seed=seed,
    )

    # ---------------------------------------------
    # Results directory
    # ---------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    results_dir = (
        project_root
        / "results"
        / "m1_04_colored_noise"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------
    # Save both signals
    # ---------------------------------------------

    save_signal_results(
        white_noise,
        results_dir,
        "white_noise",
    )

    save_signal_results(
        colored_noise,
        results_dir,
        "colored_noise",
    )

    # ---------------------------------------------
    # Summary
    # ---------------------------------------------

    print("M1.04 — Colored Noise Experiment")
    print("---------------------------------")

    print(
        f"Sampling rate: "
        f"{sampling_rate} Hz"
    )

    print(
        f"Duration: "
        f"{duration} seconds"
    )

    print(
        f"Filter alpha: "
        f"{alpha}"
    )

    print()
    print("Results saved to:")
    print(results_dir)


if __name__ == "__main__":
    main()