from pathlib import Path

import numpy as np

from anc.signals.generators import generate_sine_burst
from anc.visualization.plots import plot_waveform, plot_spectrum

def main() -> None:
    signal = generate_sine_burst(
        sampling_rate=16_000,
        duration=1.0,
        frequency=1000.0,
        burst_start=0.35,
        burst_duration=0.10,
        amplitude=1.0,
    )

    output_dir = Path("results/m1_07_burst")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "signal.npy", signal.samples)

    plot_waveform(
        signal,
        output_dir / "waveform.png",
        title="1 kHz Sine Burst",
    )

    plot_spectrum(
        signal,
        output_dir / "spectrum.png",
        title="1 kHz Sine Burst Spectrum",
    )

    print(signal.metadata)


if __name__ == "__main__":
    main()