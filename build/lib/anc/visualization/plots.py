from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anc.signals.signal import Signal


def plot_waveform(
        signal: Signal,
        output_path: str | Path,
        title: str = "Signal Waveform",
) -> None:
    """
    Plot a discrete-time signal in the time domain.
    """


    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12,4))

    plt.plot(
        signal.time, 
        signal.samples,
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)

    plt.close()

def plot_spectrum(
    signal: Signal,
    output_path: str | Path,
    title: str = "Signal Spectrum",
) -> None:
    """
    Plot the single-sided magnitude spectrum of a signal.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = signal.samples
    sampling_rate = signal.sampling_rate

    num_samples = len(samples)

    spectrum = np.fft.rfft(samples)

    frequencies = np.fft.rfftfreq(
        num_samples,
        d=1 / sampling_rate,
    )

    magnitude = np.abs(spectrum) / num_samples

    plt.figure(figsize=(12, 4))

    plt.plot(
        frequencies,
        magnitude,
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()
    