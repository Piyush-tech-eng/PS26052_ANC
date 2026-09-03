from pathlib import Path
import json

import numpy as np

from anc.filters.basic import fir_filter
from anc.signals.signal import Signal
from anc.visualization import (
    plot_waveform,
    plot_spectrum,
)


def get_tone_amplitude(
    samples: np.ndarray,
    sampling_rate: int,
    frequency: float,
) -> float:
    """
    Estimate the amplitude of a sinusoidal component
    at a known frequency using FFT magnitude.
    """

    num_samples = len(samples)

    spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(
        num_samples,
        d=1.0 / sampling_rate,
    )

    index = np.argmin(
        np.abs(frequencies - frequency)
    )

    return (
        2.0
        * np.abs(spectrum[index])
        / num_samples
    )


def main() -> None:
    # -------------------------------------------------
    # Experiment configuration
    # -------------------------------------------------

    sampling_rate = 16_000
    duration = 1.0

    low_frequency = 500.0
    high_frequency = 4_000.0

    low_amplitude = 1.0
    high_amplitude = 0.5

    filter_length = 5

    # -------------------------------------------------
    # Generate mixed-tone input
    #
    # x[n] =
    # low-frequency tone
    # +
    # high-frequency tone
    # -------------------------------------------------

    num_samples = int(
        sampling_rate * duration
    )

    n = np.arange(num_samples)

    t = n / sampling_rate

    input_samples = (
        low_amplitude
        * np.sin(
            2.0
            * np.pi
            * low_frequency
            * t
        )
        +
        high_amplitude
        * np.sin(
            2.0
            * np.pi
            * high_frequency
            * t
        )
    )

    input_signal = Signal(
        samples=input_samples,
        sampling_rate=sampling_rate,
        metadata={
            "signal_type": "mixed_tone",
            "low_frequency_hz": low_frequency,
            "high_frequency_hz": high_frequency,
            "low_amplitude": low_amplitude,
            "high_amplitude": high_amplitude,
        },
    )

    # -------------------------------------------------
    # Define moving-average FIR
    #
    # h[n] = 1 / L
    # -------------------------------------------------

    coefficients = np.ones(
        filter_length,
        dtype=np.float64,
    ) / filter_length

    # -------------------------------------------------
    # Apply FIR
    # -------------------------------------------------

    output_signal = fir_filter(
        input_signal,
        coefficients,
    )

    # -------------------------------------------------
    # Quantitative frequency verification
    # -------------------------------------------------

    input_low_amplitude = get_tone_amplitude(
        input_signal.samples,
        sampling_rate,
        low_frequency,
    )

    input_high_amplitude = get_tone_amplitude(
        input_signal.samples,
        sampling_rate,
        high_frequency,
    )

    output_low_amplitude = get_tone_amplitude(
        output_signal.samples,
        sampling_rate,
        low_frequency,
    )

    output_high_amplitude = get_tone_amplitude(
        output_signal.samples,
        sampling_rate,
        high_frequency,
    )

    low_gain = (
        output_low_amplitude
        / input_low_amplitude
    )

    high_gain = (
        output_high_amplitude
        / input_high_amplitude
    )

    # -------------------------------------------------
    # Basic low-pass verification
    #
    # The low-frequency component should experience
    # less attenuation than the high-frequency component.
    # -------------------------------------------------

    assert low_gain > high_gain, (
        "Low-pass verification failed: "
        "low-frequency gain should exceed "
        "high-frequency gain."
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
        / "m1_08_lowpass"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Save numerical artifacts
    # -------------------------------------------------

    np.save(
        results_dir / "input.npy",
        input_signal.samples,
    )

    np.save(
        results_dir / "coefficients.npy",
        coefficients,
    )

    np.save(
        results_dir / "output.npy",
        output_signal.samples,
    )

    # -------------------------------------------------
    # Save metadata
    # -------------------------------------------------

    metadata = {
        "experiment": "m1_08_lowpass",
        "sampling_rate": sampling_rate,
        "duration_seconds": duration,
        "num_samples": num_samples,
        "low_frequency_hz": low_frequency,
        "high_frequency_hz": high_frequency,
        "low_amplitude": low_amplitude,
        "high_amplitude": high_amplitude,
        "filter_type": (
            "moving_average_lowpass"
        ),
        "filter_length": filter_length,
        "coefficients": (
            coefficients.tolist()
        ),
        "input_low_amplitude": (
            float(input_low_amplitude)
        ),
        "input_high_amplitude": (
            float(input_high_amplitude)
        ),
        "output_low_amplitude": (
            float(output_low_amplitude)
        ),
        "output_high_amplitude": (
            float(output_high_amplitude)
        ),
        "low_frequency_gain": (
            float(low_gain)
        ),
        "high_frequency_gain": (
            float(high_gain)
        ),
        "verification": (
            "low_frequency_gain > "
            "high_frequency_gain"
        ),
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
    # Waveform visualizations
    # -------------------------------------------------

    plot_waveform(
        input_signal,
        results_dir / "input_waveform.png",
        title=(
            "M1.08B — Input: "
            "500 Hz + 4000 Hz"
        ),
    )

    plot_waveform(
        output_signal,
        results_dir / "output_waveform.png",
        title=(
            "M1.08B — Low-Pass FIR Output"
        ),
    )

    # -------------------------------------------------
    # Spectrum visualizations
    # -------------------------------------------------

    plot_spectrum(
        input_signal,
        results_dir / "input_spectrum.png",
        title=(
            "M1.08B — Input Spectrum"
        ),
    )

    plot_spectrum(
        output_signal,
        results_dir / "output_spectrum.png",
        title=(
            "M1.08B — Low-Pass FIR Output Spectrum"
        ),
    )

    # -------------------------------------------------
    # Console summary
    # -------------------------------------------------

    print(
        "M1.08B — Low-Pass FIR Experiment"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Sampling rate: {sampling_rate} Hz"
    )

    print(
        f"Duration: {duration} seconds"
    )

    print(
        f"Low frequency: {low_frequency} Hz"
    )

    print(
        f"High frequency: {high_frequency} Hz"
    )

    print(
        f"\nFIR coefficients:\n{coefficients}"
    )

    print(
        f"\nLow-frequency gain: "
        f"{low_gain:.6f}"
    )

    print(
        f"High-frequency gain: "
        f"{high_gain:.6f}"
    )

    print(
        "\nVerification passed:"
    )

    print(
        "Low-frequency gain is greater "
        "than high-frequency gain."
    )

    print(
        f"\nResults saved to:\n{results_dir}"
    )


if __name__ == "__main__":
    main()