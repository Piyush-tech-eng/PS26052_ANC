import numpy as np

from anc.signals.signal import Signal


def generate_sine(
        sampling_rate: int,
        duration: float,
        frequency: float,
        amplitude: float = 1.0,
        phase: float = 0.0,
) -> Signal:
    """
    Generate a discrete-time sinusoidal signal.
    x[n] = A * sin(2*pi*f*n/fs + phase)
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be a positive integer.")
    if duration <=0:
        raise ValueError("Duration must be positive.")
    if frequency < 0:
        raise ValueError("Frequency must be non-negative.")
    num_samples = int(sampling_rate * duration)
    n=np.arange(num_samples)

    samples = amplitude * np.sin(
        2 * np.pi * frequency * n / sampling_rate + phase
    )

    return Signal(
        samples=samples,
        sampling_rate=sampling_rate,
        metadata={
            "signal_type": "sine",
            "frequency_hz": frequency,
            "amplitude": amplitude,
            "phase_rad": phase,
        },
    )
def generate_multi_tone(
        sampling_rate: int,
        duration: float,
        frequencies: list[float],
        amplitudes: list[float],
        phases: list[float] | None = None,
) -> Signal:
    """
    Generate a discrete-time multi-tone signal.
    
    x[n] = sum(
        A_i*sin(2*pi*f_i*n/fs + phase_i)
        )"""

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be a positive integer.")
    if duration <=0:
        raise ValueError("Duration must be positive.")
    if len(frequencies) == 0:
        raise ValueError("Frequencies list cannot be empty.")
    if len(frequencies) != len(amplitudes):
        raise ValueError("Frequencies and amplitudes must have the same length.")
    if phases is None:
        phases = [0.0] * len(frequencies)
    if len(phases) != len(frequencies):
        raise ValueError("Phases must have the same length as frequencies.")

    num_samples = int(sampling_rate * duration)

    n = np.arange(num_samples)

    samples = np.zeros(
        num_samples,
        dtype=np.float64,
    )

    for frequency, amplitude, phase in zip(
        frequencies,
        amplitudes,
        phases,
    ):
        samples += amplitude * np.sin(
            2
            * np.pi
            * frequency
            * n
            / sampling_rate
            + phase
        )

    return Signal(
        samples=samples,
        sampling_rate=sampling_rate,
        metadata={
            "signal_type": "multi_tone",
            "frequencies_hz": frequencies,
            "amplitudes": amplitudes,
            "phases_rad": phases,
        },
    )

def generate_white_noise(
        sampling_rate: int,
        duration: float,
        std: float = 1.0,
        seed: int | None = None,        
) -> Signal:
    """
    Generate zero-mean white Gaussian noise.
    
    x[n] ~ N(0, std^2)
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be a positive integer.")
    if duration <=0:
        raise ValueError("Duration must be positive.")
    if std < 0:
        raise ValueError("std must be positive."
        )

    num_samples = int(
        sampling_rate * duration
    )

    rng = np.random.default_rng(seed)

    samples = rng.normal(
        loc=0.0,
        scale=std,
        size=num_samples,
    )

    return Signal(
        samples=samples,
        sampling_rate=sampling_rate,
        metadata={
            "signal_type": "white_noise",
            "std": std,
            "seed": seed,
        },
    )
def generate_colored_noise(
        sampling_rate: int,
        duration: float,
        std: float = 1.0,
        alpha: float = 0.95,
        seed: int | None = None,
) -> Signal:

    from anc.filters import first_order_lowpass 
    """
    Generate colored noise by passing white Gaussian noise
    through a first-order recursive low-pass filter.
    """

    white_noise = generate_white_noise(
        sampling_rate=sampling_rate,
        duration=duration,
        std=std,
        seed=seed,
    )

    colored_noise = first_order_lowpass(
        signal=white_noise,
        alpha=alpha,
    )

    colored_noise.metadata["signal_type"] = "colored_noise"
    return colored_noise

def generate_impulse(
    sampling_rate: int,
    duration: float,
    index: int = 0,
    amplitude: float = 1.0,
) -> Signal:
    """
    Generate a discrete-time impulse.

    x[n] = amplitude * delta[n - index]

    Parameters
    ----------
    sampling_rate:
        Sampling frequency in Hz.

    duration:
        Signal duration in seconds.

    index:
        Sample index at which the impulse occurs.

    amplitude:
        Impulse amplitude.
    """

    if sampling_rate <= 0:
        raise ValueError(
            "sampling_rate must be positive."
        )

    if duration <= 0:
        raise ValueError(
            "duration must be positive."
        )

    num_samples = int(sampling_rate * duration)

    if num_samples <= 0:
        raise ValueError(
            "duration produces zero samples."
        )

    if not 0 <= index < num_samples:
        raise ValueError(
            "index must satisfy "
            "0 <= index < number of samples."
        )

    samples = np.zeros(
        num_samples,
        dtype=np.float64,
    )

    samples[index] = amplitude

    return Signal(
        samples=samples,
        sampling_rate=sampling_rate,
        metadata={
            "signal_type": "impulse",
            "impulse_index": index,
            "amplitude": amplitude,
        },
    )

def generate_sine_burst(
    sampling_rate: int,
    duration: float,
    frequency: float,
    burst_start: float,
    burst_duration: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> Signal:
    """
    Generate a finite-duration sinusoidal burst.

    The sine exists only during:
        burst_start <= t < burst_start + burst_duration

    Outside that interval, the signal is zero.
    """

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")

    if duration <= 0:
        raise ValueError("duration must be positive.")

    if frequency < 0:
        raise ValueError("frequency must be non-negative.")

    if burst_start < 0:
        raise ValueError("burst_start must be non-negative.")

    if burst_duration <= 0:
        raise ValueError("burst_duration must be positive.")

    if burst_start + burst_duration > duration:
        raise ValueError(
            "Burst must fit within the total signal duration."
        )

    num_samples = int(sampling_rate * duration)
    n = np.arange(num_samples)
    t = n / sampling_rate

    active = (
        (t >= burst_start)
        & (t < burst_start + burst_duration)
    )

    samples = np.zeros(num_samples, dtype=np.float64)

    samples[active] = amplitude * np.sin(
        2 * np.pi * frequency * t[active] + phase
    )

    return Signal(
        samples=samples,
        sampling_rate=sampling_rate,
        metadata={
            "signal_type": "sine_burst",
            "frequency_hz": frequency,
            "burst_start_s": burst_start,
            "burst_duration_s": burst_duration,
            "amplitude": amplitude,
            "phase_rad": phase,
        },
    )
