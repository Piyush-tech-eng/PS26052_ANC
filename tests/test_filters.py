import numpy as np

from anc.filters import first_order_lowpass
from anc.filters.basic import fir_filter
from anc.signals.signal import Signal
from scipy.signal import lfilter



def test_lowpass_output_length():
    signal = Signal(
        samples=np.array([1.0, 2.0, 3.0, 4.0]),
        sampling_rate=1000,
    )

    output = first_order_lowpass(signal, alpha=0.5)

    assert output.num_samples == signal.num_samples

def test_lowpass_sampling_rate_preserved():
    signal = Signal(
        samples=np.array([1.0, 2.0, 3.0]),
        sampling_rate=16000,
    )

    output = first_order_lowpass(signal, alpha=0.9)

    assert output.sampling_rate == 16000

    
def test_lowpass_alpha_zero():
    """
    For alpha = 0:

    y[n] = x[n]
    """

    samples = np.array(
        [1.0, 2.0, -1.0, 4.0]
    )

    signal = Signal(
        samples=samples,
        sampling_rate=1000,
    )

    output = first_order_lowpass(
        signal,
        alpha=0.0,
    )

    assert np.allclose(
        output.samples,
        samples,
    )


def test_lowpass_invalid_alpha():
    signal = Signal(
        samples=np.array([1.0, 2.0]),
        sampling_rate=1000,
    )

    try:
        first_order_lowpass(
            signal,
            alpha=1.0,
        )

        assert False

    except ValueError:
        assert True

def test_fir_filter_known_sequence() -> None:
    signal = Signal(
        samples=np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0]
        ),
        sampling_rate=16_000,
    )

    coefficients = np.array(
        [1.0, 0.5, -0.25]
    )

    output = fir_filter(
        signal,
        coefficients,
    )

    expected = np.array(
        [1.0, 2.5, 3.75, 5.0, 6.25]
    )

    np.testing.assert_allclose(
        output.samples,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

def test_fir_matches_scipy() -> None:
    rng = np.random.default_rng(42)

    x = rng.normal(
        size=1000
    )

    h = np.array(
        [0.2, -0.1, 0.4, 0.3]
    )

    signal = Signal(
        samples=x,
        sampling_rate=16_000,
    )

    ours = fir_filter(
        signal,
        h,
    ).samples

    reference = lfilter(
        h,
        [1.0],
        x,
    )

    np.testing.assert_allclose(
        ours,
        reference,
        rtol=1e-12,
        atol=1e-12,
    )

def test_impulse_reproduces_fir_impulse_response() -> None:
    from anc.signals.generators import generate_impulse

    sampling_rate = 1000

    impulse = generate_impulse(
        sampling_rate=sampling_rate,
        duration=0.02,
        index=0,
        amplitude=1.0,
    )

    h = np.array(
        [0.10, 0.20, 0.35, 0.20, 0.10],
        dtype=np.float64,
    )

    output = fir_filter(
        impulse,
        h,
    )

    np.testing.assert_allclose(
        output.samples[:len(h)],
        h,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        output.samples[len(h):],
        0.0,
        rtol=1e-12,
        atol=1e-12,
    )

def test_moving_average_attenuates_high_frequency_more() -> None:
    sampling_rate = 16_000
    duration = 1.0

    low_frequency = 500.0
    high_frequency = 4_000.0

    num_samples = int(
        sampling_rate * duration
    )

    t = (
        np.arange(num_samples)
        / sampling_rate
    )

    samples = (
        np.sin(
            2.0
            * np.pi
            * low_frequency
            * t
        )
        +
        0.5
        * np.sin(
            2.0
            * np.pi
            * high_frequency
            * t
        )
    )

    input_signal = Signal(
        samples=samples,
        sampling_rate=sampling_rate,
        metadata={},
    )

    coefficients = np.ones(5) / 5.0

    output_signal = fir_filter(
        input_signal,
        coefficients,
    )

    def tone_amplitude(
        values: np.ndarray,
        frequency: float,
    ) -> float:
        spectrum = np.fft.rfft(values)

        frequencies = np.fft.rfftfreq(
            len(values),
            d=1.0 / sampling_rate,
        )

        index = np.argmin(
            np.abs(
                frequencies - frequency
            )
        )

        return (
            2.0
            * np.abs(spectrum[index])
            / len(values)
        )

    low_gain = (
        tone_amplitude(
            output_signal.samples,
            low_frequency,
        )
        /
        tone_amplitude(
            input_signal.samples,
            low_frequency,
        )
    )

    high_gain = (
        tone_amplitude(
            output_signal.samples,
            high_frequency,
        )
        /
        tone_amplitude(
            input_signal.samples,
            high_frequency,
        )
    )

    assert low_gain > high_gain

