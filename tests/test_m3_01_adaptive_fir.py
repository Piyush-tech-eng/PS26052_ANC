import numpy as np
import pytest

from anc.adaptive import AdaptiveFIR


def test_initial_coefficients_are_zero() -> None:

    adaptive_filter = AdaptiveFIR(
        filter_length=4,
    )

    np.testing.assert_allclose(
        adaptive_filter.coefficients,
        np.zeros(4),
    )


def test_custom_initial_coefficients() -> None:

    coefficients = np.array(
        [1.0, 2.0, 3.0],
    )

    adaptive_filter = AdaptiveFIR(
        filter_length=3,
        initial_coefficients=coefficients,
    )

    np.testing.assert_allclose(
        adaptive_filter.coefficients,
        coefficients,
    )


def test_tapped_delay_line() -> None:

    adaptive_filter = AdaptiveFIR(
        filter_length=4,
    )

    _, vector_0 = (
        adaptive_filter.process_sample(
            1.0
        )
    )

    _, vector_1 = (
        adaptive_filter.process_sample(
            2.0
        )
    )

    _, vector_2 = (
        adaptive_filter.process_sample(
            3.0
        )
    )

    np.testing.assert_allclose(
        vector_0,
        [1.0, 0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(
        vector_1,
        [2.0, 1.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(
        vector_2,
        [3.0, 2.0, 1.0, 0.0],
    )


def test_prediction_matches_dot_product() -> None:

    adaptive_filter = AdaptiveFIR(
        filter_length=3,
        initial_coefficients=np.array(
            [1.0, 0.5, -0.25],
        ),
    )

    output, vector = (
        adaptive_filter.process_sample(
            2.0
        )
    )

    expected = np.dot(
        np.array(
            [1.0, 0.5, -0.25],
        ),
        vector,
    )

    assert output == pytest.approx(
        expected
    )


def test_causal_sequence_processing() -> None:

    adaptive_filter = AdaptiveFIR(
        filter_length=3,
        initial_coefficients=np.array(
            [1.0, 0.5, -0.25],
        ),
    )

    input_signal = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )

    outputs = []

    for sample in input_signal:

        output, _ = (
            adaptive_filter.process_sample(
                sample
            )
        )

        outputs.append(
            output
        )

    expected = np.array(
        [
            1.0,
            2.5,
            3.75,
            5.0,
            6.25,
        ]
    )

    np.testing.assert_allclose(
        outputs,
        expected,
    )


def test_invalid_filter_length() -> None:

    with pytest.raises(
        ValueError
    ):

        AdaptiveFIR(
            filter_length=0,
        )


def test_invalid_initial_coefficient_length() -> None:

    with pytest.raises(
        ValueError
    ):

        AdaptiveFIR(
            filter_length=3,
            initial_coefficients=np.array(
                [1.0, 2.0],
            ),
        )


def test_non_finite_input_rejected() -> None:

    adaptive_filter = AdaptiveFIR(
        filter_length=3,
    )

    with pytest.raises(
        ValueError
    ):

        adaptive_filter.process_sample(
            np.nan
        )


def test_reset() -> None:

    adaptive_filter = AdaptiveFIR(
        filter_length=3,
        initial_coefficients=np.array(
            [1.0, 2.0, 3.0],
        ),
    )

    adaptive_filter.process_sample(
        10.0
    )

    adaptive_filter.reset()

    np.testing.assert_allclose(
        adaptive_filter.coefficients,
        [0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(
        adaptive_filter.input_state,
        [0.0, 0.0, 0.0],
    )