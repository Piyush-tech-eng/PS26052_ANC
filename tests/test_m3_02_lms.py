import numpy as np
import pytest 

from anc.adaptive import LMSFilter


def test_single_lms_update() -> None:

    lms = LMSFilter(
        filter_length = 2,
        step_size = 0.1,
    )

    output, error, coefficients = (
        lms.adapt_sample(
            reference_sample = 2.0,
            desired_sample = 1.0,
        )
    )

    #Initial coefficients are zero.
    assert output == pytest.approx(0.0)

    # e = d - y = 1 - 0.
    assert error == pytest.approx(1.0)

    #Input vector is [2, 0].
    # w_new = [0, 0] + 0.1 * 1 * [2, 0] = [0.2, 0].
    np.testing.assert_allclose(
        coefficients,
        [0.2, 0.0],
    )

def test_second_lms_update() -> None:

    lms = LMSFilter(
        filter_length=2,
        step_size=0.1,
    )

    lms.adapt_sample(
        reference_sample=2.0,
        desired_sample=1.0,
    )

    output, error, coefficients = (
        lms.adapt_sample(
            reference_sample=3.0,
            desired_sample=2.0,
        )
    )

    # Delay-line vector is [3, 2].
    #
    # Current coefficients are [0.2, 0].
    #
    # y = 0.2 * 3 + 0 * 2 = 0.6
    assert output == pytest.approx(
        0.6
    )

    # e = 2 - 0.6 = 1.4
    assert error == pytest.approx(
        1.4
    )

    # w_new =
    #
    # [0.2, 0]
    # + 0.1 * 1.4 * [3, 2]
    #
    # = [0.62, 0.28]
    np.testing.assert_allclose(
        coefficients,
        [0.62, 0.28],
    )


def test_batch_output_shapes() -> None:

    lms = LMSFilter(
        filter_length=3,
        step_size=0.01,
    )

    reference = np.array(
        [1.0, 2.0, 3.0, 4.0],
    )

    desired = np.array(
        [0.5, 1.0, 1.5, 2.0],
    )

    result = lms.adapt(
        reference,
        desired,
    )

    assert result.output.shape == (
        4,
    )

    assert result.error.shape == (
        4,
    )

    assert result.coefficient_history.shape == (
        4,
        3,
    )

    assert result.final_coefficients.shape == (
        3,
    )


def test_coefficients_change_during_learning() -> None:

    lms = LMSFilter(
        filter_length=2,
        step_size=0.05,
    )

    reference = np.array(
        [1.0, 2.0, 3.0, 4.0],
    )

    desired = np.array(
        [1.0, 2.0, 3.0, 4.0],
    )

    result = lms.adapt(
        reference,
        desired,
    )

    assert not np.allclose(
        result.final_coefficients,
        np.zeros(2),
    )


def test_invalid_step_size_zero() -> None:

    with pytest.raises(
        ValueError
    ):

        LMSFilter(
            filter_length=3,
            step_size=0.0,
        )


def test_invalid_step_size_negative() -> None:

    with pytest.raises(
        ValueError
    ):

        LMSFilter(
            filter_length=3,
            step_size=-0.1,
        )


def test_invalid_step_size_nan() -> None:

    with pytest.raises(
        ValueError
    ):

        LMSFilter(
            filter_length=3,
            step_size=np.nan,
        )


def test_mismatched_signal_lengths() -> None:

    lms = LMSFilter(
        filter_length=2,
        step_size=0.1,
    )

    with pytest.raises(
        ValueError
    ):

        lms.adapt(
            np.array(
                [1.0, 2.0],
            ),
            np.array(
                [1.0],
            ),
        )


def test_non_finite_reference_rejected() -> None:

    lms = LMSFilter(
        filter_length=2,
        step_size=0.1,
    )

    with pytest.raises(
        ValueError
    ):

        lms.adapt(
            np.array(
                [1.0, np.nan],
            ),
            np.array(
                [1.0, 2.0],
            ),
        )


def test_non_finite_desired_rejected() -> None:

    lms = LMSFilter(
        filter_length=2,
        step_size=0.1,
    )

    with pytest.raises(
        ValueError
    ):

        lms.adapt(
            np.array(
                [1.0, 2.0],
            ),
            np.array(
                [1.0, np.inf],
            ),
        )


def test_lms_learns_simple_gain() -> None:

    lms = LMSFilter(
        filter_length=1,
        step_size=0.01,
    )

    reference = np.ones(
        500,
        dtype=np.float64,
    )

    desired = 2.0 * reference

    result = lms.adapt(
        reference,
        desired,
    )

    assert result.final_coefficients[0] == (
        pytest.approx(
            2.0,
            abs=0.02,
        )
    )

    assert np.mean(
        result.error[-50:] ** 2
    ) < 0.01